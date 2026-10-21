# Orin Roofline and Bottleneck Analysis

Why is VAD slow on NVIDIA AGX Orin, what actually helped, and what headroom is left? This document is an **evidence-led** analysis: every claim is anchored to a measured number from the result JSONs in this repository (`x86/eval/results/`, `arm64/eval/results/`) or to the parent-project development reports, with caveats stated where comparisons are not apples-to-apples.

## 1. Workload characterization

VAD's compute graph has three structurally different parts:

| Part | Dominant ops | Latency character |
|---|---|---|
| Backbone (ResNet50 + FPN) | dense convolutions | compute-bound at FP16 |
| BEV encoder (6 layers) | MSDeformAttn spatial cross-attn + temporal self-attn | memory-hierarchy-bound (L1/L2 gathers + activation streams; DRAM measured at ~⅓ utilization, §3.2) |
| Head decoders (det/map/motion/ego) | deformable/standard attention + FFN | same — see §3.2 kernel census |

Measured module breakdown on x86 (PyTorch profiling, 10 samples, `x86/eval/results/vad_module_profile.json`, VAD-Base):

| Module | Latency | Share of 229.4 ms E2E |
|---|---|---|
| Backbone | 94.7 ms | 41.3% |
| BEV encoder (6 layers) | 101.6 ms | 44.3% |
| Head total (all decoders) | 132.1 ms | 57.6%* |
| — spatial cross-attn (per layer) | 10.0–11.3 ms | dominant per-layer cost |
| — temporal self-attn (per layer) | 3.4–3.8 ms | secondary |

\* The module sums exceed the E2E time because of profiling overlap and synchronization effects; treat the percentages as indicative, not additive.

Conclusion: **~85% of end-to-end time is in the transformer-style encoder + decoders**, with the spatial cross-attention (SCA) the single dominant per-layer cost. This classification drives everything below.

## 2. Orin hardware roofline inputs

Development target platform: NVIDIA Jetson AGX Orin 64 GB (Ampere, `sm_87`). Official specifications (NVIDIA Jetson AGX Orin technical brief and product page):

- GPU: 2048 CUDA cores + 64 Tensor Cores, up to 1.3 GHz
- FP32: 5.3 TFLOPS; INT8 Tensor Core: up to 170 TOPS sparse (**~85 TOPS dense**, FP16 correspondingly ~85 sparse / ~42.5 dense TFLOPS — dense figures derived as sparse/2)
- Memory: 64 GB LPDDR5, **204.8 GB/s**
- 2× NVDLA v2.0 (never used by this pipeline), 15–60 W configurable module

Measured achievable peaks on the deployment Orin (MAXN; `arm64/eval/microbench_orin.py`, results in `arm64/eval/results/orin_microbench_maxn.json`): **170 GB/s** DRAM copy (83% of datasheet), **21.7 TFLOPS** FP16 GEMM (8192³; ~51% of the dense-peak derivation), **10.5 TOPS** INT8 GEMM via `torch._int_mm` — slower than FP16 in that specific path (a torch/cuBLASLt kernel-selection artifact on sm_87, not the platform INT8 ceiling: TRT IMMA convs reach 1.78× on the backbone, §3.1) — and a **28 µs** per-kernel launch+execute floor (481 kernels per head inference ⇒ ~13.5 ms fixed cost per frame).

Two consequences for this workload:

1. The INT8:FP16 dense peak ratio is **2×**. Any measured INT8 speedup above 2× on a module signals that something other than pure arithmetic throughput changed (memory traffic, kernel selection, plugin version).
2. The attention workload's activation traffic puts real pressure on Orin's memory subsystem, but measured DRAM utilization stays at ~⅓ during head inference (§3.2) — the binding constraint is the memory hierarchy (L1/L2/latency), which INT8 arithmetic cannot relieve.

## 3. What the measurements say

> **Power-mode convention:** the module tables in §3.1–§3.4 are the parent-project measurements taken in **50 W mode** (that data only exists at 50 W); **this repo's latency convention is MAXN** — current headline numbers per mode: [power_modes.md](power_modes.md) §4.1–§4.3. The relative conclusions (INT8 pays on the backbone, ~1.0x on the head, structure ≫ precision) are power-mode-independent.

### 3.1 Backbone: compute-bound, INT8 pays

| Engine (Orin, 50 W mode) | Latency | INT8 vs FP16 |
|---|---|---|
| Backbone FP16 | 85.4 ms | — |
| Backbone INT8 | 47.9 ms | **1.78×** |

The backbone is conv-dominated and compute-bound at FP16, so it converts most of the 2× INT8 arithmetic advantage into wall-clock speedup. It is also **calibration-tolerant**: a pure-conv stack produced clean results even with early, careless calibration (principal caveat: the shipped engines use real-data calibration anyway — §3.3). A later TRT 10.4 rebuild improved it further to 30.9 ms (parent-project reports).

### 3.2 Head: the kernel mix is FP16-pinned, so INT8 pays almost nothing

| Engine (Orin, 50 W mode) | Latency | INT8 vs FP16 |
|---|---|---|
| Head FP16 | 694.1 ms | — |
| Head INT8 (matched-era build) | 678.8 ms | **1.02×** |

A ~1.0× INT8 speedup used to look like the signature of a module that is **memory-bound at both precisions** — halving arithmetic cost changes nothing if time is spent moving activations. Direct profiling says the truth is sharper: the INT8-quantized head engine **never executes an INT8 kernel**. The recipe quantizes Conv/Gemm only; attention MatMuls stay FP16 by design (§3.3), the MSDeformAttn plugin has no INT8 implementation, and conv is ≈0% of head time — so INT8's 2× arithmetic advantage has no dominant kernel to accelerate. What the head's ~291 ms actually is (full-inference kernel census of an Orin-built INT8-quantized head engine — real-data PTQ calibration, MatMul/Softmax kept FP16 by design — ncu `--set basic`, 481 kernels; `arm64/eval/results/orin_head_kernel_census.json`):

| Kernel class (full-inference census, VAD-Base INT8-quantized head, MAXN) | Share | n | SM% | Mem% | Why it stays FP16 |
|---|---|---|---|---|---|
| MSDeformAttn plugin (`opt8`, `__half`) | 38.9% | 19 | 64.3 | 64.7 | plugin has no INT8 implementation |
| TRT-fused elementwise (`__myl_*`) | 38.0% | 307 | 17.0 | 56.8 | sits between FP16-pinned boundaries: plugin outputs, attention MatMuls, LayerNorm |
| FP16 GEMMs (MatMul + Gemm) | 21.3% | 147 | 42.7 | 50.4 | MatMuls deliberately FP16 (§3.3); remaining Gemm tactics selected FP16 |
| softmax + reformat + other | 1.7% | 7 | — | — | — |

Per-kernel view: the single largest kernel is the deform plugin's `ms_deformable_im2col_gpu_kernel_opt8` — **219.5 ms of 566.7 ms captured (n=13)** — followed by `trt_ampere_h16816gemm_*` (54.8 ms, n=17) and the `__myl_*` fused-elementwise family (~150 ms across ~290 launches). Profiling methodology caveat: ncu serializes kernels and flushes caches per replay, so captured absolutes run ~2× the 291.5 ms CUDA-event wall time at MAXN — **treat shares as meaningful, absolutes as inflated** (recorded in `orin_head_kernel_census.json`). The census is for the INT8-quantized head recipe; the FP16 verification head differs only in tactic choice, not structure, so the FP16-pinned mix carries over.

Per-kernel SOL characterization: the deform kernel's SM 64.3% / Mem 64.7% means neither pipe is saturated — it is **latency/occupancy-bound** (gather-heavy deformable sampling across FPN levels), not DRAM-saturated (a DRAM-walled kernel shows Mem% pinned >90%). The `__myl_*` family runs at SM 17% / Mem 57% — partially memory-pipe-pressured within the cache hierarchy. Combined with the platform measurement that DRAM sits at ~⅓ utilization while the GPU is 99% busy during head inference (tegrastats EMC 32–35% p50 at MAXN/50 W), the correct model is **memory-hierarchy pressure (L1/L2/activation streams), not a DRAM wall**. Corroborating cross-mode evidence: MAXN→50 W halves the GPU clock while EMC stays 3199 MHz, yet head time grows 282.5→396.1 ms (+40%) — a DRAM-pinned workload would be insensitive to the GPU clock cap.

This is also why the x86 profile shows the **larger** head share on Orin (head ≈ 89% of Orin E2E in the FP16 baseline): the transformer parts degrade fastest relative to the compute-bound backbone as platform arithmetic throughput drops, while their memory-hierarchy/latency sensitivity is exactly what INT8 cannot fix.

> **Comparison caveat.** A later VAD-Base INT8 head build reached 302.2 ms by combining the vectorized (`opt8`) MSDeformAttn plugin with OptLevel=5 and build-profile tuning. That number is **not comparable** to the 694.1 ms FP16 figure: the FP16 engine was never rebuilt with the same plugin/build improvements, so plugin and builder effects are confounded with the precision change. The 1.02× figure above compares *matched-era* builds and is the honest estimate of what PTQ-INT8 alone buys on the head.

### 3.3 Quantization boundary conditions (the three fixes)

The head's INT8 behavior is bounded by three findings from the INT8 bring-up (parent-project reports; consequences visible in the x86 result JSONs):

1. **Real-data calibration is mandatory for the head.** Random-noise calibration put activation ranges off by ~100× and produced catastrophic trajectories: L2(1s) = **4.359** (`x86/eval/results/eval_int8_bb_int8_head.json` — cautionary example; its 23.7 ms latency is meaningless for a broken-accuracy engine). Real-data calibration restored L2(1s) to **0.7805** (`eval_int8_fixed.json`). The backbone, in contrast, is calibration-insensitive.
2. **MHA MatMuls stay FP16.** Attention `MatMul`s are kept in FP16 (per NVIDIA guidance for transformer attention); quantizing them destabilizes accuracy. Consequence: even a perfect INT8 head only quantizes Conv/Gemm projections, capping the reachable arithmetic advantage. The kernel census (§3.2) confirms the end state on-device: zero INT8 kernels execute in the final INT8-quantized head engine.
3. **PREFER vs OBEY precision constraints.** Custom plugins that lack INT8 implementations must use `PREFER`-class precision constraints; `OBEY`-class forcing fails the build.

### 3.4 The dominant lever was structure, not precision

The same chain applied to VAD-Small (4× fewer BEV queries, 2× fewer layers, 4× fewer FPN levels — see `docs/model_variants.md`):

| Module (Orin INT8, 50 W) | VAD-Base | VAD-Small | Reduction |
|---|---|---|---|
| Backbone | 47.9 ms | 25.6 ms | 1.87× |
| Head | 678.8 ms | 35.8 ms | **19.0×** |
| E2E | 726.7 ms | 61.3 ms | **11.9×** |

> Re-benchmark note (2026-10-20): fresh re-measurement of the same small engines: **63.7 ms (MAXN) / 87.4 ms (50 W)** — [power_modes.md](power_modes.md) §4.2.

Cutting activation volume (4× queries × 2× layers = 8× activation traffic; up to 32× arithmetic reduction in SCA) delivered ~19× on the head at 50 W (7.6× at MAXN) where precision tuning delivered ~1.0×. Smaller model structure is also the only lever that improved both latency *and* the deployment envelope at once.

### 3.5 The NaN bug boundary (plugin optimization)

The vectorized `opt8` MSDeformAttn plugin (4×`__half2` per thread) carries a **non-deterministic NaN bug at VAD-Base size on Orin** (~55% of runs; the parent project's `vadbase_opt5_cross_cos_sim_v2` measurement records `cos_sim_avg: NaN`). VAD-Small (single FPN level) is unaffected — 50 consecutive stress runs clean. Practical consequence: the fast 302.2 ms VAD-Base head build is latency-valid but **not production-usable**; VAD-Base production INT8 is now this repo's scalar-plugin OptLevel-3 PTQ build (**E2E 326.7 ms at MAXN**, [power_modes.md](power_modes.md) §4.3), while VAD-Small ships the vectorized plugin safely.

## 4. Precision ladder (x86, VAD-Base, 173-frame eval)

All from `x86/eval/results/*.json`. Metric: planning L2(1s) vs ground truth (lower is better; gate = within ~0.01 of the PyTorch baseline), latency = end-to-end per frame.

| Configuration | L2(1s) | Latency | Note |
|---|---|---|---|
| PyTorch FP32 (reference) | 0.7300 | 233.3 ms | `eval_final_20260612.json` |
| TRT FP16 (static prev-BEV) | 0.7270 | 93.5 ms | `trt_static_prevbev_results.json` |
| INT8 backbone + FP16 head | 0.7828 | 88.9 ms | `eval_int8_bb_fp16_head.json` |
| INT8 backbone + INT8 head (real-data calib) | 0.7805 | 132.7 ms | `eval_int8_fixed.json` |
| INT8 head, random calib (**broken**) | 4.359 | 23.7 ms | cautionary example, §3.3 |

Readings:

- PTQ INT8 costs ~+0.05 L2 vs FP16 (0.7805 vs 0.7270) — outside a tight ±0.01 gate.
- The x86→Orin slowdown (parent-project reports) is ~7× for the backbone and ~8× for the head at 50 W — roughly uniform, i.e. the bottleneck structure carries across platforms.

## 5. Levers, ranked by measured impact

| Rank | Lever | Measured effect | Evidence |
|---|---|---|---|
| 1 | **Model structure** (BEV queries 4×↓, layers 2×↓, FPN 4×↓) | E2E 11.9× on Orin INT8 | §3.4 table |
| 2 | **Precision: INT8 on backbone** | 1.78× on backbone | §3.1 |
| 3 | **Precision: INT8 on head (PTQ)** | ~1.0× — the head's kernel mix is FP16-pinned, and DRAM is not the wall (§3.2 census) | §3.2 |
| 4 | Plugin/kernel + build tuning (`opt8` vectorization, `max_aux_streams=1`, OptLevel sweep) | head 66.9→62.8 ms (aux streams), 62.9→58.9 ms (opt8) on VAD-Small; OptLevel best per-model (Small=3, Base=5) | parent-project reports |
| 5 | Structured pruning (FFN channel pruning) | **not viable**: −5% params, −24% accuracy; head is bandwidth-bound so parameter cuts don't cut latency | parent-project reports |

The headline: **structure beats precision**. All precision work combined moved E2E by <2×; one architecture decision moved it 11.9×.

## 6. Remaining headroom

- **Head MatMul subgraph** is still FP16 by design (§3.3). Pushing it to INT8 requires quantization-aware training on the MatMul inputs — out of scope for this release. The census bounds the upside: even a perfect MatMul→INT8 conversion touches only the 21% GEMM share; the 77% plugin+elementwise majority stays FP16-pinned until the plugin gains an INT8 implementation.
- **Fully trained VAD-Small** inherits the identical structure-determined deployment cost (~64 ms-class at MAXN) while closing the accuracy gap that the 2-epoch fast-run checkpoint leaves open.
- **DLA** (2× NVDLA v2.0 on AGX Orin) is unused by this pipeline; conv-only subgraphs (backbone) are the natural candidates.
- **Power modes** matter mostly pre-optimization: after the OptLevel=3 pass the head is no longer compute-starved, so 50 W vs MAXN E2E differ by <1% (see `docs/power_modes.md` §3).

## References

- `x86/eval/results/vad_module_profile.json` — module breakdown, architecture summary
- `x86/eval/results/trt_static_prevbev_results.json` — FP16 ladder entry (93.5 ms / 0.7270)
- `x86/eval/results/eval_int8_fixed.json` — INT8 ladder entry (132.7 ms / 0.7805)
- `x86/eval/results/eval_int8_bb_fp16_head.json`, `eval_int8_bb_int8_head.json`, `eval_final_20260612.json` — remaining ladder rows
- `arm64/eval/results/eval_orin_fp16_vs_int8*.json` — Orin head FP16-vs-INT8 fidelity and latency (dynamic-shape and static-shape variants)
- `arm64/eval/results/orin_head_kernel_census.json` — full-inference ncu kernel census of the INT8-quantized head (zero INT8 kernels; per-class shares and SOL medians)
- `arm64/eval/results/orin_microbench_maxn.json` — platform peak anchors: 170 GB/s DRAM, 21.7 TFLOPS FP16 GEMM, 10.5 TOPS `torch._int_mm` INT8, 28 µs/kernel launch floor (`arm64/eval/microbench_orin.py`)
- Parent-project development reports: Orin 50 W re-measured tables (backbone 85.4→47.9 ms, head 694.1→678.8 ms, E2E ~779.5→~726.7 ms; TRT 10.4 backbone rebuild 30.9 ms; head INT8 OptLevel sweep opt2/opt3/opt5), VAD-Small deployment report (Orin E2E 61.3 ms, head cos_sim 0.99998), opt8 NaN bug report
- NVIDIA, *Jetson AGX Orin Technical Brief*: <https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf>
- NVIDIA Jetson Orin product page: <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/>
