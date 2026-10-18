# VAD TensorRT Deployment on x86_64 and NVIDIA AGX Orin

End-to-end deployment of **VAD-Base (Bench2Drive port)** through the full TensorRT pipeline — custom plugins → ONNX export → FP16/INT8 engines → pytorch-vs-engine accuracy/latency eval — on an x86_64 dev host (RTX 4090 Laptop, `sm_89`) and NVIDIA Jetson AGX Orin (`sm_87`).

The pipeline is checkpoint-driven and reproducible: every artifact that is not committed (plugins `.so`, ONNX, TRT engines, calibration tensors) is regenerable from the scripts in this repository plus a small set of public inputs (checkpoint, Bench2Drive mini-set).

## Scope

This repository covers (translated from the project's original scope notes):

1. **VAD-Base (Bench2Drive port) on two platforms**: x86_64 @ RTX 4090 Laptop and arm64 @ Jetson AGX Orin.
2. **Orin roofline analysis with evidence**: why INT8 quantization alone cannot reach real-time on the Orin for VAD-Base — the 40,000-BEV-query head runs an almost entirely FP16-pinned kernel mix (measured: zero INT8 kernels execute, DRAM at ~1/3 utilization - not a DRAM wall); see [docs/roofline_analysis.md](docs/roofline_analysis.md).
3. **A real-time-capable variant: VAD-Small** (custom, BEV 100x100): which head layers were reduced and by how much (structure + parameter tables) — see [docs/model_variants.md](docs/model_variants.md).
4. **VAD-Small honesty note**: it was trained for only 2 epochs as a fast verification run and therefore has **no closed-loop driving score**. It exists to probe what model-structure traits buy real-time behavior under the VAD architecture.
5. **Reference comparison**: VAD-Base vs VAD-Tiny (official nuScenes models) — structure, parameters, and planning metrics — and an estimate of where our VAD-Small lands relative to VAD-Base ([docs/model_variants.md](docs/model_variants.md) §7).
6. **Autoware integration** (two closed-loop schemes): single-machine (4090 Laptop) and two-machine (4090 Laptop <-> Orin over the USB L4T bridge). **Deferred to the next phase** — this release contains the deployment pipeline only; see the Autoware note in [docs/pipeline_guide.md](docs/pipeline_guide.md).
7. *(Done)* **Orin power-mode comparison** (all four modes measured for the VAD-Small INT8 deployment config and the VAD-Base INT8 (PTQ) deployment config, plus the VAD-Base FP16 verification pair; p50/p90 + clocks/power; 30 W / 15 W entered by rebooting into the mode) — [docs/power_modes.md](docs/power_modes.md).
8. *(Deferred)* Re-train VAD-Small on the full Bench2Drive training split to make the small model competitive for closed-loop simulation.

## Hardware Platforms

All numbers in this README and in `docs/` were measured on exactly these two devices — latency figures are **not** transferable to other GPUs, accuracy gates are anchored per engine build.

### x86_64 workstation — RTX 4090 **Laptop** GPU

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 **Laptop** GPU (Ada, **sm_89**) |
| VRAM | 16 GB GDDR6 (16,376 MiB) |
| Memory bandwidth | ~576 GB/s (256-bit, 18 Gbps) |
| TGP | up to 175 W (incl. Dynamic Boost) |
| CUDA / driver | 11.8 / 580.95.05 |
| TensorRT | 10.16.1.11 (pip wheel, bundled libs) |
| OS | Linux x86_64 |

**Not a desktop RTX 4090.** The laptop part has roughly half the CUDA cores (~7.7k vs 16.4k), ~57% of the memory bandwidth (576 vs 1008 GB/s), a third of the power budget (175 W vs 450 W) and 16 GB instead of 24 GB VRAM. If you reproduce on a desktop 4090 (or any other GPU), expect **lower** latencies than this README reports — that difference is hardware, not software. Accuracy results (L2) are unaffected by this since they are properties of the engine builds, not of the GPU speed.

### Orin — NVIDIA AGX Orin 64 GB

| Component | Specification |
|---|---|
| SoC | AGX Orin 64 GB, Ampere GPU (**sm_87**), 2048 CUDA cores |
| Memory | 64 GB LPDDR5, 204.8 GB/s |
| Power modes | MAXN / 50 W / 30 W / 15 W (see docs/power_modes.md) |
| TensorRT | 10.4.0.26 (tarball) — see docs/environment.md §3.1 for the wheel/tarball libnvinfer caveat |
| OS | JetPack (L4T), Linux aarch64 |

## Key Results

### x86_64 precision/latency ladder (VAD-Base, 173-frame eval set)

| Configuration | L2 (1 s) | Latency | Result file |
|---|---|---|---|
| PyTorch FP32 (reference) | 0.7300 | 233.3 ms | `x86/eval/results/eval_final_20260612.json` |
| TRT FP16 (static prev-BEV) | 0.7270 | 93.5 ms | `x86/eval/results/trt_static_prevbev_results.json` |
| INT8 backbone + FP16 head | 0.7828 | 88.9 ms | `x86/eval/results/eval_int8_bb_fp16_head.json` |
| INT8 backbone + INT8 head (real-data calib) | 0.7805 | 132.7 ms | `x86/eval/results/eval_int8_fixed.json` |
| INT8 head with random calib (**broken**) | 4.359 | 23.7 ms | cautionary example |

L2(1s) = mean planning L2 distance at 1 s horizon vs ground-truth future ego trajectory, in meters — same metric family as the official VAD planning table. Repro gates for this repo's chain: **FP16 0.727, INT8 0.7805 (each ± 0.01)**. Reference engines rerun deterministically; fresh engine rebuilds vary within ±0.02 (measured, see docs/pipeline_guide.md §5).

### Orin (AGX Orin 64 GB, 50 W mode, INT8 engines)

| Module | VAD-Base | VAD-Small (2-epoch) | Speedup |
|---|---|---|---|
| Backbone INT8 | 47.9 ms | 25.6 ms | 1.87x |
| Head INT8 | 678.8 ms | 35.8 ms | **19.0x** |
| **End-to-end** | **726.7 ms** | **61.3 ms** | 11.9x |
| Parameters | 58.38 M | 39.89 M | 1.46x |
| Head cos_sim vs PyTorch | — | 0.99998 | quantization loss-free at output level |

Notes on the Orin table:

- **On-device FP16 verification:** the repo's FP16 pair (backbone + head) builds cleanly from a fresh checkout on the AGX Orin 64 GB unit and passes the 10-frame smoke — head-only 286.1 ms avg (3.49 FPS) at MAXN, 418.2 ms in 50 W mode (`arm64/eval/results/orin_fp16_smoke_{maxn,50w}_2026-10-16.json`; the smoke harness measures the head engine only). Full-pair latency (backbone + head) per power mode, with p50/p90 and clocks/power: see docs/power_modes.md §4.1. The VAD-Small INT8 deployment config is measured across all four modes as well (§4.2 there: the <100 ms target model — 63.7 ms at MAXN / 87.4 ms at 50 W, but 196 ms at 30 W).
- **On-device VAD-Base INT8 (PTQ) verification (final Base quantized deployment config):** the repo's INT8 pair builds from the committed calibration caches — or recalibrates on-device when they are absent (both paths verified on this unit) — passes the 10-frame head-only fidelity eval vs the x86 FP16 reference at **cos_sim ≥ 0.9986 on all 9 outputs** (gate 0.99; `arm64/eval/results/orin_int8_ptq_lite.json`), and measures **E2E 326.7 ms at MAXN / 460.8 ms at 50 W** — INT8 halves the backbone's latency and footprint (35.6 ms / 28.0 MB vs FP16's 55.4 ms / 54.1 MB) while the head stays FP16-pinned and matches FP16 latency (291.1 ms) — per-mode table: [docs/power_modes.md](docs/power_modes.md) §4.3.
- The VAD-Base head column is the accuracy-clean matched-era comparison. A faster VAD-Base head build exists (302.2 ms via the vectorized `opt8` plugin at OptLevel=5), but that combination carries a **non-deterministic NaN bug at VAD-Base size on Orin** (~55% of runs) and is not production-usable — see [docs/roofline_analysis.md](docs/roofline_analysis.md) §5.
- Quantization bought ~1.0x on the VAD-Base head (FP16-pinned kernel mix), while the structural reduction (VAD-Small) bought ~19x on the same head. This is the central finding; see [docs/roofline_analysis.md](docs/roofline_analysis.md) §3.
- VAD-Small latency is **structure-determined**, not weight-determined: a fully trained VAD-Small deploys through this chain unchanged and hits the same ~61 ms-class Orin E2E. See [docs/model_variants.md](docs/model_variants.md) §5.

### Lessons worth reading the docs for

- **Calibration quality is make-or-break**: real-frame calibration gave L2 0.7805; a random-tensor calibration produced 4.359 (broken trajectories) at a *better* latency — accuracy gates are mandatory, latency is meaningless without them.
- **Calibration caches (`.calib`) are portable** across platforms and TRT versions; **engines are not** (neither across `sm_89`/`sm_87` nor across TRT major versions). This repo commits the caches (`data/calib_data/*.calib`) so INT8 builds skip re-calibration; if a cache is absent the builders recalibrate on-device from the real calibration frames — both paths verified on-device.
- **OBEY vs PREFER precision flags behave differently** with mixed FP16/INT8 networks; the INT8 head builder's "three fixes" story is documented in [docs/pipeline_guide.md](docs/pipeline_guide.md).
- **Per-layer precision control (MatMul stays FP16)**: INT8-quantizing the head's MatMul chain (attention scores / value projections) collapses accuracy; the builders keep MatMul in FP16 and quantize Conv/GEMM where it pays.
- **The `opt8` vectorized plugin is 4x faster but NaN-unsafe at VAD-Base size on Orin** (55% of runs) — VAD-Tiny/Small sizes are immune. Production builds use the scalar plugin (OptLevel 2/3).
- **Pruning was tested and rejected**: −5% parameters → L2 0.730→0.908. Not viable; structural reduction (VAD-Small) was the lever that worked.

## Pipeline Architecture

```
                        +---------------------------+
 ckpts/vad_b2d_base.pth | Stage 1: TRT plugins      |
 Bench2Drive mini-set   |  MSDeformAttn / Rotate /  |
        |               |  SelectAndPad -> .so      |
        v               +------------+--------------+
 [Stage 2] ONNX export  <-------------+
  export_vad_prev.py -> demo_data (real two-pass prev_bev)
  export_backbone_model.py -> vad_backbone.onnx (+ _b1 variant)
  export_head_script.py -> vad_head_withprev.onnx (prev_bev static [40000,1,256])
        |
        v
 [Stage 3] TensorRT engine builds
  build_head_v2.py (BUILD_MODE=mixed)     -> head_withprev_fp16_fixed_v2.engine
  build_backbone_int8.py (int8_real)      -> backbone_int8.engine
  build_head_int8.py (real calib cache)   -> head_int8_fixed.engine
        |
        v
 [Stage 4] Evaluation (x86/eval/eval_pytorch_vs_trt.py)
  PyTorch reference vs TRT engines, L2(1s) gate + latency
  gate: FP16 ~0.727 / INT8 ~0.7805 (+/- 0.01)
```

The head ONNX exports `prev_bev` **static** ([40000, 1, 256]): the INT8 builders set no optimization profile by design, so the input must be static (a dynamic `prev_bev` makes the build fail with "no optimization profile has been defined"). Outputs stay dynamic; TRT only constrains inputs.

## Repository Layout

```
vad-trt-edge/
+-- README.md                    this document
+-- LICENSE / NOTICE             Apache-2.0; NVIDIA-licensed components in NOTICE
+-- run_pipeline.sh              x86 one-shot chain (stages 1-4)
+-- ckpts/README.md              official checkpoint links + placement
+-- ckpts/                       (gitignored) vad_b2d_base.pth, VAD_small_b2d_fast.pth
+-- configs/                     deployment/export configs (flattened, self-contained)
|   +-- VAD_base_e2e_b2d.py      VAD_small_b2d_fast.py
+-- data/
|   +-- README.md                Bench2Drive mini-set download/placement guide
|   +-- b2d_infos_val.pkl, b2d_map_infos.pkl, splits/   committed
|   +-- bench2drive/             (gitignored) mini-set: maps/ + v1/
|   +-- calib_data/              *.calib caches committed; *.npz regenerated via tools/
+-- plugins/                     TRT plugin sources (MSDeformAttn, Rotate, SelectAndPad)
+-- onnx_export/
|   +-- scripts/                 export_head_script.py (static prev_bev), export_backbone_model.py,
|   |                            export_vad_prev.py (two-pass real prev_bev), export_head_model.py
|   +-- patches/                 ONNX graph patches (patch_head, patch_rotate, bev_deploy)
|   +-- demo_data/               (gitignored, regenerated by Stage 2)
|   +-- output/                  (gitignored) vad_backbone*.onnx, vad_head_withprev.onnx
+-- tools/                       gen_calib_data.py (+ _small), run_gen_calib.sh
+-- scripts/                     sync_to_orin.sh, run_orin_pipeline.sh, detect_trt_versions.sh
+-- x86/
|   +-- plugins_build/build_x86.sh
|   +-- trt_compile/scripts/     build_head_v2.py, build_backbone_int8.py, build_head_int8.py (+ _mixed)
|   +-- trt GoldenCompile/output/   (gitignored) engines
|   +-- eval/                    eval_pytorch_vs_strt.py, profile_vad_modules.py, run_eval_*.sh, results/
+-- arm64/
|   +-- plugins_build/build_orin.sh
|   +-- trt_compile/{fp16,int8}/ Orin engine builders + run scripts
|   +-- trt_compile/output/      (gitignored)
|   +-- eval/                    eval_orin*.py, collect_gt_x86.py, run_eval_*.sh, results/
+-- docs/
    +-- pipeline_guide.md        full repro manual (per-stage commands + expected outputs/gates)
    +-- roofline_analysis.md     Orin roofline + BEV-query bottleneck evidence
    +-- model_variants.md        Base vs Small structure/params, nuScenes reference, estimates
    +-- power_modes.md           All 4 power modes x 2 engine configs measured (VAD-Small deployment config + VAD-Base FP16 verification pair)
    +-- environment.md           both machines env + TRT version matrix + zoo-repo setup
```

## Quick Start (x86)

Prerequisites (full matrix in [docs/environment.md](docs/environment.md)):

- x86_64 Ubuntu 22.04 + NVIDIA GPU (Ada/Ampere class), CUDA 11.8 toolkit, TRT 10.4 GA tarball as `TRT_ROOT`.
- **The conda env is the upstream zoo training env — don't invent one**: build it per upstream [INSTALL.md](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/INSTALL.md) and sanity-check with their [TRAIN_EVAL.md](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/TRAIN_EVAL.md); once an upstream eval/train job runs in your env, every zoo-importing stage here imports cleanly. This pipeline only needs the env's NAME (`CONDA_ENV`, default `b2d_zoo` — this repo's instance: python 3.8, torch 2.4.1+cu118, tensorrt 10.16.1.11).
- Checkpoint `ckpts/vad_b2d_base.pth` (links in `ckpts/README.md`) and the Bench2Drive mini-set under `data/bench2drive/` (guide in `data/README.md`).
- **Only two env vars are hard requirements — `TRT_ROOT` and `CUDA_HOME`**. Everything else auto-resolves: `CONDA_BASE` (`$HOME/apps/miniconda3`, fallback `$HOME/miniconda3`), `VAD_DEPLOY_ROOT` (from script location), `VAD_PROJECT_ROOT` (zoo repo auto-discovered next to this repo / sibling / `$HOME`; export only if yours lives elsewhere), and eval engine paths default into `x86/trt_compile/output/`.

## Pipeline Stages

```bash
TRT_ROOT=/path/to/TensorRT-10.4.0.26 CUDA_HOME=/usr/local/cuda-11.8 CONDA_ENV=b2d_zoo bash run_pipeline.sh
```

Per-stage detail (full manual: docs/pipeline_guide.md):

| Stage | Scripts | Outputs (gitignored) | Check |
|---|---|---|---|
| 1 | `x86/plugins .so` build | `plugins/build/libplugins.so` | dlopen OK; DT_NEEDED shows `libnvinfer.so.10` |
| 2 | export_vad_prev / export_backbone_model / export_head_script | `onnx_export/demo_data/`, 3 ONNX | backbone numeric check <=1e-3-class; head prev_bev static [40000,1,256] |
| 3 | build_head_v2 (BUILD_MODE=mixed), build_backbone_int8 (int8_real), build_head_int8 | 3 engines in `x86/trt_compile/output/` | engines deserialize; ~50-130 MB each |
| 4 | x86/eval/eval_pytorch_vs_trt.py | `x86/eval/results/*.json` | L2(1s): FP16 0.727 (reference engines), INT8 0.7805 reference / 0.763 fresh rebuild (±0.02 cross-rebuild band) |

## VAD-Small Track

`VAD_small_b2d_fast` (2-epoch fast-run checkpoint) deploys through the repo's chain: `onnx_export/scripts/export_small_onnx.py` → `x86/trt_compile/scripts/build_small_engine.py` (or the Orin small builders under `arm64/trt_compile/int8/`) → `x86/eval/eval_small_pytorch_vs_trt.py`. Calibration data for the small head: `tools/gen_calib_data_small.py`. Structure/params/fidelity findings: [docs/model_variants.md](docs/model_variants.md).

## Documentation Map

| Doc | Contents |
|---|---|
| [docs/pipeline_guide.md](docs/pipeline_guide.md) | Full x86 repro manual: per-stage commands, expected outputs, accuracy gates, troubleshooting (plugin linkage, prev_bev static, TRT pairing) |
| [docs/roofline_analysis.md](docs/roofline_analysis.md) | Orin roofline inputs, module profiling, FP16-pinned head evidence (kernel census + microbench anchors), NaN-bug boundary, precision ladder provenance |
| [docs/model_variants.md](docs/model_variants.md) | Base vs Small structure/params, "fast" honesty note, nuScenes VAD-Base vs VAD-Tiny reference, where Small sits vs Base |
| [docs/power_modes.md](docs/power_modes.md) | All 4 nvpmodel modes x 2 engine configs measured (VAD-Small deployment config + VAD-Base FP16 verification pair; latency p50/p90, clocks, rail power) and the measurement harness |
| [docs/environment.md](docs/environment.md) | Both machines' software matrix, the two-TensorRT reality, TRT version matrix, external training repo setup |
| [data/README.md](data/README.md) | Bench2Drive mini-set: what it is, download, placement |
| [ckpts/README.md](ckpts/README.md) | Checkpoint provenance and placement (weights are not committed) |

## License & Acknowledgements

Apache-2.0 (see [LICENSE](LICENSE)); third-party components and their licenses are listed in [NOTICE](NOTICE).

Built on and with thanks to:

- [VAD](https://github.com/hustvl/VAD) (hustvl) — the VAD-Base/VAD-Tiny models and the vectorized-scene-representation approach;
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) (Thinklab-SJTU, NeurIPS 2024 D&B) — closed-loop dataset/benchmark and the VAD training zoo;
- **[Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) (Thinklab-SJTU)** — the `vad_b2d_base.pth` checkpoint used throughout this pipeline is taken directly from their release (see [ckpts/README.md](ckpts/README.md) for the download links). Special thanks for training and publicly releasing it;
- NVIDIA [DL4AGX](https://github.com/NVIDIA-AI-IOT/DL4AGX) / bevformer-tensorrt — MSDeformAttn TRT plugin lineage;
- The zoo training repository referenced via `VAD_PROJECT_ROOT` (external, not redistributed).
