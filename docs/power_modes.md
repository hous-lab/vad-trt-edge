# Orin Power Modes

NVIDIA Jetson AGX Orin exposes several `nvpmodel` power modes. For embedded deployment work the distinction matters: a fixed power budget does **not** mean a fixed latency, and the mode you measure under determines whether your latency numbers are reproducible. This document records what was measured on the development Orin across all four `nvpmodel` modes for three engine configurations, plus a reproducible measurement methodology.

> Measured on an AGX Orin 64 GB module (TensorRT 10.4.0, torch 2.8.0, 2026-10). Three engine configurations, all in `arm64/eval/results/power_modes/` (per-mode JSONs with per-run latencies + tegrastats logs). Two quantized deployment configurations:
> - **VAD-Small INT8** (§4.2) — the final deployment configuration, the model carrying the **<100 ms Orin target**.
> - **VAD-Base INT8 (PTQ)** (§4.3) — the final VAD-Base quantized deployment configuration: accuracy-aligned with the FP16 pair, backbone latency/footprint halved, E2E −3%.
>
> plus the **VAD-Base FP16 verification pair** (§4.1), kept as reference data, the accuracy anchor for the PTQ pair, and for cross-config scaling read-outs.

## 1. Why power modes matter

- **Deterministic latency** — DVFS will otherwise pick whatever clock the thermal and power headroom allows at that moment. Two runs of the same engine can differ by tens of percent. Comparisons between FP16 and INT8, or between builds, are only meaningful at a fixed, recorded power mode.
- **Embedded realism** — a vehicle compute platform runs under a power budget. Measuring only at MAXN (unlimited clocks) hides the power-limited latency you will actually see in the car.
- **Frequency caps vs power walls** — the modes are *not* "allocate N watts". Each mode fixes per-domain **clock ceilings** plus a power wall. Which one binds first is an empirical question, and for this workload the answer is clear: **clock caps bind everywhere** — no config in any capped mode came near its power wall (§4.1–§4.3).

## 2. Mode table (AGX Orin)

`nvpmodel` mode IDs for the AGX Orin 64 GB module (per NVIDIA's Platform Power and Performance documentation): `0`=MAXN, `1`=MODE_15W, `2`=MODE_30W, `3`=MODE_50W. Always confirm on your module with `nvpmodel -q` — the ID-to-name mapping is module-specific.

| Mode | GPU clock ceiling | CPU (A78) ceiling | EMC ceiling | Power wall |
|---|---|---|---|---|
| **MAXN** (0) | unlimited (DVFS → 1300 MHz) | → 2.2 GHz | → 3.2 GHz | none |
| **50 W** (3) | **816 MHz** | 1.50 GHz | unlimited | 50 W |
| 30 W (2) | 612 MHz | 1.73 GHz | unlimited | 30 W |
| 15 W (1) | 408 MHz | 1.11 GHz | **2.13 GHz** | 15 W |

Notes:

- MAXN leaves all clock fields unlimited; DVFS climbs to the physical ceiling (GPU 1.3 GHz, CPU 2.2 GHz, EMC 3.2 GHz). All other modes pin clock ceilings.
- In every capped mode the **clock caps bind, not the power wall**: peak VDD_GPU_SOC rail draw was 37.4 W in MAXN, 21.9 W in 50 W, 11.5 W in 30 W, 6.8 W in 15 W (FP16 bench, §4.1; the small-model bench drew even less, §4.2; the Base INT8 pair peaked at 33.8 W in MAXN, §4.3), far under the wall in every mode. The modes are clock ceilings, not power allocations for this workload.
- Power accounting note: this unit's `tegrastats` exposes `VDD_GPU_SOC` / `VDD_CPU_CV` / `VIN_SYS_5V0` rails but no total `VDD_IN`; power rows report the `VDD_GPU_SOC` rail (GPU + SoC complex, the dominant consumer under engine inference). Rail means include the idle valleys between timed runs (GR3D idles at 305 MHz between runs), so the max column best reflects the under-load state.
- 15 W is the only mode that also caps EMC (2133 MHz vs 3199 MHz): memory bandwidth −33% on top of the GPU clock cap.

## 3. Earlier A/B (VAD-Small INT8 head, parent project)

Head-only A/B on an earlier small-model head build, isolated to expose the power-mode effect:

| | MAXN | 50 W |
|---|---|---|
| GPU clock | 1298 MHz | 814 MHz |
| GPU+SoC draw | ~29 W | ~15 W |
| EMC | 3199 MHz | 3199 MHz |
| **Head latency** | **44.9 ms** | **62.9 ms** |

After the full optimization pass (OptLevel=3 build, vectorized plugin, single auxiliary stream), an earlier report recorded E2E 61.3 ms at 50 W vs 61.9 ms at MAXN — the power-mode difference "largely disappears". **The new benchmark (§4.2) does not reproduce this**: with the final small engines it measures 87.4 ms at 50 W vs 63.7 ms at MAXN. The earlier 50 W row showed the backbone at exactly 25.6 ms in *both* modes, which this re-benchmark does not reproduce (35.8 vs 25.3 ms); the earlier ad-hoc measurement likely differed in warm-up/DVFS state. Treat §4.2 as superseding and use these engines' fresh numbers.

Practical notes from the same measurements:

- `jetson_clocks` (clock pinning at max) adds nothing once DVFS is already saturated at the mode's ceiling.
- CPU affinity tuning reduces jitter, not mean latency.
- Mode changes can require a reboot to take effect, and persist across power cycles.

## 4. Reproducible measurement methodology

`arm64/eval/power_mode_bench.py` and its driver `arm64/eval/run_power_mode_bench.sh` (run on the Orin) implement the protocol. Per mode: switch via `nvpmodel -m`, **verify the switch took effect** (`nvpmodel -q`), load both engines once, 3 warm-up + N timed runs (default 10) per engine on fixed inputs, `tegrastats --interval 100` sampling clocks and rail power during the timed window, per-run latencies with p50/p90/mean into one JSON per mode; MAXN is restored at the end.

All three engine configurations are benched with the same harness via env overrides:

```bash
# on the Orin, at the repo root:
export ORIN_TRT_ROOT=/usr/src/TensorRT-10.4.0.26   # pip wheel needs the tarball libnvinfer (environment.md §3.1)

# VAD-Small INT8 pair (final deployment config):
BB_ENGINE=arm64/trt_compile/output/backbone_int8_small_orin.engine \
HEAD_ENGINE=arm64/trt_compile/output/head_int8_small_orin.engine \
    bash arm64/eval/run_power_mode_bench.sh 0 3 2 1

# VAD-Base INT8 (PTQ) pair (final Base quantized deployment config):
BB_ENGINE=arm64/trt_compile/output/backbone_int8_orin.engine \
HEAD_ENGINE=arm64/trt_compile/output/head_int8_orin.engine \
    bash arm64/eval/run_power_mode_bench.sh 0 3
```

Methodology notes:

- No `jetson_clocks` pinning: DVFS climbs to the mode's clock ceiling under load, and tegrastats confirms the in-run clock — the measured clock is part of the result, not a controlled variable.
- Run benches on an idled, thermally settled device. A bench run started immediately after a multi-hour engine build picks up DVFS noise (measured: head p90 744 ms vs 291 ms after idling — same engines, same day); the harness's 3 warm-up runs do not substitute for thermal settle time.
- `SUFFIX=<string>` suffixes result filenames so different engine pairs do not overwrite each other's JSONs — always suffix when benching a non-default pair (an unsuffixed run overwrote the FP16 reference JSONs on the device once; they were restored from the repo's committed copies).
- Fixed inputs for comparability: the backbone runs on a fixed random tensor [6,3,736,1280]; the head runs on a real calibration frame (`frame_000` of `data/calib_data/head_calib.npz`).
- The small pair has no small-model calib npz on the device, so its head inputs are **random tensors at the engine's input shapes** (`--random-head-inputs`). Static TRT engines have data-independent latency, so this is valid for latency-only benchmarking (and the JSON's `workload` field records which mode was used).
- E2E = backbone p50 + head p50 (the two engines execute sequentially in the real pipeline, so their p50s add; each is timed over its own 3+10 runs inside one tegrastats window).

### 4.1 VAD-Base FP16 pair (verification pair, reference data)

Measured 2026-10-20 with `run_power_mode_bench.sh` (TRT 10.4.0, torch 2.8.0; 3 warm-up + 10 timed runs per engine, fixed inputs; tegrastats 100 ms; no `jetson_clocks` — plain nvpmodel + DVFS). 30 W / 15 W were entered by **booting into the mode** (live switching reports "Reboot required" on this unit); MAXN was restored the same way and verified afterwards.

| Mode | GPU clock mean/max | EMC | VDD_GPU_SOC rail (mean/max) | Backbone p50 | Head p50 | **E2E p50** | E2E p90 |
|---|---|---|---|---|---|---|---|
| MAXN (0) | 1112 / 1300 MHz | 3199 MHz | 27.3 / 37.4 W | 55.4 ms | 282.5 ms | **337.9 ms** | 338.1 ms |
| 50 W (3) | 749 / 815 MHz | 3199 MHz | 16.3 / 21.9 W | 78.8 ms | 396.1 ms | **474.9 ms** | 475.0 ms |
| 30 W (2) | 592 / 611 MHz | 3199 MHz | 9.2 / 11.5 W | 175.2 ms | 908.7 ms | **1083.9 ms** | 1084.2 ms |
| 15 W (1) | 404 / 407 MHz | **2133 MHz** | 6.0 / 6.8 W | 343.3 ms | 1783.6 ms | **2126.9 ms** | 2127.3 ms |

Read-outs (FP16 pair):

- 50 W caps the GPU at 815 MHz (confirmed in-run; MAXN reaches 1300 MHz) and throttles nothing else — EMC stays at 3199 MHz.
- E2E slowdown at 50 W is 1.40x (474.9/337.9) against a 1.60x GPU clock cap: **sub-linear** scaling — the workload keeps a memory-hierarchy/latency-bound component (L1/L2 gather pressure + a measured 28 µs/kernel launch-execute floor; DRAM itself sits at ~⅓ utilization — see roofline_analysis.md §3.2 kernel census) that the GPU clock does not scale.
- Scaling across all four modes stays sub-linear in GPU clock: 1300→611→407 MHz would predict 2.1x/3.2x, measured is 3.2x/6.3x. At low clocks the memory-hierarchy/latency-bound part dominates; at 15 W the EMC cap (2133 MHz) compounds the compute shortfall.
- Every capped mode draws far below its power wall (50 W mode peaks at 21.9 W on this rail) — clock caps, not the watt budget, limit the workload.

### 4.2 VAD-Small INT8 (final deployment config)

Same harness and protocol as §4.1, 2026-10-20. Engines: `backbone_int8_small_orin.engine` + `head_int8_small_orin.engine` (25.8 MB + 40.2 MB, parent-project Orin builds, deserialize-verified on TRT 10.4). Head inputs are random tensors at the engine's input shapes (`--random-head-inputs`, data-independent latency for a static engine); backbone input is a fixed random tensor.

| Mode | GPU clock mean/max | EMC | VDD_GPU_SOC rail (mean/max) | Backbone p50 | Head p50 | **E2E p50** | E2E p90 |
|---|---|---|---|---|---|---|---|
| MAXN (0) | 744 / 1300 MHz | 3199 MHz | 11.0 / 23.9 W | 25.3 ms | 38.4 ms | **63.7 ms** | 64.5 ms |
| 50 W (3) | 582 / 815 MHz | 3199 MHz | 8.8 / 15.1 W | 35.8 ms | 51.6 ms | **87.4 ms** | 87.8 ms |
| 30 W (2) | 527 / 611 MHz | 3199 MHz | 7.1 / 9.2 W | 87.2 ms | 108.9 ms | **196.1 ms** | 196.3 ms |
| 15 W (1) | 391 / 407 MHz | **2133 MHz** | 5.4 / 6.4 W | 173.6 ms | 211.1 ms | **384.7 ms** | 385.1 ms |

Read-outs (VAD-Small INT8):

- **<100 ms target: met at MAXN (63.7 ms) and 50 W (87.4 ms); not met at 30 W (196.1 ms) or 15 W (384.7 ms)**. The target holds as long as the GPU can run at 50 W-mode clocks (≥815 MHz): a 30 W-class budget would need further head optimization (e.g. overlapping backbone and head across frames) to get back under 100 ms.
- vs the parent-project report: MAXN is consistent (61.9 → 63.7 ms), 50 W is not (61.3 vs 87.4 ms) — see §3 for the reconciliation.
- Scaling vs MAXN: 1.37x / 3.08x / 6.04x — again sub-linear and clock-cap-governed, consistent with §4.1.
- Head input note: the small head was benched on random inputs (`--random-head-inputs`); for a static engine latency is data-independent, so the numbers are comparable to the real-input benches above.
- Every capped mode draws far below its power wall (peak 23.9 W in MAXN, 15.1 W in 50 W) — same conclusion as §4.1.

### 4.3 VAD-Base INT8 (PTQ) — the final VAD-Base quantized deployment config

Same harness and protocol as §4.1, measured 2026-10-20 on an idled device (thermal-settle note above). Engines: `backbone_int8_orin.engine` (28.0 MB) + `head_int8_orin.engine` (79.2 MB), built by this repo's Orin chain (docs/pipeline_guide.md §7) with real-data PTQ calibration (EntropyCalibration2 over the 64 real frames of `head_calib.npz`; the backbone reuses the committed cache, and the committed head cache is the product of on-device calibration — fallback verified equivalent, see read-outs).

| Mode | GPU clock mean/max | EMC | VDD_GPU_SOC rail (mean/max) | Backbone p50 | Head p50 | **E2E p50** | E2E p90 |
|---|---|---|---|---|---|---|---|
| MAXN (0) | 1109 / 1300 MHz | 3199 MHz | 25.6 / 33.8 W | 35.6 ms | 291.1 ms | **326.7 ms** | 326.9 ms |
| 50 W (3) | 748 / 815 MHz | 3199 MHz | 15.4 / 18.3 W | 50.3 ms | 410.5 ms | **460.8 ms** | 461.0 ms |

Read-outs (VAD-Base INT8 PTQ):

- **Backbone INT8 pays off**: 55.4 → 35.6 ms at MAXN and 78.8 → 50.3 ms at 50 W (both 1.56x), engine footprint 54.1 → 28.0 MB (−48%). This is where INT8 quantization actually buys something on VAD-Base.
- **Head INT8 ≈ FP16**: 291.1 vs 282.5 ms at MAXN (+3.0%) and 410.5 vs 396.1 ms at 50 W (+3.6%) — no latency benefit. The head's quantization-sensitive nodes stay FP16-pinned (the MatMul chain), so INT8 adds reformat overhead on an already-FP16 kernel mix; zero INT8 kernels execute in the head (docs/roofline_analysis.md §3.2 kernel census). The head engine is even slightly larger than FP16 (79.2 vs 75.3 MB).
- Net E2E: **326.7 ms at MAXN (−3.3% vs FP16) and 460.8 ms at 50 W (−3.0%)** — INT8 on VAD-Base buys backbone latency/footprint, not real-time. The real-time story remains VAD-Small INT8 (§4.2: 63.7 ms at MAXN / 87.4 ms at 50 W).
- **Accuracy: passes.** Head-only fidelity vs the x86 FP16 reference over 10 real calibration frames: **cos_sim ≥ 0.9986 on all 9 outputs** (avg 0.9990–1.0000; gate 0.99 aligned / 0.95 acceptable) — `arm64/eval/results/orin_int8_ptq_lite.json`. The x86 L2(1s) pipeline gate for this same recipe (real-data PTQ + MatMul-FP16 pinning) is 0.7805 (docs/pipeline_guide.md §5, within the ±0.01 repro gate and the ±0.02 cross-rebuild band).
- Clock caps bind, not the power wall (rail peaks 33.8 W in MAXN, 18.3 W in 50 W) — same conclusion as §4.1.
- Bench provenance: `power_mode_{MAXN,50w}_base_int8.json` + tegrastats logs.

## 5. References

- This repo's benchmark: `arm64/eval/power_mode_bench.py` + `arm64/eval/run_power_mode_bench.sh`, results in `arm64/eval/results/power_modes/` — per-config per-mode JSONs (`power_mode_<mode>.json` FP16, `power_mode_<mode>_small.json`, `power_mode_<mode>_base_int8.json`) with per-run latencies (raw tegrastats logs stay local, gitignored).
- Engine provenance: the FP16 pair is built by this repo's chain (docs/pipeline_guide.md §7); the small pair is the parent project's Orin builds; the Base INT8 pair is built by this repo's own chain (§4.3). Engines are sm_87 (Orin) — gitignored, rebuild per the scripts above.
- Earlier measurements: head-only smoke results (`arm64/eval/results/orin_fp16_smoke_*.json`), §3 parent-project A/B. NVIDIA docs: Platform Power and Performance <https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html>; nvpmodel/jetson_clocks admin guide <https://docs.nvidia.com/jetson/archives/r35.1/AdminGuide/topics/workflow_development_tools.html>.
