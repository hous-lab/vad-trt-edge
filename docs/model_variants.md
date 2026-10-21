# Model Variants: VAD-Base vs VAD-Small (fast-trained) — Bench2Drive

This repository ships two model variants of the VAD (Vectorized scene representation for Autonomous Driving) perception head, ported to the Bench2Drive closed-loop benchmark and packaged for TensorRT deployment on x86_64 and Jetson Orin:

- **VAD-Base** — the full-size Bench2Drive port of VAD. This is the reference variant: fully trained, highest accuracy, and the slowest to deploy.
- **VAD-Small** (`VAD_small_b2d_fast`) — a deployment-oriented downsized variant. It uses the same head I/O contract and plugin set as VAD-Base, so it drops into the identical export → build → quantize → eval chain.

> **Honesty note on "fast"**: the shipped VAD-Small checkpoint was trained for only **2 epochs** as a *fast verification training run*. It exists to validate the whole deployment chain end-to-end with a small model and to benchmark deployment efficiency. It is **not** a fully trained, closed-loop-competitive model. See §4.

---

## 1. Naming bridge

During development the small variant was called **"tiny"** (`VAD_tiny_e2e_b2d_fast`). For the open-source release it was renamed **small**. The external training repository (the Bench2Drive zoo repo, referenced via the `VAD_PROJECT_ROOT` environment variable) still uses the old `tiny` names in its configs and work directories — that repository is not part of this open-source release and was not renamed.

| Open-source name (this repo) | Dev / training name (external zoo repo) | Where it appears |
|---|---|---|
| `VAD_small_b2d_fast` | `VAD_tiny_e2e_b2d_fast` | model / checkpoint name |
| `configs/VAD_small_b2d_fast.py` | `adzoo/vad/configs/VAD/VAD_tiny_e2e_b2d_fast.py` | training config (zoo repo) |
| `configs/VAD_small_b2d_fast.py` | `configs/VAD_tiny_e2e_b2d_fast.py` | flattened deployment config (this repo, pre-rename) |
| `ckpts/VAD_small_b2d_fast.pth` | `work_dirs/vad_tiny_e2e_b2d_fast/epoch_2.pth` | checkpoint (2-epoch fast run) |

If you cross-reference the training repository's docs, expect to see `tiny` where this repo says `small`. They are the same model.

## 2. Structure comparison

Measured from the checkpoints' `state_dict` and the configs (source: `docs/` development reports of the parent project; parameter counts are pure `nn.Parameter` counts, excluding ~0.05M of BN running-stat buffers):

| | VAD-Base | VAD-Small | Reduction |
|---|---|---|---|
| BEV resolution (queries) | 200×200 = 40,000 | 100×100 = 10,000 | **4×** fewer BEV cells |
| BEV encoder layers | 6 | 3 | **2×** |
| FPN sampling levels | 4 | 1 (backbone C5 only) | **4×** |
| Detection decoder layers | 6 | 3 | **2×** |
| Map decoder layers | 6 | 3 | **2×** |
| Temporal queue (`queue_length`) | 4 | 3 | 1.33× |
| Agent / map queries | 300 / 100 | 300 / 100 | unchanged |
| `prev_bev` shape (static) | [40000, 1, 256] | [10000, 1, 256] | 4× |
| **Total parameters** | **58.38 M** | **39.89 M** | **1.46× (−31.7%)** |
| — img_backbone (ResNet50) | 23.51 M | 23.51 M | **unchanged** |
| — img_neck (FPN) | 3.28 M | 1.11 M | 2.9× |
| — pts_bbox_head | 31.59 M | 15.26 M | 2.1× |

Key structural invariants: the head I/O contract (9 output tensors — detection, map, motion, and ego planning), the plugin set (MSDeformAttn, rotate, select-and-pad), the hidden dimension (256), and the prev-frame BEV queue mechanism are **identical** between the two variants. A VAD-Small engine is a drop-in replacement for a VAD-Base engine from the deployment pipeline's point of view.

## 3. Measured deployment results

All numbers below are INT8 TensorRT engines on **NVIDIA Jetson AGX Orin, 50 W power mode** (parent-project provenance — kept as the matched-era structural comparison; **this repo's latency convention is MAXN**, headline numbers in [power_modes.md](power_modes.md) §4.2/§4.3; the underlying per-output fidelity values for VAD-Small are the `cos_sim` JSONs listed in the References).

> The repo's own **VAD-Base INT8 (PTQ) pair** (built by this repo's chain, real-data PTQ calibration, accuracy-aligned with FP16: cos_sim ≥ 0.9986 on-device, L2(1s) gate 0.7805) is the **final VAD-Base quantized deployment configuration** — measured **E2E 326.7 ms at MAXN** (docs/power_modes.md §4.3). The table below keeps the parent-project matched-era Base-vs-Small structural comparison.

| Module | VAD-Base (Orin INT8) | VAD-Small (Orin INT8) | Speedup |
|---|---|---|---|
| Backbone (ResNet50+FPN) | 47.9 ms | 25.6 ms | 1.87× |
| Head (BEVFormer-style + decoders) | 678.8 ms | 35.8 ms | **19.0×** |
| **End-to-end** | **726.7 ms** | **61.3 ms** | **11.9×** |

> Re-benchmark note (2026-10-20): the 61.3 ms above is the parent-project report. A fresh re-benchmark of the same Orin engines with this repo's power-mode harness measured **63.7 ms (MAXN) / 87.4 ms (50 W)** — see [power_modes.md](power_modes.md) §4.2 (§3 there reconciles the 50 W difference).

Output-fidelity check for VAD-Small (TRT INT8 head engine vs PyTorch GPU reference on the same inputs, 10 real Bench2Drive frames): **mean cosine similarity 0.99998** (min 0.9996) across all 9 outputs — i.e. quantization is loss-free at output level. This is an output-fidelity measurement, **not** a closed-loop driving score.

Two caveats on the table above:

1. The VAD-Base column is the *scalar-plugin* INT8 build (OptLevel=2). A faster VAD-Base head build exists (302.2 ms head with the vectorized `opt8` plugin and OptLevel=5), but it is **not production-usable** — that plugin combination has a non-deterministic NaN bug at VAD-Base size on Orin (~55% of runs; see `docs/roofline_analysis.md` §5). 726.7 ms is the accuracy-clean VAD-Base baseline.
2. Parameters shrink only 1.46× while latency shrinks ~12×, because head latency is driven by **activation volume / memory bandwidth**, not by weight count (see `docs/roofline_analysis.md`).

## 4. What "fast" means — and does not mean

The VAD-Small checkpoint was trained for **2 epochs on a 100-frame subset** of Bench2Drive (source: external zoo repo `work_dirs/vad_tiny_e2e_b2d_fast/epoch_2.pth`). This is a *fast verification training run*, used to:

- validate the complete deployment chain (export → plugins → FP16/INT8 build → eval) on a small model with the same I/O contract as VAD-Base;
- provide a small-model baseline for the INT8 track;
- measure how much of the end-to-end latency is structural (see §5).

It is explicitly **not**:

- a fully trained model;
- competitive on closed-loop Bench2Drive metrics — do not cite its driving scores;
- a claim that "VAD-Small achieves 61.3 ms at production accuracy". The 61.3 ms is a *structure-determined deployment* measurement with a 2-epoch weight.

## 5. What a fully trained VAD-Small would look like

Qualitative expectations only — no numbers are invented here:

- **Deployment cost is structure-determined.** The engine's compute graph, plugin set, and activation volumes do not depend on weight values. A fully trained VAD-Small would deploy through this repo's chain unchanged and hit the **same ~64 ms-class Orin E2E at MAXN** (63.7 ms; 87.4 ms at 50 W — assuming the same precision settings and plugin set).
- **Accuracy is not structure-determined.** Closed-loop driving scores require full training on the complete Bench2Drive training split. The 2-epoch checkpoint says nothing about what a fully trained VAD-Small would score.
- **The quantization behavior should carry over** (the calibration statistics depend on activation distributions, which are similar for a converged model), but recalibration on the new checkpoint is required and the fidelity numbers in §3 should be re-measured.

## 6. How to (re)train

Training happens in the external Bench2Drive zoo repository, not in this repo. This repo's `configs/` directory holds **deployment/export configs** (flattened, self-contained copies used by the ONNX export and engine-build steps); the training configs live in the zoo repo.

1. Set up the zoo repo and the Bench2Drive dataset — see `docs/environment.md` (§ External training repository).
2. Train with the zoo-repo training config `adzoo/vad/configs/VAD/VAD_tiny_e2e_b2d_fast.py` (dev naming). For a production-grade small model, extend the schedule well beyond the 2-epoch fast run used for the shipped checkpoint.
3. Copy the resulting checkpoint into `ckpts/` and regenerate calibration data (`tools/run_gen_calib.sh`) before building INT8 engines — calibration statistics are checkpoint-specific.
4. If you retrain VAD-Base, the same steps apply with the zoo repo's `VAD_base_e2e_b2d.py` config.

## 7. Reference: official VAD-Base vs VAD-Tiny (nuScenes)

For orientation: the architecture recipe of this repo's VAD-Small (BEV 200×200→100×100, 6→3 BEV-encoder layers, 4→1 FPN levels, 6→3 decoder layers, width and query counts unchanged) is the same recipe as the official VAD-Tiny (§1). The official release reports open-loop nuScenes planning metrics, so the numbers below are not comparable to Bench2Drive closed-loop numbers — they are a public sanity anchor for "what this structural shrink costs in the original paper setting".

| Model | L2 (1s) | L2 (2s) | L2 (3s) | Col (1s) | Col (2s) | Col (3s) | FPS |
|--|--|--|--|--|--|--|--|
| VAD-Base | 0.41 | 0.70 | 1.05 | 0.07 | 0.17 | 0.41 | 4.5 |
| VAD-Tiny | 0.46 | 0.76 | 1.12 | 0.21 | 0.35 | 0.58 | 16.8 |

The official Base→Tiny step costs **+0.05 m L2(1s)** (0.41→0.46, ~12% relative) and gains **3.7× FPS** (4.5→16.8) on nuScenes. Because VAD-Small's recipe equals Tiny's, this is the best public proxy for the expected Base→Small gap on Bench2Drive — though the two settings (open-loop nuScenes vs closed-loop Bench2Drive) differ, so treat it as an order-of-magnitude guide only. Validating the actual Bench2Drive gap requires a fully trained VAD-Small checkpoint (§6).

### Sources

- Planning table in the official VAD repository README (HustVL/VAD), retrieved 2026-10.
- Structure recipe per §1/§2; official checkpoint parameter counts are not published by the VAD release, so this repo reports only locally measured parameter counts (§2).

## References

- Parameter counts, structure table, Orin latency table: parent-project development reports (`VAD-Tiny quantization deployment report`, `README` §6.3/§11); parameter counts measured from checkpoint `state_dict`.
- VAD-Small output fidelity: `tiny_head_opt3_cos_sim.json`-class measurement artifacts of the parent project (mean cos_sim 0.99998, 10 frames, INT8 head vs PyTorch).
- Structure duplication used by deployment: flattened configs under `configs/`.
- External: [VAD (hustvl/VAD)](https://github.com/hustvl/VAD), [Bench2Drive (Thinklab-SJTU/Bench2Drive)](https://github.com/Thinklab-SJTU/Bench2Drive).
