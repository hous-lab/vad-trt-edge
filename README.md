# VAD TensorRT Deployment on Jetson Orin

End-to-end deployment of **VAD-Base [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad%2Fvad/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py)** on NVIDIA Jetson AGX Orin (`sm_87`) and an x86_64 dev host (RTX 4090 Laptop, `sm_89`).

## Jetson Orin

| Module (MAXN p50) | VAD-Base INT8 (PTQ) | VAD-Small INT8 | Ratio |
|---|---|---|---|
| Backbone | 35.6 ms | 25.3 ms | 1.4x |
| Head | 291.1 ms | 38.4 ms | **7.6x** |
| **End-to-end** | **326.7 ms** | **63.7 ms** | **5.1x** |
| Engine footprint | 107.2 MB (28.0 + 79.2) | 66.0 MB (25.8 + 40.2) | 1.6x |
| Parameters | 58.38 M | 39.89 M | 1.46x |
| Head quantization fidelity | cos_sim ≥ 0.9986 vs x86 FP16 (all 9 outputs) | cos_sim 0.99998 vs PyTorch | accuracy-aligned on both configs |

> **Orin power-mode comparison** (all four modes measured for the VAD-Small INT8 deployment config and the VAD-Base INT8 (PTQ) deployment config, plus the VAD-Base FP16 verification pair; p50/p90 + clocks/power; 30 W / 15 W entered by rebooting into the mode) — [docs/power_modes.md](docs/power_modes.md).
> **Orin roofline analysis with evidence**: why INT8 quantization alone cannot reach real-time on the Orin for VAD-Base — the 40,000-BEV-query head runs an almost entirely FP16-pinned kernel mix (measured: zero INT8 kernels execute, DRAM at ~1/3 utilization - not a DRAM wall); see [docs/roofline_analysis.md](docs/roofline_analysis.md).

## x86_64

| Configuration | L2 (1 s) | Latency | Result file |
|---|---|---|---|
| PyTorch FP32 (reference) | 0.7300 | 233.3 ms | [log](x86/eval/results/eval_final_20260612.json) |
| TRT FP16 (static prev-BEV) | 0.7270 | 93.5 ms | [log](x86/eval/results/trt_static_prevbev_results.json) |
| INT8 backbone + FP16 head | 0.7828 | 88.9 ms | [log](x86/eval/results/eval_int8_bb_fp16_head.json) |
| INT8 backbone + INT8 head (real-data calib) | 0.7805 | 132.7 ms | [log](x86/eval/results/eval_int8_fixed.json) |

## VAD-Base INT8 vs VAD-Small INT8

| | VAD-Base | VAD-Small |
|---|---|---|
| BEV resolution (BEV queries) | 200×200 = 40,000 | 100×100 = 10,000 (**4x** fewer) |
| BEV encoder / detection decoder / map decoder layers | 6 / 6 / 6 | 3 / 3 / 3 (2x shallower) |
| FPN sampling levels | 4 | 1 (backbone C5 only) |
| Temporal queue length | 4 | 3 |
| Agent / map queries | 300 / 100 | 300 / 100 (unchanged) |
| Total parameters | 58.38 M | 39.89 M (1.46x fewer) |
| — pts_bbox_head | 31.59 M | 15.26 M (2.1x) |
| — img_backbone (ResNet50) | 23.51 M | 23.51 M (unchanged) |

> **Model differences (checkpoint-derived; full table: [docs/model_variants.md](docs/model_variants.md) §2):**
> **VAD-Small honesty note**: it was trained for only 2 epochs as a fast verification run and therefore has **no closed-loop driving score**. It exists to probe what model-structure traits buy real-time behavior under the VAD architecture.

## Quick Start (x86)

Prerequisites (full matrix in [docs/environment.md](docs/environment.md)):
- ENV: build it per upstream [INSTALL.md](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/INSTALL.md) and sanity-check with their [TRAIN_EVAL.md](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/TRAIN_EVAL.md).
- Checkpoints: [ckpts/README.md](ckpts/README.md).
- Data: [data/README.md](data/README.md).
- Our x86_64: Ubuntu 22.04 + NVIDIA GPU (Ada/Ampere class), CUDA 11.8 toolkit, TRT 10.4 GA tarball as `TRT_ROOT`.
- **Only two env vars are hard requirements — `TRT_ROOT` and `CUDA_HOME`**. Everything else auto-resolves: `CONDA_BASE` (`$HOME/apps/miniconda3`, fallback `$HOME/miniconda3`), `VAD_DEPLOY_ROOT` (from script location), `VAD_PROJECT_ROOT` (zoo repo auto-discovered next to this repo / sibling / `$HOME`; export only if yours lives elsewhere), and eval engine paths default into `x86/trt_compile/output/`.
- Every artifact that is not committed (plugins `.so`, ONNX, TRT engines, calibration data) is regenerable from the scripts in this repository.

## Hardware Platforms

### x86_64 — RTX 4090 **Laptop** GPU

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

### Orin — NVIDIA Jetson AGX Orin 64 GB

| Component | Specification |
|---|---|
| SoC | Jetson AGX Orin 64 GB, Ampere GPU (**sm_87**), 2048 CUDA cores |
| Memory | 64 GB LPDDR5, 204.8 GB/s |
| Power modes | MAXN / 50 W / 30 W / 15 W (see [docs/power_modes.md](docs/power_modes.md)) |
| TensorRT | 10.4.0.26 (tarball) — see docs/environment.md §3.1 for the wheel/tarball libnvinfer caveat |
| OS | JetPack 6.2.2, L4T 36.5.0, Linux aarch64 |

## License & Acknowledgements

Apache-2.0 (see [LICENSE](LICENSE)); third-party components and their licenses are listed in [NOTICE](NOTICE).

Built on and with thanks to:

- [VAD](https://github.com/hustvl/VAD) (hustvl) — the VAD-Base/VAD-Tiny models and the vectorized-scene-representation approach;
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) (Thinklab-SJTU, NeurIPS 2024 D&B) — closed-loop dataset/benchmark and the VAD training zoo;
- **[Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) (Thinklab-SJTU)** — the `vad_b2d_base.pth` checkpoint used throughout this pipeline is taken directly from their release (see [ckpts/README.md](ckpts/README.md) for the download links). Special thanks for training and publicly releasing it;
- NVIDIA [DL4AGX](https://github.com/NVIDIA-AI-IOT/DL4AGX) / bevformer-tensorrt — MSDeformAttn TRT plugin lineage;
- The zoo training repository referenced via `VAD_PROJECT_ROOT` (external, not redistributed).
