# Environment Setup

Hardware, software versions, and machine roles for running this pipeline. Two machines are involved: an **x86_64 development host** (export, quantization, x86 engines, eval) and an **NVIDIA Jetson AGX Orin** target (deployment engines and eval). An **external training repository** (the Bench2Drive zoo repo) supplies training and is only needed if you retrain models.

> Paths in the scripts are resolved via environment variables (`VAD_PROJECT_ROOT`, `VAD_DEPLOY_ROOT`) — set them to your checkout locations. `scripts/detect_trt_versions.sh` inventories whatever TensorRT installs exist on the current machine.

## 1. x86_64 development host

Development-machine configuration (a consumer GPU is sufficient; any CUDA 11.8-capable Ada/Ampere card works — the dev host used an RTX 4090, `sm_89`):

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 4090 (Ada, `sm_89`) |
| CUDA | 11.8 |
| TensorRT (tarball builders) | 10.4 GA (`10.4.0.26`) — primary build path |
| TensorRT (other installs present) | 8.6.1.6, 10.7, 10.16 (see §4) |
| Conda env | `b2d_zoo` (Python 3.8) — the zoo training environment (see §5) |

Conda environment `b2d_zoo`, the package set relevant to this pipeline (verify with `pip list`; the versions the released engines were produced with):

| Package | Version | Used for |
|---|---|---|
| torch | 2.4.1+cu118 | ONNX export, calibration data generation |
| tensorrt (python) | 10.16.1.11 (lean/dispatch 10.4.0) | export-side tooling, eval scripting |
| onnx | 1.17.0 | export |
| onnx-graphsurgeon | 0.5.2 | graph patching (patches/ directory) |
| onnxsim | 0.4.36 | ONNX simplification |
| onnxruntime | 1.19.2 | ONNX-side numeric checks |
| mmcv | mmcv-full source build (pip shim reports 0.0.1) | zoo model definition |
| mmdet / mmdet3d / mmengine | zoo checkout | imported from the zoo repo via PYTHONPATH, not pip |
| numpy | 1.20.0 | calibration npz handling |

## 2. The two-TensorRT reality (important)

There are two TensorRT installations on the dev host, and they serve different roles:

1. **Conda `tensorrt` python package** — used by export/quantization tooling and for script-side convenience.
2. **Tarball TensorRT** (`trtexec`, C++ libraries, plugin build against `TRT_ROOT/include` / `TRT_ROOT/lib`) — used for **engine building and runtime**.

The binding rule: **an engine must be deserialized by a runtime of the same TensorRT version family it was built with.** Point `LD_LIBRARY_PATH` at the tarball's `lib/` directory that matches your engine's build version. `scripts/detect_trt_versions.sh` lists all candidate installs (it scans `/opt/TensorRT-*`, `/usr/local/TensorRT*`, `/usr/src/TensorRT*`, and python packages) — run it first when debugging a deserialize/load failure.

## 3. Jetson AGX Orin target

| Component | Version |
|---|---|
| Module | Jetson AGX Orin 64 GB (`sm_87`, Ampere) |
| JetPack | 6.x (L4T r36.x, Ubuntu 22.04) |
| TensorRT | 10.3.0 (system, pip dist-packages) + a user-local 10.4 install |
| `trtexec` | `/usr/src/tensorrt/bin/trtexec` |

Notes:

- Both TRT 10.3 and 10.4 have been used on the Orin; engine artifacts record which built them (e.g. `*_trt104.engine` suffixes). Keep runtime and builder versions matched per §2.
- Plugins are compiled per-architecture: x86 engines need the `sm_89` plugin `.so`, Orin engines need `sm_87` (see `plugins/`, `x86/plugins_build/`, `arm64/plugins_build/`).
- Power modes (MAXN / 50 W / ...) measurably change latency — see `docs/power_modes.md`; the repo's Orin numbers use the 50 W mode.

### Host ↔ Orin connectivity

- The Orin is reached over the USB device-mode bridge (L4T default `192.168.55.0/24`; Orin side `192.168.55.1`). An ssh config alias `orin` points at it.
- Sync with `scripts/sync_to_orin.sh` (rsync, deliberately **no `--delete`**): `bash scripts/sync_to_orin.sh [host] [remote_dir]`.
- **Clock drift caveat (observed on this unit):** this Orin unit's system clock can lag by weeks (no RTC battery / NTP on the test bench) - remote file mtimes look stale even right after a sync. Verify fresh artifacts by size or content hash, not by mtime; on-device build logs' timestamps likewise.
- USB-bridge throughput is limited (~292 Mb/s measured over rsync on this link). Large checkpoints (the base checkpoint is ~667 MB ≈ 20 minutes at that rate) are **excluded from the default sync list** — transfer them via USB drive or shared storage.

### 3.1 Wheel-vs-tarball libnvinfer mismatch (observed on this unit)

The Orin has a JetPack system `libnvinfer.so.10` (10.3.0.30, dpkg `libnvinfer-dev`) and a pip `tensorrt` 10.4.0 python wheel. At `import tensorrt` the wheel resolves `libnvinfer.so.10` from the system path, so a **10.4 binding talks to a 10.3 native library**; deserialization then fails with

```
IRuntime::deserializeCudaEngine: ... The engine plan file is not compatible with
this version of TensorRT, expecting library version 10.3.0.30 got <...>
```

even for an engine built locally on the same device. The fix is to make the wheel load the tarball's 10.4 native libs:

```bash
export LD_LIBRARY_PATH=/usr/src/TensorRT-10.4.0.26/lib:${LD_LIBRARY_PATH:-}
```

`arm64/eval/run_eval_{fp16,int8}_orin.sh` now auto-detect `ORIN_TRT_ROOT` when it is not exported and prepend its `lib/` to `LD_LIBRARY_PATH` (verified: both FP16 engines deserialize OK with the tarball lib, FAIL without it).

## 4. TensorRT version matrix

| Pipeline stage | TRT version used |
|---|---|
| ONNX export (torch 2.0 + onnx 1.17 + graphsurgeon) | TRT-version independent |
| x86 engine builds + runtime | TRT 10.4 GA tarball |
| Orin engine builds + runtime | TRT 10.3.0 (system) and/or 10.4 (user install) |
| Legacy engines from early bring-up | TRT 8.6.1.6 (engines from 8.x do **not** load in 10.x) |

Cross-platform portability rules (learned the hard way, see parent-project reports):

- **Calibration caches (`.calib`) are portable** across platforms and TRT versions — they are plain activation-range tables. Commit them, reuse them (`data/calib_data/`).
- **Engines (`.engine`) are NOT portable** — neither across GPU architectures (`sm_89` vs `sm_87`) nor across TRT major versions. Rebuild per platform.

## 5. External training repository (retraining only)

Training does not happen in this repository. The models were trained in a Bench2Drive zoo repository (mmcv/mmdet3d-based), referenced from scripts via `VAD_PROJECT_ROOT`. The conda env this pipeline runs in is simply the zoo repo's development environment: **set it up by following the upstream docs, not custom instructions here** (this project's instance is named `b2d_zoo`; the name is arbitrary — scripts take `CONDA_ENV`).

1. Build the zoo env per upstream [INSTALL.md](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/INSTALL.md) (mmcv-full CUDA ops need a source build against your CUDA version).
2. Full dataset per upstream [DATA_PREP.md](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/DATA_PREP.md) — only needed for retraining; pipeline-only use needs just the bundled mini-set (`data/README.md`).
3. **Sanity gate:** run one upstream eval or training job per [TRAIN_EVAL.md](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/TRAIN_EVAL.md). Once an upstream job runs in your env, every zoo-importing stage of this pipeline (ONNX export, calibration data, eval) imports cleanly.
4. Then point this pipeline at the env: `CONDA_ENV=<name>` (+ `VAD_PROJECT_ROOT=<zoo-path>` if the zoo checkout is not next to this repo or in `$HOME` — `run_pipeline.sh` searches those locations automatically).
5. Training configs: `adzoo/vad/configs/VAD/` in the zoo repo (see `docs/model_variants.md` §6 for the config names and the naming bridge).

## 6. Verification checklist

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import tensorrt as trt; print(trt.__version__)"
bash scripts/detect_trt_versions.sh          # inventory all TRT installs
trtexec --version                            # builder TRT (tarball)
ssh orin 'nvpmodel -q && trtexec --version'  # target reachability + TRT
```

## 7. References

- NVIDIA TensorRT download/support matrix: <https://developer.nvidia.com/tensorrt>
- NVIDIA JetPack documentation: <https://developer.nvidia.com/embedded/jetpack>
- NVIDIA Jetson AGX Orin Technical Brief (hardware specs): <https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf>
- Bench2Drive: <https://github.com/Thinklab-SJTU/Bench2Drive> (paper: <https://arxiv.org/abs/2406.03877>)
- VAD: <https://github.com/hustvl/VAD> (paper: <https://arxiv.org/abs/2303.12077>)
- Parent-project environment notes and platform reports (`docs/` of the development repository).
