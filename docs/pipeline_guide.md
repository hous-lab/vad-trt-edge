# Pipeline Guide — full x86 reproduction manual

Step-by-step manual for reproducing the entire chain on the x86 dev host. The one-shot entry point `run_pipeline.sh` automates stages 1-4; this guide explains what each stage does, what it produces, and how to verify it. Orin-side steps are in §7.

All stages are driven by environment variables (all have defaults next to the repo):

| Variable | Meaning | Example |
|---|---|---|
| `TRT_ROOT` | TRT 10.x tarball root (builders + plugin build + runtime lib) | `/usr/src/TensorRT-10.4.0.26` |
| `CUDA_HOME` | CUDA 11.8 toolkit root (nvcc + libcudart for plugin build) | `/usr/local/cuda-11.8` |
| `CONDA_BASE` / `CONDA_ENV` | conda location / env name (torch cu118 + tensorrt python) | `$HOME/apps/miniconda3` / `b2d_zoo` |
| `VAD_PROJECT_ROOT` | external training (zoo) repo | sibling dir `Bench2DriveZoo_uniad_vad` |
| `VAD_DEPLOY_ROOT` | this repo root (scripts auto-set it) | repo root |
| `VAD_BACKBONE_ENGINE` / `VAD_HEAD_ENGINE` | eval engine overrides | output engines |

Only `TRT_ROOT` and `CUDA_HOME` are hard requirements (the one-shot script refuses to start without them); everything else has an automatic default: `VAD_DEPLOY_ROOT` from the script location, `VAD_PROJECT_ROOT` auto-discovered (parent-of-parent, sibling, `$HOME`), `CONDA_BASE` with a `$HOME/miniconda3` fallback, `CONDA_ENV` default `b2d_zoo`, and eval engine paths default into `x86/trt_compile/output/`. The conda env itself is the upstream zoo training environment — build it per [INSTALL.md](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/INSTALL.md) (see docs/environment.md §5 for the sanity-gate recipe).

## 1. Inputs (checkpoint + dataset)

1. Place the checkpoint: `ckpts/vad_b2d_base.pth` (links: ckpts/README.md).
2. Place the Bench2Drive mini-set: `data/bench2drive/{maps,v1}/` (guide: data/README.md). The mini-set's 173-frame eval split is `data/splits/bench2drive_base_train_val_split.json`.
3. (INT8 only) Calibration npz files: regenerable via `tools/run_gen_calib.sh` (~11 GB backbone + ~26 GB head npz). This repo ships only the small `.calib` caches; `data/calib_data/*.npz` may be symlinks to elsewhere.

## 2. Stage 1 — TRT plugins

```bash
TRT_ROOT=/usr/src/TensorRT-10.4.0.26 CUDA_HOME=/usr/local/cuda-11.8 \
    bash x86/plugins_build/build_x86.sh
```

Produces `plugins/build/libplugins.so`. Verify:

```bash
readelf -d plugins/build/libplugins.so | grep NEEDED   # expect libnvinfer.so.10
LD_LIBRARY_PATH=plugins/build:$TRT_ROOT/lib:$CUDA_HOME/lib64 \
  python -c "import ctypes; ctypes.CDLL('plugins/build/libplugins.so'); print('ok')"
```

Why the checks matter (a real failure mode): the plugin .so must be compiled by the **same CUDA major version as the toolkit you link**. nvcc 11 emits `cudaLaunchKernel` (C API, in cudart 11); nvcc 12 emits `__cudaLaunchKernel` (only in cudart 12). A plugin built by nvcc 12 but linked against cudart 11 fails to dlopen with "undefined symbol: __cudaLaunchKernel". `build_x86.sh` pins the CUDA compiler and link dirs from `TRT_ROOT`/`CUDA_HOME` so cmake cannot silently pick the wrong toolkit.

## 3. Stage 2 — ONNX export

```bash
(cd onnx_export && python scripts/export_vad_prev.py)       # demo_data (real two-pass prev_bev)
(cd onnx_export && python scripts/export_backbone_model.py) # vad_backbone.onnx + _b1
(cd onnx_export && python scripts/export_head_script.py)    # vad_head_withprev.onnx
```

Run from an env with torch+cu118 and `VAD_PROJECT_ROOT` set. Expected outputs:

- `onnx_export/demo_data/b2d_vadv1_prev.pts_bbox_head.forward/` — per-frame input bins (4 mlvl_feats + 3 img_metas + prev_bev) used by engine builds and eval.
- `onnx_export/output/vad_backbone.onnx` (107 MB, input [6,3,736,1280]) plus `vad_backbone_b1.onnx` (batch-first [1,6,3,736,1280]); both pass a ≤1e-2 max-abs-err numeric check vs PyTorch on 3 frames.
- `onnx_export/output/vad_head_withprev.onnx` (131 MB). **prev_bev must be static [40000, 1, 256]** — the INT8 builders set no optimization profile by design (see §6 troubleshooting).

Verify head input shapes:

```bash
python -c "import onnx; m=onnx.load('onnx_export/output/vad_head_withprev.onnx'); \
[print(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]"
```

## 4. Stage 3 — TRT engine builds

From `x86/trt_compile/scripts/` (env as above + `TRT_ROOT` in `LD_LIBRARY_PATH`):

```bash
BUILD_MODE=mixed python build_head_v2.py          # -> head_withprev_fp16_fixed_v2.engine (~80 MB)
BUILD_MODE=int8_real python build_backbone_int8.py # -> backbone_int8.engine (~53 MB)
python build_head_int8.py                          # -> head_int8_fixed.engine (~55 MB)
```

Notes:

- All three load `plugins/build/libplugins.so` via ctypes (RTLD_GLOBAL).
- The INT8 builders read `data/calib_data/*.calib` caches. **The calibration cache is keyed by network signature** — after any network change (e.g. making prev_bev static), TRT will report a cache miss and re-run calibration from the npz tensors, then rewrite the cache file. Budget ~10 min calibration + ~15 min tactic timing per head build on a 4090 Laptop.
- Expected builder log signature on cache hit: `Reading Calibration Cache ... Generated calibration scales using calibration cache`.
- The FP16 gate uses the *reference* FP16 pair (built 2026-06/08): stage 3 builds the backbone only as INT8, so a fresh backbone_fp16.engine is not part of this pipeline.

## 5. Stage 4 — evaluation (accuracy gates)

```bash
cd x86/eval
bash run_eval_int8_bb.sh       # INT8 backbone + FP16 head  -> tag int8_bb_fp16_head
bash run_eval_int8_fixed.sh    # INT8 backbone + INT8 head  -> L2 gate 0.7805
bash run_eval_full_int8.sh     # INT8 backbone + mixed head -> tag full_int8
```

Or directly (env overrides; the eval defaults resolve to this repo's engines):

```bash
# FP16 reference-engine eval - point VAD_*_ENGINE at your reference pair, e.g.
VAD_BACKBONE_ENGINE=/path/to/reference/backbone_fp16.engine \
VAD_HEAD_ENGINE=/path/to/reference/head_withprev_fp16_fixed_v2.engine \
python eval_pytorch_vs_trt.py
```

| Eval config | Expected L2(1s) | Result file |
|--|--|--|
| FP16 pair, reference engines (deterministic across reruns) | **0.7267** | trt_static_prevbev_results.json; repro_2026-10-16_refengines_fp16.json |
| FP16 head rebuild on reference backbone | 0.7411 | repro_2026-10-16_refbb_newfp16head.json |
| INT8 pair, reference build | 0.7805 | eval_int8_fixed.json |
| INT8 pair, fresh rebuild (new calib cache) | **0.7630** | repro_2026-10-16_int8_rebuilt_engines.json |

**Engine-rebuild variance (measured):** with the *reference* engines the eval reproduces its numbers exactly (engines are deterministic for fixed inputs; FP16 pair: 0.7267). Rebuilding shifts results: the rebuilt FP16 head on the reference backbone measured 0.7411 (+0.014), and the rebuilt INT8 pair with the recalibrated head cache measured 0.7630 (-0.018 vs the reference-build INT8 0.7805 -- *better*, the new cache helped). TRT re-selects tactics per build and the prev_bev temporal feedback loop compounds small output deltas. Note this pipeline's stage 3 builds the backbone only as INT8; the FP16 eval uses the reference backbone engine. One caveat: `run_pipeline.sh` stage 4 defaults to the **INT8** engines -- always confirm which engines a result belongs to (the eval header always prints "FP16 Backbone + FP16 Head" regardless of the actual engines; the printed `Backbone:`/`Head:` paths are authoritative). Treat +/-0.02 around each configuration's measured value as the realistic cross-rebuild band.

## 6. Troubleshooting (real failure modes hit while reproducing)

| Symptom | Cause | Fix |
|--|--|--|
| `libplugins.so: undefined symbol: __cudaLaunchKernel` | plugin compiled by nvcc 12 but linked cudart 11 (11.x cudart lacks that symbol) | rebuild via `build_x86.sh` with `TRT_ROOT`+`CUDA_HOME` set — it pins nvcc + link dirs |
| `libnvinfer.so.8` in DT_NEEDED | cmake linked system TRT 8.6 instead of tarball TRT 10 | same fix — `build_x86.sh` forwards `TRT_ROOT` and `-L$TRT_ROOT/lib` |
| engine build: "no optimization profile has been defined" | head ONNX exported with dynamic `prev_bev` | keep prev_bev static in `export_head_script.py` (this repo's copy does) |
| `ModuleNotFoundError: mmcv._ext` from system python | conda activation skipped (CONDA_BASE wrong) | scripts use `${CONDA_ENV:-b2d_zoo}` + `CONDA_BASE` auto-fallback; verify `conda activate b2d_zoo` imports zoo code |
| zoo import fails at `iou3d_boxes_iou_bev_forward` assert | wrong conda env (p1bevrc ABI mismatch) | use `b2d_zoo` — the only env whose mmcv `_ext` matches the zoo repo |
| engine deserialize fails | TRT runtime ≠ builder TRT family | see docs/environment.md §2; `scripts/detect_trt_versions.sh` |
| INT8 L2 catastrophic (>1.0) | calib data not real frames (random tensors) | use real-frame calib (tools/gen_calib_data.py); see roofline doc §3.3 |

## 7. Orin quick section

```bash
bash scripts/sync_to_orin.sh [host] [remote_dir]   # rsync, no --delete

# on the Orin (your deploy dir, e.g. /data/home/<user>/vad_deploy_pipeline):
export ORIN_TRT_ROOT=/usr/src/TensorRT-10.4.0.26
bash arm64/plugins_build/build_orin.sh
bash arm64/trt_compile/fp16/run_build_fp16_orin.sh          # backbone + head FP16
bash arm64/trt_compile/int8/run_build_int8_orin.sh          # backbone + head INT8

# smoke eval (engines + real calib frames; cos_sim gate 0.99 aligned / 0.95 acceptable)
python arm64/eval/eval_orin_lite.py \
  --backbone-engine arm64/trt_compile/output/backbone_fp16_orin.engine \
  --head-engine arm64/trt_compile/output/head_fp16_orin.engine \
  --plugin-lib arm64/plugins_build/build/libplugins_orin.so \
  --calib-npz data/calib_data/head_calib.npz --num-frames 10
```

Verified on the bench unit (AGX Orin 64 GB, MAXN): the FP16 build (backbone 54.1 MB + head 79.0 MB) completes cleanly (zero errors in the build log) and the 10-frame head-only smoke passes at 286.1 ms avg (3.49 FPS) — result file `arm64/eval/results/orin_fp16_smoke_maxn_2026-10-16.json` (the smoke harness measures the head engine only) — per-mode (50/30/15 W) benchmark tables: docs/power_modes.md §4. 30 W/15 W cannot be entered live on this unit (reboot required) — see docs/power_modes.md §4.1.

The INT8 build (`run_build_int8_orin.sh`, step 0 links the calib npz) is the **VAD-Base INT8 (PTQ) deployment pair**: real-data calibration (EntropyCalibration2, 64 real frames), MatMul chain pinned FP16, sensitive layers FP32 — the same recipe as the x86 INT8 gate (§5). Calibration is cache-first: both builders reuse the committed `data/calib_data/{backbone,head}_int8_orin.calib` when present (the exact calibration state of the reference engines); if absent, they recalibrate on-device from `data/calib_data/head_calib.npz` — the fallback is not theoretical: the reference head engine's cache was produced exactly that way on this unit (the x86-cache copy arrived under the wrong filename, so the builder fell back; the on-device cache has the same size and the engine passed the fidelity gate below).

Reference engines measure 28.0 MB (backbone) + 79.2 MB (head). Verified accuracy on this unit — 10-frame head-only fidelity vs the x86 FP16 reference (GT collected once on the x86 host via `arm64/eval/collect_gt_x86.py`; the 398 MB GT npz is gitignored, regenerate it there):

```bash
# on the x86 host (conda env): collect the FP16 GT once
python arm64/eval/collect_gt_x86.py    # -> arm64/eval/results/gt_x86_fp16.npz

# on the Orin: INT8 fidelity eval (gate: cos_sim > 0.99 aligned / > 0.95 acceptable)
python arm64/eval/eval_orin_lite.py \
  --backbone-engine arm64/trt_compile/output/backbone_int8_orin.engine \
  --head-engine arm64/trt_compile/output/head_int8_orin.engine \
  --plugin-lib arm64/plugins_build/build/libplugins_orin.so \
  --calib-npz data/calib_data/head_calib.npz \
  --gt-npz arm64/eval/results/gt_x86_fp16.npz --num-frames 10 \
  --output arm64/eval/results/orin_int8_ptq_lite.json   # measured: min cos_sim 0.9986
```

Two Orin-specific gotchas hit during bring-up (both now handled in scripts):
- the pip `tensorrt` wheel (10.4.0) must load the tarball's `libnvinfer` (10.4.0.26) via `LD_LIBRARY_PATH`, or engine deserialization fails — `run_eval_*_orin.sh` auto-detect `ORIN_TRT_ROOT` now; details in docs/environment.md §3.1;
- `nvpmodel -m 1/-m 2` answers "Reboot required" on this unit — verify with `nvpmodel -q` after switching (docs/power_modes.md §4.1).

Details: docs/power_modes.md §4 (methodology + per-config measured tables for all 4 modes), docs/environment.md §3 (device + connectivity), arm64/trt_compile/{fp16,int8}/ (builders).

## 8. Autoware note

Autoware closed-loop integration (single-machine and two-machine schemes) is planned for the next phase of this project; the scope is described in README.md item 6 and the deployment engines built here are the input to that work. This repo currently ships no Autoware integration code.
