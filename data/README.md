# Data

Input data and calibration artifacts for the deployment pipeline. Everything here is either a small index file (committed) or a pointer to data you must download or regenerate (not committed).

```
data/
├── b2d_infos_val.pkl                      # eval-set frame index (173 frames)
├── b2d_map_infos.pkl                      # preprocessed map infos for the eval scenes
├── splits/
│   └── bench2drive_base_train_val_split.json
├── bench2drive/                           # Bench2Drive mini-set (291 MB, git-ignored)
│   ├── maps/Town10HD_HD_map.npz
│   └── v1/<scenario>/{anno, camera, lidar, expert_assessment}
└── calib_data/
    ├── backbone_int8.calib                # committed, portable
    ├── head_int8_fixed.calib              # committed, portable
    └── *_calib.npz                        # NOT committed (≈39.6 GB total) — regenerate
```

## 1. Index files

- **`b2d_infos_val.pkl`** — a Python list of **173 frame info dicts** covering the eval scenes (same 173 frames the x86 eval scripts report as `num_samples`). Each entry points into `data/bench2drive/` and carries camera/lidar/annotation paths and calibration metadata.
- **`b2d_map_infos.pkl`** — preprocessed vectorized map information (9.8 MB) for the HD maps used by the eval scenes.
- **`splits/bench2drive_base_train_val_split.json`** — split definition in the zoo repo's format: `{"val": ["v1/HazardAtSideLane_Town10HD_Route373_Weather9"]}`. The eval set draws from this scene.

## 2. Bench2Drive mini-set (`data/bench2drive/`, not committed)

A **291 MB subset of the official Bench2Drive dataset**: 2 scenarios (`HazardAtSideLane_Town10HD_Route373_Weather9`, `BlockedIntersection_Town10HD_Route391_Weather1`), the Town10HD HD map, and per-scenario directories `anno/`, `camera/` (six cameras: `rgb_front`, `rgb_front_left`, `rgb_front_right`, `rgb_back`, `rgb_back_left`, `rgb_back_right`, each with numbered JPEG frames), `lidar/`, and `expert_assessment/`.

It is enough to run the full deployment chain — export → engine build → calibration-data generation → L2/fidelity eval — but **not** enough for training. Download the full dataset from the official repository: <https://github.com/Thinklab-SJTU/Bench2Drive> (see license terms there). Extract so that the `maps/` and `v1/` directories land under `data/bench2drive/`.

## 3. Calibration data ([data/calib_data/](calib_data))

**Committed files** — TensorRT INT8 calibration caches produced by the `IInt8EntropyCalibrator2`-based builders:

| File | Size | Produced by |
|---|---|---|
| `backbone_int8.calib` | 7.6 KB | backbone INT8 build |
| `head_int8_fixed.calib` | 386 KB | head INT8 build (real-data, fixed-shape) |

Calibration caches are plain activation-range tables: they are **portable across platforms and TensorRT versions** and are committed so you can rebuild engines without regenerating calibration. The builders read them via `read_calibration_cache()` and skip data collection when present.

**Not committed** — the raw calibration tensors (`*_calib.npz`, **~39.6 GB**: backbone 11.7 GB + head 27.9 GB on disk). Regenerate them with:

```bash
conda activate <env>            # export environment, see docs/environment.md
bash tools/run_gen_calib.sh     # PyTorch FP32 forward on 200 real Bench2Drive frames
```

[tools/gen_calib_data.py](../tools/gen_calib_data.py) runs the trained PyTorch model on **real Bench2Drive scenes** and dumps per-tensor activations to npz; the TRT calibrators then consume the npz.

> **Do not substitute random data.** Random-noise calibration put the head's activation ranges off by ~100× and produced L2(1s) = 4.36 (vs 0.78 with real data) — see [docs/roofline_analysis.md](../docs/roofline_analysis.md) §3.3. The committed caches already come from real data.

After regenerating npz, rebuild engines to produce fresh `.calib` caches (or keep using the committed ones — see the portability note above).

## 4. References

- Calibration/quantization lessons: [docs/roofline_analysis.md](../docs/roofline_analysis.md) §3.3; parent-project INT8 reports.
- Eval scripts that consume these files: [x86/eval/](../x86/eval), [arm64/eval/](../arm64/eval); calibration generation: [tools/gen_calib_data.py](../tools/gen_calib_data.py), [tools/run_gen_calib.sh](../tools/run_gen_calib.sh).
- Bench2Drive: <https://github.com/Thinklab-SJTU/Bench2Drive>
