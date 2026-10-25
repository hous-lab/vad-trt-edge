# Checkpoints

Weights used by the deployment pipeline. Both files are hundreds of MB, so they are **not stored in this repository** — this file records how to obtain each one and how to verify it after download. Place downloaded files in this directory ([ckpts/](.)). The in-house weight lives on the project's Hugging Face repo **[hous-lab/vad-trt-edge](https://huggingface.co/hous-lab/vad-trt-edge)** (files + SHA-256 table below; `vad_b2d_base.pth` stays upstream-only).

| File | Size | SHA-256 |
|---|---|---|
| `vad_b2d_base.pth` | 667 MB (699,792,372 B) | `0f2d48b5c6cdb06d0dffb21612791fe740f3f5c7273b3b5e78e1f19e4a527eab` |
| `VAD_small_b2d_fast.pth` | 455 MB (477,441,203 B) | `385be4c66d8ed264560b27ffd721f0ecd639e34e71bba8f6df895437c8616683` |

Verify after download:

```bash
sha256sum vad_b2d_base.pth
```

## 1. `vad_b2d_base.pth` — download from the upstream Bench2DriveZoo release

This checkpoint was **not trained by this project**: it is taken directly from the upstream [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) release (VAD row of their model table). Download from either mirror:

- Hugging Face: <https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/vad_b2d_base.pth>
- Baidu Cloud: <https://pan.baidu.com/s/1rK7Z_D-JsA7kBJmEUcMMyg?pwd=1234> (extract code `1234`)

It ships with the upstream `VAD_base_e2e_b2d` training config (`adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py`). It is the starting point for the whole pipeline — the accuracy gates (FP16 0.727 / INT8 0.7805 L2(1s)) are anchored on this exact checkpoint.

## 2. `VAD_small_b2d_fast.pth` — in-house VAD-Small (cloud download)

VAD-Small trained with a 2-epoch fast schedule on a 100-frame subset — an efficiency-validation weight, **not a competitive model** (see [docs/model_variants.md](../docs/model_variants.md) §4). Distributed via the project's Hugging Face repo:

- **Download link: <https://huggingface.co/hous-lab/vad-trt-edge/blob/main/VAD_small_b2d_fast.pth>** (mirror: <https://hf-mirror.com/hous-lab/vad-trt-edge/blob/main/VAD_small_b2d_fast.pth>)

To reproduce: set up the training environment ([docs/environment.md](../docs/environment.md) §5) and run the `VAD_small_b2d_fast` schedule from [configs/VAD_small_b2d_fast.py](../configs/VAD_small_b2d_fast.py) in the zoo repository.

## Notes

- Checkpoint file sizes include FP32 weights **plus optimizer state** (AdamW), so they are larger than the parameter counts suggest (VAD-Base: 58.38 M params; VAD-Small: 39.89 M params).
- All checkpoints share the same head I/O contract; any of them flows through the same export → build → eval chain. VAD-Small additionally needs its calibration data regenerated if you retrain (calibration statistics are checkpoint-specific — [tools/run_gen_calib.sh](../tools/run_gen_calib.sh)).

## References

- Variant structure and naming: [docs/model_variants.md](../docs/model_variants.md)
- Training-environment setup: [docs/environment.md](../docs/environment.md) §5
- Architecture/code upstream: [VAD](https://github.com/hustvl/VAD) (Apache-2.0), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- Weights release: [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo)
