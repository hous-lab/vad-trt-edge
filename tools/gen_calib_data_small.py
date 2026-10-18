"""VAD-Small 真实数据校准生成器

输出: data/calib_data/calib_data_small/head_calib_small.npz, backbone_calib_small.npz
与 base 版 gen_calib_data.py 一致, 但适配 small (VAD-Small):
  - 1 级 FPN (mlvl_feats.0)
  - BEV 100x100 (prev_bev [10000,1,256])
  - grid_length = (60/100, 30/100) = (0.6, 0.3)
"""
import os, sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

PIPELINE_ROOT = os.environ.get(
    "VAD_DEPLOY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.environ.get(
    "VAD_PROJECT_ROOT",
    os.path.normpath(os.path.join(PIPELINE_ROOT, "..", "..", "Bench2DriveZoo_uniad_vad")))

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from mmcv.utils import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.utils import load_checkpoint
from torch.utils.data import DataLoader as TorchDataLoader
from mmcv.parallel import collate

CONFIG_PATH = os.path.join(PIPELINE_ROOT, "configs/VAD_small_e2e_b2d.py")
CKPT_PATH = os.path.join(PIPELINE_ROOT, "ckpts/VAD_small_b2d_fast.pth")
OUTPUT_DIR = os.path.join(PIPELINE_ROOT, "data/calib_data/calib_data_small")
NUM_FRAMES = 64
BEV_H = BEV_W = 100


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading VAD-Small PyTorch model...")
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CKPT_PATH, map_location='cpu')
    model = model.cuda().eval()
    model.video_test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = TorchDataLoader(dataset, batch_size=1, shuffle=False,
                                  num_workers=0, collate_fn=collate)

    captured = {}

    _orig_extract_feat = model.extract_feat

    def patched_extract_feat(img, *a, **kw):
        feats = _orig_extract_feat(img, *a, **kw)
        if isinstance(feats, (list, tuple)):
            captured['mlvl_feats'] = [t.detach().cpu().numpy().copy() for t in feats]
        if isinstance(img, torch.Tensor):
            captured['img'] = img.detach().cpu().numpy().copy()
        return feats

    model.extract_feat = patched_extract_feat

    def hook_head_inputs(module, args, kwargs):
        captured['head_args'] = args
        captured['head_kwargs'] = kwargs

    handle2 = model.pts_bbox_head.register_forward_pre_hook(
        hook_head_inputs, with_kwargs=True)

    head_data = {}
    bb_inputs = []
    processed = 0
    for batch_data in data_loader:
        if processed >= NUM_FRAMES:
            break
        try:
            captured.clear()
            cuda_batch = {}
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    cuda_batch[k] = v.cuda()
                elif isinstance(v, list):
                    cuda_batch[k] = [e.cuda() if isinstance(e, torch.Tensor) else e for e in v]
                else:
                    cuda_batch[k] = v
            with torch.no_grad():
                _ = model(cuda_batch, return_loss=False, rescale=True)

            if 'mlvl_feats' in captured and 'head_args' in captured:
                feat0 = captured['mlvl_feats'][0]  # [1,6,256,23,40]
                if len(captured['head_args']) >= 2:
                    img_metas = captured['head_args'][1]
                else:
                    img_metas = captured['head_kwargs'].get('img_metas', None)
                prev_bev = None
                if len(captured['head_args']) >= 3:
                    prev_bev = captured['head_args'][2]
                else:
                    prev_bev = captured['head_kwargs'].get('prev_bev', None)
                if prev_bev is not None and isinstance(prev_bev, torch.Tensor):
                    prev_bev_np = prev_bev.detach().cpu().numpy().copy()
                else:
                    prev_bev_np = np.zeros((BEV_H * BEV_W, 1, 256), dtype=np.float32)

                meta = img_metas[0] if isinstance(img_metas, list) else img_metas
                can_bus = meta.get('can_bus', None)
                if isinstance(can_bus, torch.Tensor):
                    can_bus = can_bus.cpu().numpy()
                can_bus = np.asarray(can_bus, dtype=np.float32).flatten()

                # shift (small: grid_length = (60/100, 30/100))
                delta_x = can_bus[0]
                delta_y = can_bus[1]
                ego_angle = can_bus[-2] / np.pi * 180
                grid_length_y = 60.0 / BEV_H
                grid_length_x = 30.0 / BEV_W
                translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
                translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
                bev_angle = ego_angle - translation_angle
                shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / BEV_H
                shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / BEV_W
                shift = np.array([[shift_x, shift_y]], dtype=np.float32)

                lidar2img = meta.get('lidar2img', None)
                if isinstance(lidar2img, torch.Tensor):
                    lidar2img = lidar2img.cpu().numpy()
                lidar2img = np.asarray(lidar2img, dtype=np.float32).reshape(6, 4, 4)

                head_data[f'frame_{processed:03d}'] = {
                    'mlvl_feats.0': feat0.astype(np.float32),
                    'img_metas.0[shift]': shift,
                    'img_metas.0[can_bus]': can_bus.reshape(1, 18),
                    'img_metas.0[lidar2img]': lidar2img,
                    'prev_bev': prev_bev_np.astype(np.float32),
                }
                if 'img' in captured:
                    img_np = captured['img']
                    if img_np.ndim == 5:
                        img_np = img_np.reshape(-1, *img_np.shape[2:])
                    bb_inputs.append(img_np.astype(np.float32))

            processed += 1
            if processed % 10 == 0:
                print(f"  Captured {processed}/{NUM_FRAMES}")
        except Exception as e:
            print(f"  Error frame {processed}: {e}")

    model.extract_feat = _orig_extract_feat
    handle2.remove()

    head_npz = os.path.join(OUTPUT_DIR, 'head_calib_small.npz')
    save = {}
    for fk, fd in head_data.items():
        for n, v in fd.items():
            save[f'{fk}__{n}'] = v
    np.savez(head_npz, **save)
    print(f"Head calib saved: {head_npz} ({len(head_data)} frames)")

    bb_npz = os.path.join(OUTPUT_DIR, 'backbone_calib_small.npz')
    np.savez(bb_npz, **{f'frame_{i:03d}': arr for i, arr in enumerate(bb_inputs)})
    print(f"Backbone calib saved: {bb_npz} ({len(bb_inputs)} frames)")


if __name__ == '__main__':
    main()
