"""
真实数据校准生成器
输出: data/calib_data/head_calib.npz, data/calib_data/backbone_calib.npz

关键设计: prev_bev 必须用上一帧 head 的真实 bev_embed 输出
"""
import os, sys, ctypes
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# 路径配置
PIPELINE_ROOT = os.environ.get("VAD_DEPLOY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.environ.get("VAD_PROJECT_ROOT", os.path.normpath(os.path.join(PIPELINE_ROOT, "..", "..", "Bench2DriveZoo_uniad_vad")))
DEPLOY_ROOT = PIPELINE_ROOT

# 设置 LD_LIBRARY_PATH (参考 eval 脚本第 11-13 行)
_torch_lib = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages', 'torch', 'lib')
if os.path.isdir(_torch_lib):
    os.environ['LD_LIBRARY_PATH'] = _torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from mmcv.utils import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.utils import load_checkpoint
from torch.utils.data import DataLoader as TorchDataLoader
from mmcv.parallel import collate

# 配置
CONFIG_PATH = "adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py"
CKPT_PATH = os.path.join(DEPLOY_ROOT, "ckpts/vad_b2d_base.pth")
OUTPUT_DIR = os.path.join(DEPLOY_ROOT, "data/calib_data")
NUM_FRAMES = 200  # 校准帧数


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading PyTorch FP32 VAD model...")
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CKPT_PATH, map_location='cpu')
    model = model.cuda().eval()
    model.video_test_mode = True

    # 数据加载器
    dataset = build_dataset(cfg.data.test)
    data_loader = TorchDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    # 收集每帧数据
    backbone_inputs = []  # 6×3×H×W per frame
    head_data_per_frame = {}

    # 关键: 通过 hook 捕获 Head 的输入和输出
    captured = {}

    # Hook Backbone 输入和输出 (extract_feat 的整体 hook 不便, 改为 hook img_backbone+img_neck)
    # 采用直接 hook 整个 model 的 forward 之前是不行的, 我们 hook extract_feat
    def hook_extract_feat(module, args, kwargs, output):
        # output 是 mlvl_feats (list of tensors)
        if isinstance(output, (list, tuple)):
            captured['mlvl_feats'] = [t.detach().cpu().numpy().copy() for t in output]
        # args[0] 是 img tensor
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            captured['img'] = args[0].detach().cpu().numpy().copy()
        elif 'img' in kwargs and isinstance(kwargs['img'], torch.Tensor):
            captured['img'] = kwargs['img'].detach().cpu().numpy().copy()

    # Hook Head 的 forward 来捕获其参数
    def hook_head_inputs(module, args, kwargs):
        # mlvl_feats, img_metas, prev_bev, ...
        # 注意 forward 签名: forward(mlvl_feats, img_metas, prev_bev=None, ...)
        captured['head_args'] = args
        captured['head_kwargs'] = kwargs

    # Hook Head 的输出
    def hook_head_output(module, args, kwargs, output):
        # output 是 dict, 含 bev_embed
        if isinstance(output, dict) and 'bev_embed' in output:
            captured['bev_embed'] = output['bev_embed'].detach().cpu().numpy().copy()

    # 注册 hook: extract_feat 是 model 的方法, 但 register_forward_hook 是 Module 方法
    # 这里 hook 整个 model 的 forward 输出无法捕获 mlvl_feats, 改为 hook extract_feat 子模块
    # VAD model 中 extract_feat 调用 img_backbone+img_neck, mlvl_feats 是其返回
    # 直接 patch extract_feat 方法更稳妥
    _original_extract_feat = model.extract_feat

    def patched_extract_feat(img, *a, **kw):
        feats = _original_extract_feat(img, *a, **kw)
        if isinstance(feats, (list, tuple)):
            captured['mlvl_feats'] = [t.detach().cpu().numpy().copy() for t in feats]
        if isinstance(img, torch.Tensor):
            captured['img'] = img.detach().cpu().numpy().copy()
        return feats

    model.extract_feat = patched_extract_feat

    handle2 = model.pts_bbox_head.register_forward_pre_hook(hook_head_inputs, with_kwargs=True)
    handle3 = model.pts_bbox_head.register_forward_hook(hook_head_output, with_kwargs=True)

    print(f"Capturing {NUM_FRAMES} frames...")
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

            # 保存这一帧的数据
            if 'mlvl_feats' in captured and 'head_args' in captured:
                # mlvl_feats: list of [1, 6, 256, H, W]
                feat0 = captured['mlvl_feats'][0]
                feat1 = captured['mlvl_feats'][1] if len(captured['mlvl_feats']) > 1 else None
                feat2 = captured['mlvl_feats'][2] if len(captured['mlvl_feats']) > 2 else None
                feat3 = captured['mlvl_feats'][3] if len(captured['mlvl_feats']) > 3 else None

                # 从 head_args 提取 img_metas
                if len(captured['head_args']) >= 2:
                    img_metas = captured['head_args'][1]
                else:
                    img_metas = captured['head_kwargs'].get('img_metas', None)

                # 提取 prev_bev (可能在 args[2] 或 kwargs['prev_bev'])
                prev_bev = None
                if len(captured['head_args']) >= 3:
                    prev_bev = captured['head_args'][2]
                else:
                    prev_bev = captured['head_kwargs'].get('prev_bev', None)

                if prev_bev is not None and isinstance(prev_bev, torch.Tensor):
                    prev_bev_np = prev_bev.detach().cpu().numpy().copy()
                else:
                    # 第一帧没有 prev_bev, 使用零
                    prev_bev_np = np.zeros((40000, 1, 256), dtype=np.float32)

                # 提取 img_metas[0] 的 shift, can_bus, lidar2img
                meta = img_metas[0] if isinstance(img_metas, list) else img_metas

                # 处理 can_bus
                can_bus = meta.get('can_bus', None)
                if isinstance(can_bus, torch.Tensor):
                    can_bus = can_bus.cpu().numpy()
                can_bus = np.asarray(can_bus, dtype=np.float32).flatten()

                # 计算 shift (与 eval 脚本一致)
                delta_x = can_bus[0]
                delta_y = can_bus[1]
                ego_angle = can_bus[-2] / np.pi * 180
                grid_length_y = 60.0 / 200
                grid_length_x = 30.0 / 200
                translation_length = np.sqrt(delta_x**2 + delta_y**2)
                translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
                bev_angle = ego_angle - translation_angle
                shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / 200
                shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / 200
                shift = np.array([[shift_x, shift_y]], dtype=np.float32)

                lidar2img = meta.get('lidar2img', None)
                if isinstance(lidar2img, torch.Tensor):
                    lidar2img = lidar2img.cpu().numpy()
                lidar2img = np.asarray(lidar2img, dtype=np.float32).reshape(6, 4, 4)

                # 保存这一帧的所有数据
                head_data_per_frame[f'frame_{processed:03d}'] = {
                    'mlvl_feats.0': feat0.astype(np.float32),
                    'mlvl_feats.1': feat1.astype(np.float32) if feat1 is not None else None,
                    'mlvl_feats.2': feat2.astype(np.float32) if feat2 is not None else None,
                    'mlvl_feats.3': feat3.astype(np.float32) if feat3 is not None else None,
                    'img_metas.0[shift]': shift,
                    'img_metas.0[can_bus]': can_bus.reshape(1, 18),
                    'img_metas.0[lidar2img]': lidar2img,
                    'prev_bev': prev_bev_np.astype(np.float32),
                }

                # 同时保存 backbone 输入
                if 'img' in captured:
                    img_np = captured['img']
                    # img shape 可能是 [1, 6, 3, H, W] -> reshape to [6, 3, H, W]
                    if img_np.ndim == 5:
                        img_np = img_np.reshape(-1, *img_np.shape[2:])
                    backbone_inputs.append(img_np.astype(np.float32))

            processed += 1
            if processed % 10 == 0:
                print(f"  Captured {processed}/{NUM_FRAMES} frames")

        except Exception as e:
            print(f"  Error at frame {processed}: {e}")
            import traceback
            traceback.print_exc()

    # 还原 extract_feat
    model.extract_feat = _original_extract_feat
    handle2.remove()
    handle3.remove()

    # 保存
    print(f"\nSaving Head calibration data...")
    head_npz_path = os.path.join(OUTPUT_DIR, 'head_calib.npz')
    # 用 dict 保存为 .npz
    save_dict = {}
    for frame_key, frame_data in head_data_per_frame.items():
        for input_name, value in frame_data.items():
            if value is not None:
                # key 格式: frame_000__mlvl_feats.0
                save_dict[f"{frame_key}__{input_name}"] = value
    np.savez(head_npz_path, **save_dict)
    print(f"  Saved {len(head_data_per_frame)} frames to {head_npz_path}")
    print(f"  File size: {os.path.getsize(head_npz_path)/1024/1024:.2f} MB")

    # 保存 backbone 数据
    print(f"\nSaving Backbone calibration data...")
    bb_npz_path = os.path.join(OUTPUT_DIR, 'backbone_calib.npz')
    bb_save_dict = {f'frame_{i:03d}': arr for i, arr in enumerate(backbone_inputs)}
    np.savez(bb_npz_path, **bb_save_dict)
    print(f"  Saved {len(backbone_inputs)} frames to {bb_npz_path}")
    print(f"  File size: {os.path.getsize(bb_npz_path)/1024/1024:.2f} MB")

    # 打印数据统计
    print(f"\n=== 数据统计 (Head 第一帧) ===")
    if len(head_data_per_frame) > 0:
        first = head_data_per_frame['frame_000']
        for k, v in first.items():
            if v is not None:
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, mean={v.mean():.4f}, std={v.std():.4f}, min={v.min():.4f}, max={v.max():.4f}")


if __name__ == '__main__':
    main()
