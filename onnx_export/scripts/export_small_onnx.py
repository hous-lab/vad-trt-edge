"""VAD-Small ONNX 导出 (backbone 1级FPN + head BEV100 3层)

复用 base 的 ONNX 补丁基础设施 (vad_b2d_onnx/patch), 适配 small:
  - 1 级 FPN (mlvl_feats.0)
  - BEV 100x100, encoder 3 层
  - grid_length = (0.6, 0.3)
head 输入从 small 校准数据 (head_calib_small.npz) 读取。

用法 (x86):
  conda activate p1bevrc
  python export_small_onnx.py [--backbone-only|--head-only]
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

DEPLOY_ROOT = os.environ.get(
    "VAD_DEPLOY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.environ.get(
    "VAD_PROJECT_ROOT",
    os.path.normpath(os.path.join(DEPLOY_ROOT, "..", "..", "Bench2DriveZoo_uniad_vad")))
ONNX_ROOT = os.path.join(PROJECT_ROOT, "vad_b2d_onnx")

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ONNX_ROOT)
sys.path.insert(0, DEPLOY_ROOT)

import numpy as np
import torch
import torch.nn as nn
from mmcv.utils import Config
from mmcv.models import build_model
from mmcv.utils import load_checkpoint

OUTPUT_DIR = os.environ.get("VAD_ONNX_DIR", os.path.join(DEPLOY_ROOT, "onnx_export/output"))
CALIB_SMALL = os.path.join(DEPLOY_ROOT, "data/calib_data/calib_data_small/head_calib_small.npz")
CONFIG_PATH = os.path.join(DEPLOY_ROOT, "configs/VAD_small_e2e_b2d.py")
CKPT_PATH = os.path.join(DEPLOY_ROOT, "ckpts/VAD_small_b2d_fast.pth")


def build_model_small(device="cuda:0"):
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CKPT_PATH, map_location="cpu", strict=False)
    return model.to(device).eval()


# ======================== Backbone ========================
class BackboneWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.img_backbone = model.img_backbone
        self.img_neck = model.img_neck

    def forward(self, img):
        B = img.size(0)
        feats = self.img_backbone(img)
        feats = self.img_neck(feats)
        outs = []
        for f in feats:
            BN, C, H, W = f.size()
            outs.append(f.view(B, BN // B, C, H, W))
        return tuple(outs)


def export_backbone(model):
    wrapper = BackboneWrapper(model).cuda().eval()
    dummy = torch.randn(6, 3, 736, 1280, device="cuda")
    with torch.no_grad():
        _ = wrapper(dummy)
    onnx_path = os.path.join(OUTPUT_DIR, "vad_backbone_small.onnx")
    torch.onnx.export(wrapper, (dummy,), onnx_path,
                      input_names=['img'], output_names=['out.0'],
                      opset_version=15, do_constant_folding=False)
    print(f"[small] backbone ONNX: {onnx_path} "
          f"({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")


# ======================== Head ========================
def patch_head(head):
    from bev_deploy.patch.bevformer.patch import (
        patch_spatial_cross_attn_forward as psca,
        patch_point_sampling as pps,
        patch_bevformer_encoder_forward as pbef)
    from patch.patch_head import (
        patch_VADHead_forward as pvhf,
        patch_VADPerceptionTransformer_get_bev_features as pvpgb,
        patch_VADHead_select_and_pad_query as pvhsq,
        patch_VADHead_select_and_pad_pred_map as pvhsapm)
    for lyr in range(3):  # small: 3 encoder layers
        mod = head.transformer.encoder.layers[lyr].attentions[1]
        head.transformer.encoder.layers[lyr].attentions[1].forward = \
            lambda *a, _m=mod, **kw: psca(_m, *a, **kw)
    head.transformer.encoder.point_sampling = lambda *a, **kw: pps(
        head.transformer.encoder, *a, **kw)
    head.transformer.encoder.forward = lambda *a, **kw: pbef(
        head.transformer.encoder, *a, **kw)
    head.transformer.get_bev_features = lambda *a, **kw: pvpgb(
        head.transformer, *a, **kw)
    head.select_and_pad_query = lambda q, p, c, **kw: pvhsq(head, q, p, c, **kw)
    head.select_and_pad_pred_map = lambda *a, **kw: pvhsapm(head, *a, **kw)
    head.patched_forward = lambda *a, **kw: pvhf(head, *a, **kw)
    return head


class HeadExportWrapper(nn.Module):
    def __init__(self, head, ish):
        super().__init__()
        self.head = head
        self.register_buffer('img_shape_val', ish)

    def forward(self, x0, sft, can, l2i, prev_bev=None):
        r = self.head.patched_forward(
            [x0],
            [{'lidar2img': l2i, 'shift': sft, 'can_bus': can,
              'img_shape': self.img_shape_val}],
            prev_bev=prev_bev, only_bev=False)
        return (r['bev_embed'], r['all_cls_scores'], r['all_bbox_preds'],
                r['all_traj_preds'], r['all_traj_cls_scores'],
                r['map_all_cls_scores'], r['map_all_bbox_preds'],
                r['map_all_pts_preds'], r['ego_fut_preds'])


def export_head(model):
    data = np.load(CALIB_SMALL, allow_pickle=True, mmap_mode='r')

    def load(n, s):
        return torch.from_numpy(
            np.ascontiguousarray(data[f'frame_000__{n}'], dtype=np.float32)
        ).reshape(s).cuda()

    x0 = load('mlvl_feats.0', (1, 6, 256, 23, 40))
    sft = load('img_metas.0[shift]', (1, 2))
    can = load('img_metas.0[can_bus]', (1, 18))
    l2i = load('img_metas.0[lidar2img]', (6, 4, 4))
    ish = torch.tensor([[736, 1280, 3] * 6], dtype=torch.float32).cuda()

    head = patch_head(model.pts_bbox_head)
    print(f"  rotate_prev_bev: {head.transformer.rotate_prev_bev}")
    wr = HeadExportWrapper(head, ish).cuda().eval()

    with torch.no_grad():
        out = wr(x0, sft, can, l2i, None)
        prev_bev = out[0]
        out = wr(x0, sft, can, l2i, prev_bev)
    print(f"[small] head test forward OK, prev_bev: {tuple(prev_bev.shape)}")

    onnx_path = os.path.join(OUTPUT_DIR, "vad_head_small.onnx")
    all_inputs = (x0, sft, can, l2i, prev_bev)
    input_names = ['mlvl_feats.0', 'img_metas.0[shift]', 'img_metas.0[can_bus]',
                   'img_metas.0[lidar2img]', 'prev_bev']
    output_names = ['out.bev_embed', 'out.all_cls_scores', 'out.all_bbox_preds',
                    'out.all_traj_preds', 'out.all_traj_cls_scores',
                    'out.map_all_cls_scores', 'out.map_all_bbox_preds',
                    'out.map_all_pts_preds', 'out.ego_fut_preds']
    # 全静态: prev_bev [10000,1,256] + 所有输出 shape 在 trace 时固定。
    # 不使用 dynamic_axes, TRT 无需 optimization profile, 可选更优 tactic。
    torch.onnx.export(wr, all_inputs, onnx_path,
                      input_names=input_names, output_names=output_names,
                      opset_version=15,
                      do_constant_folding=False)
    print(f"[small] head ONNX: {onnx_path} "
          f"({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-only", action="store_true")
    p.add_argument("--head-only", action="store_true")
    args = p.parse_args()
    model = build_model_small()
    if not args.head_only:
        export_backbone(model)
    if not args.backbone_only:
        export_head(model)
    print("[small] export done")


if __name__ == "__main__":
    main()
