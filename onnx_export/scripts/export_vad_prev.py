"""VAD-B2D demo_data 捕获: 生成 head 导出所需的真实输入/输出张量 (Phase 1).

流程 (函数化分阶段, 与 hous_reproduce/07 的封装风格一致):
  1. build_model_and_loader()   构建完整 VAD 模型与测试数据集
  2. capture_head_inputs()      完整前向至第 capture_idx 帧 (模型内部时序状态
                                已经过前序帧累积, prev_bev 为真实时序输出),
                                用 pre-hook 抓取 pts_bbox_head 的实参
  3. transform_meta()           meta 值张量化: fn_lidar2img + fn_canbus
                                (证据#1/#3/#4: shift 预计算, lidar2img/can_bus 升级张量)
  4. save_demo_data()           存为 .bin + export_info.json

产物: $ONNX_ROOT/demo_data/b2d_vadv1_prev.pts_bbox_head.forward/
  (ONNX_ROOT = Bench2DriveZoo_uniad_vad/vad_b2d_onnx, 与 export_head_model.load_data 对应)
"""
import os
import sys
import json
import copy
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_ROOT = os.environ.get('VAD_DEPLOY_ROOT', os.environ.get(
    'DEPLOY_ROOT', os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))))
ONNX_ROOT = os.path.normpath(os.path.join(
    DEPLOY_ROOT, '..', '..', 'Bench2DriveZoo_uniad_vad', 'vad_b2d_onnx'))
PROJECT_ROOT = os.path.normpath(os.path.join(
    DEPLOY_ROOT, '..', '..', 'Bench2DriveZoo_uniad_vad'))

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ONNX_ROOT)

BEV_H, BEV_W = 200, 200
GRID_LENGTH = [0.3, 0.15]
PREFIX = "b2d_vadv1_prev.pts_bbox_head.forward"
CAPTURE_IDX = 4          # 跳过前 4 帧, 在第 4 帧捕获 (时序状态已预热)

OUT_KEYS = ['bev_embed', 'all_cls_scores', 'all_bbox_preds', 'all_traj_preds',
            'all_traj_cls_scores', 'map_all_cls_scores', 'map_all_bbox_preds',
            'map_all_pts_preds', 'ego_fut_preds']


def build_model_and_loader():
    from torch.utils.data import DataLoader as TorchDataLoader
    from mmcv.parallel import collate
    from mmcv.utils import Config, load_checkpoint
    from mmcv.datasets import build_dataset
    from mmcv.models import build_model

    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile("adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py")
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, 'ckpts/vad_b2d_base.pth', map_location='cpu')
    model = model.cuda().eval()
    model.use_grid_mask = False

    dataset = build_dataset(cfg.data.test)
    loader = TorchDataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=collate)
    return model, loader


def capture_head_inputs(model, loader):
    """完整前向至 CAPTURE_IDX 帧, 抓取 head 实参与真实输出.
    返回 dict: mlvl(list4) / img_metas(list[dict]) / prev_bev(Tensor) / outs(dict9)"""
    cap_in, cap_out = {}, {}

    def pre_hook(module, args, kwargs):
        cap_in['mlvl'] = args[0]
        cap_in['img_metas'] = args[1]
        cap_in['prev_bev'] = kwargs.get('prev_bev', None)

    def post_hook(module, args, kwargs, out):
        cap_out['outs'] = out

    h1 = model.pts_bbox_head.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = model.pts_bbox_head.register_forward_hook(post_hook, with_kwargs=True)

    # 旁路 eval 指标计算 (gt_attr_labels 置空), 捕获阶段不需要
    orig_stp = model.simple_test_pts
    model.simple_test_pts = lambda *a, **kw: (
        orig_stp(*a, **{**kw, 'gt_attr_labels': None})
        if 'gt_attr_labels' in kw else orig_stp(*a, **kw))

    result = None
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx < CAPTURE_IDX:
                continue
            print(f"在帧 {batch_idx} 捕获...")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()
                elif isinstance(v, list):
                    data[k] = [e.cuda() if isinstance(e, torch.Tensor) else e for e in v]
            model(data, return_loss=False, rescale=True)
            # 注: 跳帧 continue 不产生前向, 此帧即模型首帧, prev_bev 为 None
            # (head 导出阶段用两遍法生成真实 prev_bev, 见 export_head_script.py)
            result = dict(
                mlvl=[t.detach() for t in cap_in['mlvl']],
                img_metas=copy.deepcopy(cap_in['img_metas']),
                prev_bev=(cap_in['prev_bev'].detach()
                          if cap_in['prev_bev'] is not None else None),
                outs={k: v.detach().clone() for k, v in cap_out['outs'].items()
                      if isinstance(v, torch.Tensor)},
            )
            break
    model.simple_test_pts = orig_stp
    h1.remove()
    h2.remove()
    return result


def transform_meta(cap):
    """meta 值张量化 (与 03_export_head 的输入变换一致)"""
    from bev_deploy.patch.bevformer.patch import fn_lidar2img, fn_canbus
    meta_list = fn_lidar2img(cap['img_metas'])
    meta_list = fn_canbus(cap['mlvl'][0], meta_list, BEV_H, BEV_W, GRID_LENGTH)
    return meta_list


def save_demo_data(cap, meta_list):
    out_dir = os.path.join(DEPLOY_ROOT, 'onnx_export/demo_data', PREFIX)
    os.makedirs(out_dir, exist_ok=True)
    meta = meta_list[0]

    for i, t in enumerate(cap['mlvl']):
        t.cpu().numpy().tofile(f'{out_dir}/mlvl_feats.{i}.bin')
    meta['shift'].cpu().numpy().tofile(f'{out_dir}/img_metas.0[shift].bin')
    np.asarray(meta['can_bus'].cpu().numpy(), dtype=np.float32).tofile(
        f'{out_dir}/img_metas.0[can_bus].bin')
    np.asarray(meta['lidar2img'].cpu().numpy(), dtype=np.float32).tofile(
        f'{out_dir}/img_metas.0[lidar2img].bin')
    np.asarray(meta['img_shape'], dtype=np.float32).tofile(
        f'{out_dir}/img_metas.0[img_shape].bin')
    if cap['prev_bev'] is not None:
        cap['prev_bev'].cpu().numpy().tofile(f'{out_dir}/prev_bev.bin')
    for name in OUT_KEYS:
        v = cap['outs'].get(name)
        if isinstance(v, torch.Tensor):
            v.cpu().numpy().tofile(f'{out_dir}/out.{name}.bin')

    info = {
        'input_names': ['mlvl_feats.0', 'mlvl_feats.1', 'mlvl_feats.2', 'mlvl_feats.3',
                        'img_metas.0[shift]', 'img_metas.0[can_bus]',
                        'img_metas.0[lidar2img]'],
        'output_names': [f'out.{n}' for n in OUT_KEYS],
        'session': PREFIX,
        'dir': out_dir,
    }
    with open(f'{out_dir}/export_info.json', 'w') as fp:
        json.dump(info, fp, indent=2)
    print(f"demo_data saved: {out_dir}")


def main():
    print("Loading PyTorch FP32 VAD model...")
    model, loader = build_model_and_loader()
    cap = capture_head_inputs(model, loader)
    print("  prev_bev:", tuple(cap['prev_bev'].shape) if cap['prev_bev'] is not None else "None (首帧语义)")
    meta_list = transform_meta(cap)
    save_demo_data(cap, meta_list)


if __name__ == '__main__':
    main()
