"""
Orin platform evaluation script: TRT Hybrid pipeline on 173 validation samples.
Runs ON the Orin (ARM64). Patches the PyTorch VAD model to use TRT engines for
backbone and head inference, while keeping the rest of the pipeline (data loading,
post-processing, metric computation) in PyTorch.
"""
import os, sys, time, json, copy, ctypes, platform
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# Platform detection (confirm we are on aarch64 / Orin)
print(f"Platform: {platform.machine()}")
if platform.machine() != 'aarch64':
    print("WARNING: Expected aarch64 (Orin). Continuing anyway...")

_torch_lib = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages', 'torch', 'lib')
if os.path.isdir(_torch_lib):
    os.environ['LD_LIBRARY_PATH'] = _torch_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# Path config
PIPELINE_ROOT = os.environ.get("VAD_DEPLOY_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.environ.get("VAD_PROJECT_ROOT", os.path.normpath(os.path.join(PIPELINE_ROOT, "..", "..", "Bench2DriveZoo_uniad_vad")))
DEPLOY_ROOT = os.environ.get("VAD_DEPLOY_ROOT", PIPELINE_ROOT)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TRT version detection
print(f"TensorRT version: {trt.__version__}")

from mmcv.utils import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.utils import load_checkpoint
from torch.utils.data import DataLoader as TorchDataLoader
from mmcv.parallel import collate

CONFIG_PATH = "adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py"
CKPT_PATH = os.path.join(DEPLOY_ROOT, "ckpts/vad_b2d_base.pth")
FEAT_ENGINE = os.environ.get("VAD_BACKBONE_ENGINE",
    os.path.join(DEPLOY_ROOT, "arm64/trt_compile/output/backbone_fp16_orin.engine"))
PLUGINS = [
    os.path.join(DEPLOY_ROOT, "arm64/plugins_build/build/libplugins_orin.so"),
]
for lib in PLUGINS:
    if os.path.exists(lib):
        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    else:
        print(f"WARNING: Plugin lib not found at {lib}")
trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), '')

# Load TRT engines
feat_engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open(FEAT_ENGINE, 'rb').read())
feat_ctx = feat_engine.create_execution_context()
feat_stream = cuda.Stream()

HEAD_ENGINE = os.environ.get("VAD_HEAD_ENGINE",
    os.path.join(DEPLOY_ROOT, "arm64/trt_compile/output/head_fp16_orin.engine"))
head_engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open(HEAD_ENGINE, 'rb').read())
head_ctx = head_engine.create_execution_context()
head_stream = cuda.Stream()

# Head I/O
head_io_names = []
head_in_shapes = []
head_out_shapes = []
for i in range(head_engine.num_io_tensors):
    n = head_engine.get_tensor_name(i)
    m = head_engine.get_tensor_mode(n)
    s = tuple(head_engine.get_tensor_shape(n))
    if m == trt.TensorIOMode.INPUT:
        head_in_shapes.append((n, s))
    else:
        head_out_shapes.append((n, s))
    head_io_names.append(n)

# Get feat I/O
feat_io_names = []
feat_out_shapes = []
for i in range(feat_engine.num_io_tensors):
    n = feat_engine.get_tensor_name(i)
    m = feat_engine.get_tensor_mode(n)
    s = tuple(feat_engine.get_tensor_shape(n))
    if m == trt.TensorIOMode.OUTPUT:
        feat_out_shapes.append(s)
    feat_io_names.append(n)

def trt_feat_infer(inp_tensor):
    out_tensors = []
    for shape in feat_out_shapes:
        t = torch.zeros(shape, dtype=torch.float32, device='cuda:0')
        out_tensors.append(t)
    all_tensors = [inp_tensor] + out_tensors
    for i in range(feat_engine.num_io_tensors):
        feat_ctx.set_tensor_address(feat_io_names[i], int(all_tensors[i].data_ptr()))
    feat_ctx.execute_async_v3(stream_handle=feat_stream.handle)
    feat_stream.synchronize()
    return out_tensors

# Load model
print("Loading model...")
cfg = Config.fromfile(CONFIG_PATH)
cfg.model.train_cfg = None
cfg.model.pretrained = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CKPT_PATH, map_location='cpu')
model = model.cuda().eval()

def trt_infer(engine, ctx, stream, inp_tensors, out_shapes):
    out_tensors = [torch.zeros(s, dtype=torch.float32, device='cuda:0') for _, s in out_shapes]
    all_io = list(inp_tensors) + out_tensors
    assert len(all_io) == engine.num_io_tensors, f"{len(all_io)} vs {engine.num_io_tensors}"
    # prev_bev 已改为静态 shape [40000, 1, 256] (bs=1),engine 中无 -1 维度,
    # 以下 set_input_shape 分支变为 no-op,保留作向后兼容。
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            engine_shape = tuple(engine.get_tensor_shape(name))
            actual_shape = tuple(all_io[i].shape)
            if -1 in engine_shape:
                ctx.set_input_shape(name, actual_shape)
        ctx.set_tensor_address(name, int(all_io[i].data_ptr()))
    ctx.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    return out_tensors

def _feat_trt_wrapper(img, img_metas, len_queue=None):
    B, N, C, H, W = img.shape
    img_flat = img.reshape(B * N, C, H, W).contiguous()
    trt_out = trt_feat_infer(img_flat)
    return [t.transpose(0, 1).contiguous() for t in trt_out]

def _head_trt_wrapper(mlvl_feats, img_metas, prev_bev=None, only_bev=False,
                       ego_his_trajs=None, ego_lcf_feat=None):
    dev = 'cuda:0'
    meta = img_metas[0]
    
    def to_cuda_f32(arr, shape):
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr.copy()).float().reshape(shape).cuda()
        elif isinstance(arr, torch.Tensor):
            return arr.clone().float().reshape(shape).cuda()
        elif isinstance(arr, list):
            arr = np.array(arr, dtype=np.float32)
            return torch.from_numpy(arr).float().reshape(shape).cuda()
        else:
            return torch.tensor(arr, dtype=torch.float32).reshape(shape).cuda()
    
    can_arr = meta['can_bus']
    if isinstance(can_arr, torch.Tensor):
        can_arr = can_arr.cpu().numpy()
    can_arr = np.asarray(can_arr, dtype=np.float32).flatten()

    # CRITICAL: shift computation from can_bus (matching VAD model logic)
    delta_x = can_arr[0]
    delta_y = can_arr[1]
    ego_angle = can_arr[-2] / np.pi * 180  # rad -> deg
    grid_length_y = 60.0 / 200  # real_h / bev_h
    grid_length_x = 30.0 / 200  # real_w / bev_w
    translation_length = np.sqrt(delta_x**2 + delta_y**2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    bev_angle = ego_angle - translation_angle
    shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / 200
    shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / 200
    shift = torch.tensor([[shift_x, shift_y]], dtype=torch.float32, device='cuda:0')

    can_bus = to_cuda_f32(can_arr, (1, 18))
    lidar2img = to_cuda_f32(meta['lidar2img'], (6, 4, 4))
    
    inp_tensors = [f.contiguous() for f in mlvl_feats] + [shift, can_bus, lidar2img]
    if prev_bev is not None:
        inp_tensors.append(prev_bev.contiguous())
    else:
        prev_shape = head_in_shapes[-1][1]
        if -1 in prev_shape:
            prev_bev_zeros = torch.zeros(40000, 1, 256, dtype=torch.float32, device='cuda:0').contiguous()
        else:
            prev_bev_zeros = torch.zeros(*prev_shape, dtype=torch.float32, device='cuda:0').contiguous()
        inp_tensors.append(prev_bev_zeros)
    
    out_tensors = trt_infer(head_engine, head_ctx, head_stream, inp_tensors, head_out_shapes)
    
    out_map = {}
    for (n, _), t in zip(head_out_shapes, out_tensors):
        out_map[n] = t

    name_map = {
        'out.bev_embed': 'bev_embed', 'out.all_cls_scores': 'all_cls_scores',
        'out.all_bbox_preds': 'all_bbox_preds', 'out.all_traj_preds': 'all_traj_preds',
        'out.all_traj_cls_scores': 'all_traj_cls_scores',
        'out.map_all_cls_scores': 'map_all_cls_scores',
        'out.map_all_bbox_preds': 'map_all_bbox_preds',
        'out.map_all_pts_preds': 'map_all_pts_preds',
        'out.ego_fut_preds': 'ego_fut_preds',
    }
    result = {}
    for trt_full, py_key in name_map.items():
        if trt_full in out_map:
            result[py_key] = out_map[trt_full]
    result.update({
        'enc_cls_scores': None, 'enc_bbox_preds': None,
        'map_enc_cls_scores': None, 'map_enc_bbox_preds': None, 'map_enc_pts_preds': None,
    })
    return result

# Patch the model
model.extract_img_feat = lambda img, img_metas, len_queue=None: _feat_trt_wrapper(img, img_metas, len_queue)
model.pts_bbox_head.forward = lambda mlvl_feats, img_metas, prev_bev=None, only_bev=False, ego_his_trajs=None, ego_lcf_feat=None: \
    _head_trt_wrapper(mlvl_feats, img_metas, prev_bev, only_bev, ego_his_trajs, ego_lcf_feat)
model.video_test_mode = True

# Build dataset
dataset = build_dataset(cfg.data.test)
data_loader = TorchDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

# Run evaluation
print(f"Running TRT Full Pipeline evaluation...")
results_trt = []
latencies = []
processed = 0

for batch_data in data_loader:
    if processed >= 173:
        break
    try:
        cuda_batch = {}
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                cuda_batch[k] = v.cuda()
            elif isinstance(v, list):
                cuda_batch[k] = [e.cuda() if isinstance(e, torch.Tensor) else e for e in v]

        with torch.no_grad():
            t0 = time.perf_counter()
            result = model(cuda_batch, return_loss=False, rescale=True)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        while isinstance(result, (list, tuple)):
            result = result[0]

        if isinstance(result, dict) and 'pts_bbox' in result:
            cpu_res = {}
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    cpu_res[k] = v.cpu()
                elif isinstance(v, dict):
                    cpu_d = {}
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            cpu_d[kk] = vv.cpu()
                        else:
                            cpu_d[kk] = vv
                    cpu_res[k] = cpu_d
                else:
                    cpu_res[k] = v
            if 'boxes_3d' in result and hasattr(result['boxes_3d'], 'tensor'):
                cpu_res['boxes_3d'] = result['boxes_3d']
                cpu_res['boxes_3d'].tensor = result['boxes_3d'].tensor.cpu()
            results_trt.append(cpu_res)
            processed += 1

    except Exception as e:
        print(f"  ERROR sample {processed}: {e}")
        import traceback; traceback.print_exc()

    if processed % 10 == 0:
        avg = np.mean(latencies[-10:]) if latencies else 0
        print(f"  {processed}/173 samples, avg: {avg:.1f}ms")

# Compute metrics
avg_ms = np.mean(latencies) if latencies else 0
print(f"\n=== TRT Full Pipeline ===")
print(f"Latency: {avg_ms:.1f}ms ({1000/avg_ms:.2f}FPS)")

plan_keys = ['plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s', 'plan_obj_col_1s', 'plan_obj_col_3s']
metric_acc = {}
num_valid = 0
for r in results_trt:
    mr = r.get('metric_results', {})
    if not mr.get('fut_valid_flag', False):
        continue
    num_valid += 1
    for k in plan_keys:
        v = mr.get(k, 0)
        if isinstance(v, torch.Tensor): v = v.item()
        metric_acc[k] = metric_acc.get(k, 0) + v

if num_valid > 0:
    for k in plan_keys:
        mv = metric_acc.get(k, 0) / num_valid
        print(f"  {k}: {mv:.4f}")
    print(f"  Valid samples: {num_valid}/{processed}")

save_path = os.environ.get("EVAL_RESULT_PATH", "/tmp/trt_orin_results.json")
with open(save_path, 'w') as f:
    json.dump({
        'latency_ms': avg_ms, 'fps': 1000/avg_ms if avg_ms > 0 else 0,
        'num_samples': processed, 'num_valid': num_valid,
        'planning': {k: float(metric_acc.get(k, 0) / num_valid) for k in plan_keys},
    }, f, indent=2)
print(f"\nSaved to {save_path}")
