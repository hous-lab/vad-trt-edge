"""轻量级 Orin 部署精度评估 (不依赖 mmcv)

用法：
  在 Orin 上运行此脚本，使用 calib_data 中的真实输入数据，跑 TRT 引擎，
  与提供的 GT 输出（来自 x86 端 PyTorch FP32）做 cosine similarity 对比。

精度判定：
  - cos_sim > 0.99: 精度高度对齐 (与 x86 等价)
  - cos_sim > 0.95: 精度可接受
  - cos_sim < 0.95: 精度可能有问题

输入：
  - --backbone-engine: TRT backbone 引擎路径
  - --head-engine: TRT head 引擎路径
  - --gt-npz: GT 输出 NPZ 文件路径 (从 x86 端提供，可选)
  - --calib-npz: 真实输入数据 (head_calib.npz)
  - --num-frames: 评估帧数 (默认 10)
"""
import os
import sys
import argparse
import ctypes
import json
import time

import numpy as np
import torch
import tensorrt as trt


def load_engine(path, plugin_lib):
    ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), '')
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(path, 'rb') as f:
        return runtime.deserialize_cuda_engine(f.read())


def trt_run(engine, ctx, stream, inputs_dict):
    """Run TRT engine. inputs_dict: {name: torch.Tensor cuda}.
    Returns: {name: torch.Tensor cuda} for all outputs.
    """
    outputs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            t = inputs_dict[name]
            engine_shape = tuple(engine.get_tensor_shape(name))
            actual_shape = tuple(t.shape)
            if -1 in engine_shape:
                ctx.set_input_shape(name, actual_shape)
            ctx.set_tensor_address(name, int(t.data_ptr()))
        else:
            shape = tuple(ctx.get_tensor_shape(name))
            t = torch.zeros(shape, dtype=torch.float32, device='cuda:0').contiguous()
            ctx.set_tensor_address(name, int(t.data_ptr()))
            outputs[name] = t
    ctx.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    return outputs


def cos_sim(a, b):
    """Cosine similarity for arbitrary shape tensors."""
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def max_abs_diff(a, b):
    return float((a - b).abs().max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone-engine', required=True)
    parser.add_argument('--head-engine', required=True)
    parser.add_argument('--plugin-lib', required=True)
    parser.add_argument('--calib-npz', required=True, help='Real input data')
    parser.add_argument('--gt-npz', default=None, help='Ground truth outputs from x86 PyTorch (optional)')
    parser.add_argument('--num-frames', type=int, default=10)
    parser.add_argument('--output', default='/tmp/eval_orin_lite.json')
    args = parser.parse_args()

    print(f"=== Orin Lightweight Eval ===")
    print(f"  TRT: {trt.__version__}")
    print(f"  Backbone engine: {args.backbone_engine}")
    print(f"  Head engine: {args.head_engine}")
    print(f"  Calib NPZ: {args.calib_npz}")
    print(f"  GT NPZ: {args.gt_npz if args.gt_npz else 'None (only latency)'}")
    print(f"  Frames: {args.num_frames}")

    # Load engines
    head_engine = load_engine(args.head_engine, args.plugin_lib)
    head_ctx = head_engine.create_execution_context()
    stream = torch.cuda.Stream()

    # Load real input data
    calib = np.load(args.calib_npz, allow_pickle=True, mmap_mode='r')
    gt = None
    if args.gt_npz and os.path.exists(args.gt_npz):
        gt = np.load(args.gt_npz, allow_pickle=True)

    # Get head IO names
    head_in_names = []
    head_out_names = []
    for i in range(head_engine.num_io_tensors):
        name = head_engine.get_tensor_name(i)
        if head_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            head_in_names.append(name)
        else:
            head_out_names.append(name)
    print(f"  Head inputs: {head_in_names}")
    print(f"  Head outputs: {head_out_names}")

    results = {
        'trt_version': trt.__version__,
        'frames': [],
        'avg_latency_ms': 0.0,
        'avg_cos_sim': {},
    }

    latencies = []
    cos_sims_per_output = {n: [] for n in head_out_names}

    # Iterate frames
    valid_frames = []
    for i in range(args.num_frames):
        fk = f'frame_{i:03d}'
        if all(f'{fk}__{n}' in calib.files for n in head_in_names):
            valid_frames.append(fk)

    print(f"\nProcessing {len(valid_frames)} frames...")
    for fk in valid_frames:
        # Load inputs
        inputs = {}
        for name in head_in_names:
            arr = np.ascontiguousarray(calib[f'{fk}__{name}'], dtype=np.float32)
            inputs[name] = torch.from_numpy(arr).float().cuda()

        # Warmup + run
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = trt_run(head_engine, head_ctx, stream, inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000
        latencies.append(lat_ms)

        # Compare with GT if available
        frame_metrics = {'frame': fk, 'latency_ms': lat_ms}
        if gt is not None:
            for name in head_out_names:
                gt_key = f'{fk}__{name}'
                if gt_key in gt.files:
                    gt_arr = torch.from_numpy(np.ascontiguousarray(gt[gt_key])).float().cuda()
                    cs = cos_sim(outputs[name], gt_arr)
                    frame_metrics[f'cos_sim_{name}'] = cs
                    cos_sims_per_output[name].append(cs)
        results['frames'].append(frame_metrics)
        print(f"  {fk}: lat={lat_ms:.1f}ms" + 
              (", cos_sims=" + ", ".join([f"{n}={cs:.4f}" for n,cs in [(name, cos_sim(outputs[name], 
               torch.from_numpy(np.ascontiguousarray(gt[f'{fk}__{name}'])).float().cuda()))
               for name in head_out_names if gt is not None and f'{fk}__{name}' in gt.files][:3]])
               if gt is not None else ""))

    if latencies:
        results['avg_latency_ms'] = float(np.mean(latencies))
        results['fps'] = 1000.0 / results['avg_latency_ms']
    if gt is not None:
        for name, sims in cos_sims_per_output.items():
            if sims:
                results['avg_cos_sim'][name] = float(np.mean(sims))

    print(f"\n=== Summary ===")
    print(f"Avg latency: {results['avg_latency_ms']:.1f}ms ({results.get('fps', 0):.2f} FPS)")
    if results['avg_cos_sim']:
        for name, cs in results['avg_cos_sim'].items():
            print(f"  cos_sim {name}: {cs:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
