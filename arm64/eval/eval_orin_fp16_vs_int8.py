"""Orin 上 FP16 vs INT8 量化精度对比评估 (同平台，不依赖 mmcv)

用法：
  在 Orin 上运行，使用 calib_data/head_calib.npz 的真实输入，
  分别跑 FP16 和 INT8 Head 引擎，对比两者的输出。

精度判定：
  - cos_sim > 0.99: 量化精度极好
  - cos_sim > 0.95: 量化精度可接受
  - cos_sim < 0.95: 量化精度可能有问题
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
    """加载 TRT 引擎并初始化插件。"""
    ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), '')
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print(f"ERROR: Failed to load engine: {path}")
        print(f"  TRT version: {trt.__version__}")
        print(f"  Engine may have been built with a different TRT version.")
        sys.exit(1)
    return engine


def get_io_names(engine):
    """获取引擎的输入和输出名称。"""
    in_names, out_names = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            in_names.append(name)
        else:
            out_names.append(name)
    return in_names, out_names


def trt_run(engine, ctx, stream, inputs_dict):
    """运行 TRT 引擎。返回 {name: torch.Tensor cuda}。"""
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
    """余弦相似度。"""
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def max_abs_diff(a, b):
    """最大绝对误差。"""
    return float((a - b).abs().max())


def relative_error(a, b):
    """相对误差 (L2 norm)。"""
    a = a.flatten().float()
    b = b.flatten().float()
    return float((a - b).norm() / (a.norm() + 1e-8))


def main():
    parser = argparse.ArgumentParser(description='Orin FP16 vs INT8 量化精度对比')
    parser.add_argument('--fp16-engine', required=True, help='FP16 Head 引擎路径')
    parser.add_argument('--int8-engine', required=True, help='INT8 Head 引擎路径')
    parser.add_argument('--plugin-lib', required=True, help='插件库路径')
    parser.add_argument('--calib-npz', required=True, help='真实输入数据 (head_calib.npz)')
    parser.add_argument('--num-frames', type=int, default=10, help='评估帧数')
    parser.add_argument('--output', default='/tmp/eval_orin_fp16_vs_int8.json')
    args = parser.parse_args()

    print(f"=== Orin FP16 vs INT8 量化精度对比 ===")
    print(f"  TRT Python: {trt.__version__}")
    print(f"  FP16 engine: {args.fp16_engine}")
    print(f"  INT8 engine: {args.int8_engine}")
    print(f"  Plugin: {args.plugin_lib}")
    print(f"  Calib NPZ: {args.calib_npz}")
    print(f"  Frames: {args.num_frames}")

    # 加载引擎
    print(f"\nLoading FP16 engine...")
    fp16_engine = load_engine(args.fp16_engine, args.plugin_lib)
    fp16_ctx = fp16_engine.create_execution_context()
    print(f"  FP16 engine loaded: {fp16_engine.num_io_tensors} IO tensors")

    print(f"Loading INT8 engine...")
    int8_engine = load_engine(args.int8_engine, args.plugin_lib)
    int8_ctx = int8_engine.create_execution_context()
    print(f"  INT8 engine loaded: {int8_engine.num_io_tensors} IO tensors")

    stream = torch.cuda.Stream()

    # 加载输入数据
    calib = np.load(args.calib_npz, allow_pickle=True, mmap_mode='r')

    # 获取 IO 名称
    fp16_in, fp16_out = get_io_names(fp16_engine)
    int8_in, int8_out = get_io_names(int8_engine)
    print(f"\n  FP16 inputs: {fp16_in}")
    print(f"  FP16 outputs: {fp16_out}")
    print(f"  INT8 inputs: {int8_in}")
    print(f"  INT8 outputs: {int8_out}")

    # 验证 IO 一致性
    if fp16_in != int8_in:
        print(f"  WARNING: Input names differ between FP16 and INT8!")
    if fp16_out != int8_out:
        print(f"  WARNING: Output names differ between FP16 and INT8!")

    out_names = fp16_out  # 使用 FP16 的输出名称作为参考

    # 收集有效帧
    valid_frames = []
    for i in range(args.num_frames):
        fk = f'frame_{i:03d}'
        if all(f'{fk}__{n}' in calib.files for n in fp16_in):
            valid_frames.append(fk)

    print(f"\nProcessing {len(valid_frames)} valid frames...")

    # 结果统计
    fp16_latencies = []
    int8_latencies = []
    cos_sims_per_output = {n: [] for n in out_names}
    max_diffs_per_output = {n: [] for n in out_names}
    rel_errors_per_output = {n: [] for n in out_names}

    results = {
        'trt_version': trt.__version__,
        'fp16_engine': args.fp16_engine,
        'int8_engine': args.int8_engine,
        'frames': [],
        'summary': {},
    }

    for fk in valid_frames:
        # 加载输入
        inputs = {}
        for name in fp16_in:
            arr = np.ascontiguousarray(calib[f'{fk}__{name}'], dtype=np.float32)
            inputs[name] = torch.from_numpy(arr).float().cuda()

        # 运行 FP16
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fp16_outs = trt_run(fp16_engine, fp16_ctx, stream, inputs)
        torch.cuda.synchronize()
        fp16_lat = (time.perf_counter() - t0) * 1000
        fp16_latencies.append(fp16_lat)

        # 运行 INT8
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        int8_outs = trt_run(int8_engine, int8_ctx, stream, inputs)
        torch.cuda.synchronize()
        int8_lat = (time.perf_counter() - t0) * 1000
        int8_latencies.append(int8_lat)

        # 对比
        frame_metrics = {
            'frame': fk,
            'fp16_latency_ms': fp16_lat,
            'int8_latency_ms': int8_lat,
        }
        for name in out_names:
            if name in fp16_outs and name in int8_outs:
                cs = cos_sim(fp16_outs[name], int8_outs[name])
                md = max_abs_diff(fp16_outs[name], int8_outs[name])
                re = relative_error(fp16_outs[name], int8_outs[name])
                frame_metrics[f'cos_sim_{name}'] = cs
                frame_metrics[f'max_diff_{name}'] = md
                frame_metrics[f'rel_err_{name}'] = re
                cos_sims_per_output[name].append(cs)
                max_diffs_per_output[name].append(md)
                rel_errors_per_output[name].append(re)

        results['frames'].append(frame_metrics)

        # 打印前3个输出的简要信息
        cs_str = ", ".join([
            f"{n[:20]}={cos_sims_per_output[n][-1]:.4f}"
            for n in out_names[:3] if cos_sims_per_output[n]
        ])
        print(f"  {fk}: fp16={fp16_lat:.1f}ms, int8={int8_lat:.1f}ms, {cs_str}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"=== Summary ===")
    print(f"{'='*60}")

    if fp16_latencies:
        avg_fp16 = float(np.mean(fp16_latencies[1:]))  # 跳过第一帧 (warmup)
        avg_int8 = float(np.mean(int8_latencies[1:]))
        speedup = avg_fp16 / avg_int8 if avg_int8 > 0 else 0
        print(f"\nLatency (excluding first frame warmup):")
        print(f"  FP16 avg: {avg_fp16:.1f}ms ({1000/avg_fp16:.2f} FPS)")
        print(f"  INT8 avg: {avg_int8:.1f}ms ({1000/avg_int8:.2f} FPS)")
        print(f"  Speedup:  {speedup:.2f}x")
        results['summary']['fp16_avg_latency_ms'] = avg_fp16
        results['summary']['int8_avg_latency_ms'] = avg_int8
        results['summary']['int8_speedup'] = speedup

    print(f"\nPrecision (INT8 vs FP16):")
    for name in out_names:
        if cos_sims_per_output[name]:
            avg_cs = float(np.mean(cos_sims_per_output[name]))
            avg_md = float(np.mean(max_diffs_per_output[name]))
            avg_re = float(np.mean(rel_errors_per_output[name]))
            min_cs = float(np.min(cos_sims_per_output[name]))
            status = "GOOD" if avg_cs > 0.99 else ("OK" if avg_cs > 0.95 else "CHECK")
            print(f"  [{status}] {name}:")
            print(f"    cos_sim: avg={avg_cs:.6f}, min={min_cs:.6f}")
            print(f"    max_diff: avg={avg_md:.6f}")
            print(f"    rel_err: avg={avg_re:.6f}")
            results['summary'][f'{name}_cos_sim_avg'] = avg_cs
            results['summary'][f'{name}_cos_sim_min'] = min_cs
            results['summary'][f'{name}_rel_err_avg'] = avg_re

    # 整体判定
    all_cos_sims = [cs for sims in cos_sims_per_output.values() for cs in sims]
    if all_cos_sims:
        overall_avg = float(np.mean(all_cos_sims))
        overall_min = float(np.min(all_cos_sims))
        verdict = "PASS" if overall_avg > 0.95 else "REVIEW"
        print(f"\n{'='*60}")
        print(f"Overall Verdict: {verdict}")
        print(f"  Avg cos_sim across all outputs: {overall_avg:.6f}")
        print(f"  Min cos_sim across all outputs: {overall_min:.6f}")
        print(f"{'='*60}")
        results['summary']['overall_cos_sim_avg'] = overall_avg
        results['summary']['overall_cos_sim_min'] = overall_min
        results['summary']['verdict'] = verdict

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
