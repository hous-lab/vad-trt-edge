"""在 x86 上用 FP16 TRT 引擎跑前 N 帧，输出作为 Orin 评估的 GT 参考。

用法：
  python collect_gt_x86.py --num-frames 10 --output gt_x86_fp16.npz
"""
import os
import sys
import argparse
import ctypes
import time

import numpy as np
import torch
import tensorrt as trt

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_engine(path, plugin_lib):
    ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), '')
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(path, 'rb') as f:
        return runtime.deserialize_cuda_engine(f.read())


def trt_run(engine, ctx, stream, inputs_dict):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-frames', type=int, default=10)
    parser.add_argument('--engine', default=os.path.join(PIPELINE_ROOT,
        "x86/trt_compile/output/head_withprev_fp16_fixed_v2.engine"))
    parser.add_argument('--plugin-lib', default=os.path.join(PIPELINE_ROOT,
        "plugins/build/libplugins.so"))
    parser.add_argument('--calib-npz', default=os.path.join(PIPELINE_ROOT,
        "data/calib_data/head_calib.npz"))
    parser.add_argument('--output', default=os.path.join(PIPELINE_ROOT,
        "arm64/eval/results/gt_x86_fp16.npz"))
    args = parser.parse_args()

    print(f"=== x86 GT Collection (FP16 TRT) ===")
    print(f"  TRT: {trt.__version__}")
    print(f"  Engine: {args.engine}")
    print(f"  Frames: {args.num_frames}")

    head_engine = load_engine(args.engine, args.plugin_lib)
    head_ctx = head_engine.create_execution_context()
    stream = torch.cuda.Stream()

    calib = np.load(args.calib_npz, allow_pickle=True, mmap_mode='r')

    head_in_names = []
    head_out_names = []
    for i in range(head_engine.num_io_tensors):
        name = head_engine.get_tensor_name(i)
        if head_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            head_in_names.append(name)
        else:
            head_out_names.append(name)
    print(f"  Inputs: {head_in_names}")
    print(f"  Outputs: {head_out_names}")

    save_dict = {}
    latencies = []
    for i in range(args.num_frames):
        fk = f'frame_{i:03d}'
        if not all(f'{fk}__{n}' in calib.files for n in head_in_names):
            print(f"  Skipping {fk}: missing inputs")
            continue

        inputs = {}
        for name in head_in_names:
            arr = np.ascontiguousarray(calib[f'{fk}__{name}'], dtype=np.float32)
            inputs[name] = torch.from_numpy(arr).float().cuda()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = trt_run(head_engine, head_ctx, stream, inputs)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        for name, t in outputs.items():
            save_dict[f'{fk}__{name}'] = t.cpu().numpy()
        print(f"  {fk}: latency={lat:.1f}ms, output_shapes={[t.shape for t in outputs.values()][:3]}...")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, **save_dict)
    print(f"\nSaved GT: {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")
    print(f"x86 FP16 latency: {np.mean(latencies):.1f}ms ({1000/np.mean(latencies):.2f} FPS)")


if __name__ == '__main__':
    main()
