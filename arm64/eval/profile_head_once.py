#!/usr/bin/env python3
"""Run ONE marked head-engine inference for Nsight Compute (ncu) profiling.

Loads the engine, one warm-up run (unprofiled), then torch.cuda.profiler.start()
around a single inference so ncu --profile-from-start off profiles exactly it.
"""
import argparse, os, sys
import numpy as np
import torch
import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_orin_lite import load_engine, trt_run

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--plugin-lib", required=True)
ap.add_argument("--random-inputs", action="store_true")
args = ap.parse_args()

engine = load_engine(args.engine, args.plugin_lib)
ctx = engine.create_execution_context()
stream = torch.cuda.Stream()

inputs = {}
for i in range(engine.num_io_tensors):
    n = engine.get_tensor_name(i)
    if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
        shape = tuple(d if d != -1 else 1 for d in engine.get_tensor_shape(n))
        inputs[n] = torch.randn(shape, dtype=torch.float32, device="cuda:0")

trt_run(engine, ctx, stream, inputs)  # warm-up (unprofiled)
torch.cuda.synchronize()
torch.cuda.profiler.start()
trt_run(engine, ctx, stream, inputs)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("profiled run done")
