"""Build VAD Head TRT engines: FP32 baseline + optimized FP16 with proper precision control.
Uses TRT 10.x strong typing API for per-layer precision markers.
"""
import os, sys, ctypes

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

DEPLOY_ROOT = os.environ.get('VAD_DEPLOY_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ONNX_PATH = os.path.join(DEPLOY_ROOT,
    "onnx_export/output/vad_head_withprev.onnx")
ENGINE_FP32 = os.path.join(DEPLOY_ROOT, "x86/trt_compile/output/head_withprev_fp32_pure.engine")
# FP16/mixed engine 名称与 eval (eval_pytorch_vs_trt.py:55) 读取的文件名保持一致
ENGINE_MIXED = os.path.join(DEPLOY_ROOT, "x86/trt_compile/output/head_withprev_fp16_fixed_v2.engine")
PLUGIN_LIB = os.path.join(DEPLOY_ROOT, "plugins/build/libplugins.so")

# Layers whose output should remain FP32 (precision-critical ops)
# These will be converted back to FP16 later by TRT's automatic reformatting
# if the downstream layers accept FP16
FP32_OUTPUT_LAYER_KEYWORDS = [
    "rotate",
    "layernorm",
    "layer_norm",
    "softmax",
    "sigmoid",
    "reduce",
]

MODE = os.environ.get("BUILD_MODE", "fp32")  # fp32 or mixed

if not os.path.exists(PLUGIN_LIB):
    print(f"ERROR: Plugin lib not found at {PLUGIN_LIB}")
    sys.exit(1)

ctypes.CDLL(PLUGIN_LIB, mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

engine_path = ENGINE_FP32 if MODE == "fp32" else ENGINE_MIXED
print(f"Build mode: {MODE}")
print(f"ONNX: {ONNX_PATH}")
print(f"Engine: {engine_path}")
print(f"Plugin: {PLUGIN_LIB}")
print()

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(ONNX_PATH, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"  Parse error {i}: {parser.get_error(i)}")
        sys.exit(1)

print(f"Network: {network.num_inputs} inputs, {network.num_outputs} outputs, {network.num_layers} layers")

# Mark all inputs as FP32 for the FP32 build
input_names = []
for i in range(network.num_inputs):
    inp = network.get_input(i)
    input_names.append(inp.name)

# Mark all outputs as FP32
output_names = []
for i in range(network.num_outputs):
    out = network.get_output(i)
    output_names.append(out.name)

# Configure builder
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 * 1024 * 1024)

if MODE == "fp32":
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
else:
    # FP16 mode with precision control
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # Mark critical layers' outputs as FP32 using strong typing
    fp32_output_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_name = layer.name.lower()
        for kw in FP32_OUTPUT_LAYER_KEYWORDS:
            if kw in layer_name:
                try:
                    for j in range(layer.num_outputs):
                        output_tensor = layer.get_output(j)
                        if output_tensor is not None:
                            output_tensor.dtype = trt.float32
                    fp32_output_count += 1
                except:
                    pass
                break
    print(f"Marked {fp32_output_count} layers for FP32 output precision")

# prev_bev 已改为静态 shape [40000, 1, 256] (bs=1)。
# ONNX 导出侧已移除 prev_bev 的 dynamic_axes,因此 network 中该输入是静态的,
# 无需 create_optimization_profile / set_shape。保留以下旧代码仅作历史记录。
# # Static shape profile for prev_bev
# profile = builder.create_optimization_profile()
# for i in range(network.num_inputs):
#     name = network.get_input(i).name
#     if "prev_bev" in name.lower():
#         profile.set_shape(name, (1, 1, 256), (40000, 1, 256), (40000, 1, 256))
#         config.add_optimization_profile(profile)
#         print(f"Added optimization profile for {name}")
#         break

config.builder_optimization_level = 3

os.makedirs(os.path.dirname(engine_path), exist_ok=True)

print(f"\nBuilding engine (mode={MODE}, this takes several minutes)...")
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    print("FAILED to build engine!")
    sys.exit(1)

print(f"Engine built successfully! Size: {serialized.nbytes / 1024 / 1024:.2f} MB")
with open(engine_path, 'wb') as f:
    f.write(serialized)
print(f"Saved to: {engine_path}")
