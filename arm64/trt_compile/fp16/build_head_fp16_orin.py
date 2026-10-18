"""Build VAD Head FP16 TRT engine for AGX Orin (ARM64).

Runs on the Orin device, using the Orin-built plugin library and the shared ONNX model.
FP16 mode; precision-sensitive layers (LayerNorm/Softmax/Sin/Cos, etc.) stay FP32.
prev_bev input uses static shape [40000, 1, 256] (bs=1; since 2026-08-11 no optimization profile).

Precision policy:
  - global FP16
  - precision-sensitive ops (LayerNorm/Softmax/Sigmoid/Sin/Cos/Reduce, etc.) stay FP32
  - PREFER_PRECISION_CONSTRAINTS soft constraint (same as the x86 INT8 recipe)
"""
import os
import sys
import ctypes

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# ======================== TRT API compatibility helpers ========================
def get_trt_version():
    """Get TRT major version number."""
    ver_str = trt.__version__
    return int(ver_str.split('.')[0])

TRT_MAJOR = get_trt_version()

def build_engine(builder, network, config):
    """Build engine with TRT version compatibility."""
    if TRT_MAJOR >= 10:
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            print("FAILED to build engine!")
            sys.exit(1)
        return serialized
    else:
        # TRT 8.x
        engine = builder.build_cuda_engine(network, config)
        if engine is None:
            print("FAILED to build engine!")
            sys.exit(1)
        # Serialize for saving
        return engine.serialize()

def set_workspace(config, size_bytes):
    """Set workspace with TRT version compatibility."""
    if TRT_MAJOR >= 10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size_bytes)
    else:
        config.max_workspace_size = size_bytes

def save_engine(serialized, path):
    """Save serialized engine to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(serialized)
    print(f"Engine saved: {path} ({serialized.nbytes / 1024 / 1024:.2f} MB)")


# ======================== Path config (relative to PIPELINE_ROOT) ========================
PIPELINE_ROOT = os.environ.get('VAD_DEPLOY_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ONNX_PATH = os.path.join(PIPELINE_ROOT, "onnx_export/output/vad_head_withprev.onnx")
ENGINE_PATH = os.path.join(PIPELINE_ROOT, "arm64/trt_compile/output/head_fp16_orin.engine")
PLUGIN_LIB = os.path.join(PIPELINE_ROOT, "arm64/plugins_build/build/libplugins_orin.so")


# ======================== Precision-control keywords ========================
# FP32 precision-sensitive op keywords (keep FP32 output)
FP32_LAYER_KEYWORDS = [
    "layernorm", "layer_norm", "softmax", "sigmoid",
    "sin", "cos", "pow", "sqrt", "erf",
    "reducesum", "reducemean", "reducemax", "reducemin",
    "rotate", "Rotate",
]


# ======================== Main build flow ========================
def main():
    print("=" * 60)
    print("VAD Head FP16 Orin build")
    print("=" * 60)
    print(f"TRT version: {trt.__version__} (major={TRT_MAJOR})")
    print(f"ONNX:     {ONNX_PATH}")
    print(f"Engine:   {ENGINE_PATH}")
    print(f"Plugin:   {PLUGIN_LIB}")
    print(f"Mode:     FP16 + FP32 precision-sensitive layers")
    print("=" * 60)
    print()

    # Check files
    if not os.path.exists(ONNX_PATH):
        print(f"ERROR: ONNX not found: {ONNX_PATH}")
        sys.exit(1)
    if not os.path.exists(PLUGIN_LIB):
        print(f"ERROR: plugin lib not found: {PLUGIN_LIB}")
        sys.exit(1)

    # Load plugin library
    ctypes.CDLL(PLUGIN_LIB, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    print("Plugin library loaded")

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Parsing ONNX: {ONNX_PATH}")
    with open(ONNX_PATH, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  parse error {i}: {parser.get_error(i)}")
            sys.exit(1)

    print(f"Network: {network.num_inputs} inputs, {network.num_outputs} outputs, "
          f"{network.num_layers} layers")

    # Print input/output info
    print("\nInputs:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  {inp.name}: shape={tuple(inp.shape)}, dtype={inp.dtype}")
    print("\nOutputs:")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {out.name}: shape={tuple(out.shape)}, dtype={out.dtype}")

    # FP32 敏感层精度控制
    fp32_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_name = layer.name.lower()
        for kw in FP32_LAYER_KEYWORDS:
            if kw.lower() in layer_name:
                try:
                    layer.precision = trt.float32
                    for j in range(layer.num_outputs):
                        output = layer.get_output(j)
                        if output is not None:
                            output.dtype = trt.float32
                    fp32_count += 1
                except Exception:
                    pass
                break
    print(f"\n标记 {fp32_count} 个敏感层为 FP32 (关键字: {FP32_LAYER_KEYWORDS})")

    # Configure builder
    config = builder.create_builder_config()
    set_workspace(config, 8 * 1024 * 1024 * 1024)  # 8GB

    # FP16 mode
    config.set_flag(trt.BuilderFlag.FP16)
    # 使用 PREFER_PRECISION_CONSTRAINTS 软约束 (与 x86 INT8 方案一致, 避免插件兼容性问题)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # Optimization level
    config.builder_optimization_level = 3

    # prev_bev 已改为静态 shape [40000, 1, 256] (bs=1),无需 optimization profile。
    # 以下旧动态 profile 代码保留作历史记录:
    # # prev_bev 动态形状 profile
    # profile = builder.create_optimization_profile()
    # profile_set = False
    # for i in range(network.num_inputs):
    #     name = network.get_input(i).name
    #     if "prev_bev" in name.lower():
    #         profile.set_shape(name, (1, 1, 256), (40000, 1, 256), (40000, 1, 256))
    #         print(f"动态形状 profile: {name}")
    #         profile_set = True
    #         break
    # if profile_set:
    #     config.add_optimization_profile(profile)

    # Build engine
    print(f"\nBuilding FP16 engine (this may take a while)...")
    serialized = build_engine(builder, network, config)

    # Save engine
    print(f"Build OK! Engine size: {serialized.nbytes / 1024 / 1024:.2f} MB")
    save_engine(serialized, ENGINE_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
