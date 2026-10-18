"""Build VAD Backbone FP16 TRT engine for AGX Orin (ARM64).

Runs on the Orin device, using the Orin-built plugin library and the shared ONNX model.
FP16 only (no INT8 calibration); suited to quick validation and precision-sensitive cases.

Backbone input: img [6, 3, 736, 1280] (6 cameras, 3 channels, 736x1280)
Backbone output: 4 multi-scale feature maps (FPN)
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
ONNX_PATH = os.path.join(PIPELINE_ROOT, "onnx_export/output/vad_backbone.onnx")
ENGINE_PATH = os.path.join(PIPELINE_ROOT, "arm64/trt_compile/output/backbone_fp16_orin.engine")
PLUGIN_LIB = os.path.join(PIPELINE_ROOT, "arm64/plugins_build/build/libplugins_orin.so")


# ======================== Main build flow ========================
def main():
    print("=" * 60)
    print("VAD Backbone FP16 Orin build")
    print("=" * 60)
    print(f"TRT version: {trt.__version__} (major={TRT_MAJOR})")
    print(f"ONNX:     {ONNX_PATH}")
    print(f"Engine:   {ENGINE_PATH}")
    print(f"Plugin:   {PLUGIN_LIB}")
    print(f"Mode:     FP16 only (no INT8 calibration)")
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

    # Configure builder
    config = builder.create_builder_config()
    set_workspace(config, 8 * 1024 * 1024 * 1024)  # 8GB

    # FP16 mode
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # Optimization level
    config.builder_optimization_level = 3

    # Build engine
    print(f"\nBuilding FP16 engine (this may take a while)...")
    serialized = build_engine(builder, network, config)

    # Save engine
    print(f"Build OK! Engine size: {serialized.nbytes / 1024 / 1024:.2f} MB")
    save_engine(serialized, ENGINE_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
