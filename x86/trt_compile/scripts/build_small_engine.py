"""VAD-Small INT8 TRT 引擎编译 (PTQ 真实校准, 复用 base 三大修复)

backbone: img [6,3,736,1280] -> 1 级 FPN
head: mlvl_feats.0 + metas + prev_bev -> 9 输出, BEV 100x100
"""
import argparse
import ctypes
import os
import sys

import numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
DEPLOY_ROOT = os.environ.get(
    "VAD_DEPLOY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.environ.get(
    "VAD_ONNX_DIR", os.path.join(DEPLOY_ROOT, "onnx_export/output"))
CALIB_DIR = os.environ.get(
    "VAD_CALIB_DIR", os.path.join(DEPLOY_ROOT, "data/calib_data/calib_data_small"))
ENGINE_DIR = os.environ.get(
    "TRT_OUTPUT_DIR", os.path.join(DEPLOY_ROOT, "x86/trt_compile/output"))

FP32_LAYER_KEYWORDS = [
    "layernorm", "layer_norm", "softmax", "sigmoid",
    "sin", "cos", "pow", "sqrt", "erf",
    "reducesum", "reducemean", "reducemax", "reducemin",
    "rotate", "Rotate",
]
FP16_PLUGIN_KEYWORDS = [
    "MSDeformAttn", "MultiScaleDeformableAttention",
    "SelectAndPad", "RotateTRT",
]


def get_trt_major():
    return int(trt.__version__.split(".")[0])


class BackboneCalib(trt.IInt8EntropyCalibrator2):
    def __init__(self):
        super().__init__()
        self.npz = np.load(os.path.join(CALIB_DIR, "backbone_calib_small.npz"),
                           mmap_mode="r")
        self.keys = list(self.npz.files)
        self.idx = 0
        self._refs = []
        print(f"[Calib] backbone_small: {len(self.keys)} frames")

    def get_batch_size(self):
        return 6

    def get_batch(self, names):
        if self.idx >= len(self.keys):
            return None
        arr = np.ascontiguousarray(self.npz[self.keys[self.idx]],
                                   dtype=np.float32)
        self.idx += 1
        t = torch.from_numpy(arr).float().cuda()
        self._refs = [t]
        return [int(t.data_ptr())]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass


class HeadCalib(trt.IInt8EntropyCalibrator2):
    def __init__(self):
        super().__init__()
        self.npz = np.load(os.path.join(CALIB_DIR, "head_calib_small.npz"),
                           allow_pickle=True, mmap_mode="r")
        self.input_names = ['mlvl_feats.0', 'img_metas.0[shift]',
                            'img_metas.0[can_bus]', 'img_metas.0[lidar2img]',
                            'prev_bev']
        self.valid = []
        for i in range(64):
            k = f'frame_{i:03d}'
            if all(f'{k}__{n}' in self.npz.files for n in self.input_names):
                self.valid.append(k)
        self.idx = 0
        self._refs = []
        print(f"[Calib] head_small: {len(self.valid)} frames")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.valid):
            return None
        k = self.valid[self.idx]
        self.idx += 1
        ptrs, self._refs = [], []
        for name in names:
            key = f'{k}__{name}'
            arr = (np.ascontiguousarray(self.npz[key], dtype=np.float32)
                   if key in self.npz.files
                   else np.zeros(1, dtype=np.float32))
            t = torch.from_numpy(arr).float().cuda()
            self._refs.append(t)
            ptrs.append(int(t.data_ptr()))
        return ptrs

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass


def build(kind, plugin_lib, gpu_id):
    torch.cuda.set_device(gpu_id)
    if kind == "backbone":
        onnx_path = os.path.join(OUTPUT_DIR, "vad_backbone_small.onnx")
        engine_path = os.path.join(ENGINE_DIR, "backbone_int8_small.engine")
        calib = BackboneCalib()
    else:
        onnx_path = os.path.join(OUTPUT_DIR, "vad_head_small.onnx")
        engine_path = os.path.join(ENGINE_DIR, "head_int8_small.engine")
        calib = HeadCalib()

    ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  parse err {i}: {parser.get_error(i)}")
            sys.exit(1)
    print(f"[small] {kind}: {network.num_inputs} in, {network.num_outputs} out, "
          f"{network.num_layers} layers")

    config = builder.create_builder_config()
    import os as _os
    _aux = _os.environ.get("MAX_AUX_STREAMS", "1")
    if _aux:
        config.max_aux_streams = int(_aux)
        print(f"  max_aux_streams = {_aux}")
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 * 1024**3)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.int8_calibrator = calib

    if kind == "head":
        mc = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type == trt.LayerType.MATRIX_MULTIPLY:
                layer.precision = trt.float16
                for j in range(layer.num_outputs):
                    o = layer.get_output(j)
                    if o is not None:
                        o.dtype = trt.float16
                mc += 1
        print(f"  MatMul->FP16: {mc}")
        fc = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for kw in FP32_LAYER_KEYWORDS:
                if kw.lower() in layer.name.lower():
                    try:
                        layer.precision = trt.float32
                        for j in range(layer.num_outputs):
                            o = layer.get_output(j)
                            if o is not None:
                                o.dtype = trt.float32
                        fc += 1
                    except Exception:
                        pass
                    break
        print(f"  FP32 sensitive: {fc}")
        pc = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for kw in FP16_PLUGIN_KEYWORDS:
                if kw in layer.name:
                    try:
                        layer.precision = trt.float16
                        for j in range(layer.num_outputs):
                            o = layer.get_output(j)
                            if o is not None:
                                o.dtype = trt.float16
                        pc += 1
                    except Exception:
                        pass
                    break
        print(f"  FP16 plugin: {pc}")
        # prev_bev 已改为静态 shape [10000, 1, 256] (bs=1, VAD-Small bev 100x100),无需 optimization profile。
        # 以下旧动态 profile 代码保留作历史记录:
        # profile = builder.create_optimization_profile()
        # for i in range(network.num_inputs):
        #     name = network.get_input(i).name
        #     if "prev_bev" in name.lower():
        #         profile.set_shape(name, (1, 1, 256), (10000, 1, 256),
        #                           (10000, 1, 256))
        #         print(f"  prev_bev profile: 1 -> 10000")
        #         break
        # config.add_optimization_profile(profile)

    config.builder_optimization_level = 3
    print(f"[small] compiling {kind} INT8 ...")
    if get_trt_major() >= 10:
        serialized = builder.build_serialized_network(network, config)
    else:
        serialized = builder.build_cuda_engine(network, config).serialize()
    if serialized is None:
        print("编译失败!"); sys.exit(1)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[small] engine saved: {engine_path} "
          f"({serialized.nbytes/1024/1024:.2f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["backbone", "head"], required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--plugin-lib",
                   default=os.path.join(DEPLOY_ROOT,
                                        "plugins/build/libplugins.so"))
    args = p.parse_args()
    build(args.kind, args.plugin_lib, args.gpu)
