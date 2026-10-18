"""Build VAD Head INT8 mixed precision TRT engine for AGX Orin (ARM64).

在 Orin 设备上运行，使用 Orin 编译的插件库和共享的 ONNX 模型。

三大修复 (从 x86 版本完整移植):
  1. 真实数据校准 (从 calib_data/head_calib.npz 加载, 懒加载 mmap)
  2. MatMul 显式强制 FP16 (Transformer MHA 不能 INT8)
  3. PREFER_PRECISION_CONSTRAINTS 软约束 (不使用 OBEY, 避免插件兼容性问题)

其他保留:
  - 自定义插件 (MSDeformAttn/SelectAndPad/RotateTRT) 保持 FP16
  - 精度敏感算子 (LayerNorm/Softmax/Sigmoid/Sin/Cos/Reduce) 保持 FP32
"""
import os
import sys
import ctypes

import numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# ======================== TRT API 兼容性辅助 ========================
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


# ======================== 路径配置 (相对路径, 基于 PIPELINE_ROOT) ========================
PIPELINE_ROOT = os.environ.get('VAD_DEPLOY_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ONNX_PATH = os.path.join(PIPELINE_ROOT, "onnx_export/output/vad_head_withprev.onnx")
ENGINE_PATH = os.path.join(PIPELINE_ROOT, "arm64/trt_compile/output/head_int8_orin.engine")
PLUGIN_LIB = os.path.join(PIPELINE_ROOT, "arm64/plugins_build/build/libplugins_orin.so")
# 校准数据共享 x86 版本生成的 head_calib.npz (平台无关, 仅用于统计激活范围)
CALIB_NPZ = os.path.join(PIPELINE_ROOT, "data/calib_data/head_calib.npz")
CALIB_CACHE = os.path.join(PIPELINE_ROOT, "data/calib_data/head_int8_orin.calib")


# ======================== 精度控制关键字 ========================
# FP32 精度敏感算子关键字 (保持 FP32 输出)
FP32_LAYER_KEYWORDS = [
    "layernorm", "layer_norm", "softmax", "sigmoid",
    "sin", "cos", "pow", "sqrt", "erf",
    "reducesum", "reducemean", "reducemax", "reducemin",
    "rotate", "Rotate",
]

# FP16 插件关键字 (保持 FP16, 不做 INT8)
FP16_PLUGIN_KEYWORDS = [
    "MSDeformAttn", "MultiScaleDeformableAttention",
    "SelectAndPad",
    "RotateTRT",
]


# ======================== 真实数据 INT8 校准器 ========================
class HeadRealCalibrator(trt.IInt8EntropyCalibrator2):
    """从 .npz 加载真实校准数据的 Head INT8 校准器。

    数据格式: frame_000__mlvl_feats.0, frame_000__prev_bev, ...
    每帧包含 8 个输入: 4 个 mlvl_feats + 3 个 img_metas + prev_bev
    """

    def __init__(self, calib_npz_path, cache_file, num_frames=64):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = 1
        self.calib_npz_path = calib_npz_path
        self.num_frames = num_frames

        # Lazy loading: 不全部加载，只记录键名（避免 26GB 内存爆炸）
        print(f"[Calibrator] 准备懒加载真实校准数据: {calib_npz_path}")
        self.data = np.load(calib_npz_path, allow_pickle=True, mmap_mode='r')

        # 收集有效帧索引
        self.valid_frame_keys = []
        all_files = set(self.data.files)
        input_names = ['mlvl_feats.0', 'mlvl_feats.1', 'mlvl_feats.2', 'mlvl_feats.3',
                       'img_metas.0[shift]', 'img_metas.0[can_bus]', 'img_metas.0[lidar2img]', 'prev_bev']
        for i in range(num_frames):
            frame_key = f'frame_{i:03d}'
            if all(f'{frame_key}__{n}' in all_files for n in input_names):
                self.valid_frame_keys.append(frame_key)

        self.input_names = input_names
        print(f"[Calibrator] 找到 {len(self.valid_frame_keys)} 帧有效数据 (将使用前 {min(num_frames, len(self.valid_frame_keys))} 帧)")

        # 打印第一帧统计 (验证数据真实性)
        if len(self.valid_frame_keys) > 0:
            first_key = self.valid_frame_keys[0]
            for n in input_names[:4]:  # 只打印 mlvl_feats
                arr = self.data[f'{first_key}__{n}']
                print(f"  {n}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}, range=[{arr.min():.2f}, {arr.max():.2f}]")

        self.current_idx = 0
        self._gpu_refs = []

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.current_idx >= len(self.valid_frame_keys) or self.current_idx >= self.num_frames:
            return None

        frame_key = self.valid_frame_keys[self.current_idx]
        self.current_idx += 1

        if self.current_idx % 10 == 0:
            print(f"  [Calibrator] 进度: {self.current_idx}/{min(self.num_frames, len(self.valid_frame_keys))}")

        # 懒加载: 仅在需要时从 mmap 中读取
        # 用 PyTorch 共享 TRT 的 CUDA 上下文（避免 pycuda 多上下文冲突）
        ptrs = []
        self._gpu_refs = []
        for name in names:
            key = f'{frame_key}__{name}'
            if key in self.data.files:
                arr = np.ascontiguousarray(self.data[key], dtype=np.float32)
            else:
                # 找不到对应的输入名，回退零数据
                print(f"  [Calibrator] WARN: 未找到 {key}，使用零数据")
                arr = np.zeros(1, dtype=np.float32)
            t = torch.from_numpy(arr).float().cuda()
            self._gpu_refs.append(t)
            ptrs.append(int(t.data_ptr()))
        return ptrs

    def read_calibration_cache(self):
        if os.path.isfile(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                cache = f.read()
            print(f"[Calibrator] 读取缓存: {self.cache_file} ({len(cache)} bytes)")
            return cache
        return None

    def write_calibration_cache(self, cache):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"[Calibrator] 写入缓存: {self.cache_file} ({len(cache)} bytes)")


# ======================== 主编译流程 ========================
def main():
    print("=" * 60)
    print("VAD Head INT8 Orin 编译 (三大修复版)")
    print("=" * 60)
    print(f"TRT 版本: {trt.__version__} (major={TRT_MAJOR})")
    print(f"ONNX:     {ONNX_PATH}")
    print(f"Engine:   {ENGINE_PATH}")
    print(f"Calib NPZ: {CALIB_NPZ}")
    print(f"Plugin:   {PLUGIN_LIB}")
    print(f"校准缓存: {CALIB_CACHE}")
    print("=" * 60)
    print()

    # 检查文件
    for f in [ONNX_PATH, PLUGIN_LIB, CALIB_NPZ]:
        if not os.path.exists(f):
            print(f"错误: 文件不存在: {f}")
            sys.exit(1)

    # 删除老的 calib cache (避免污染) — 仅在 VAD_RECALIBRATE=1 时强制重校准。
    # 默认复用已存在的缓存: 与 x86 门控构建共享同一份校准范围 (跨平台可移植),
    # 使 x86 L2 门 (eval_int8_fixed.json 0.7805) 直接代表本构建的校准。
    if os.environ.get("VAD_RECALIBRATE") == "1":
        if os.path.exists(CALIB_CACHE):
            os.remove(CALIB_CACHE)
            print(f"删除老缓存: {CALIB_CACHE}")
    elif os.path.exists(CALIB_CACHE):
        print(f"[Calibrator] 复用已有校准缓存: {CALIB_CACHE} "
              f"(set VAD_RECALIBRATE=1 to recalibrate on-device)")

    # 加载插件
    ctypes.CDLL(PLUGIN_LIB, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    print("插件库加载完成")

    # 创建 builder/network/parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print("解析 ONNX...")
    with open(ONNX_PATH, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  错误 {i}: {parser.get_error(i)}")
            sys.exit(1)
    print(f"网络: {network.num_inputs} 输入, {network.num_outputs} 输出, {network.num_layers} 层")

    # 打印输入/输出
    print("\n输入:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  {inp.name}: shape={tuple(inp.shape)}, dtype={inp.dtype}")
    print("\n输出:")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {out.name}: shape={tuple(out.shape)}, dtype={out.dtype}")

    # 配置 builder
    config = builder.create_builder_config()
    set_workspace(config, 8 * 1024 * 1024 * 1024)

    # INT8 + FP16 混合精度
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    # 关键修复 3: 软约束 (PREFER_PRECISION_CONSTRAINTS)
    # 注意: OBEY_PRECISION_CONSTRAINTS 会因为 MSDeformAttnPlugin 不支持 INT8→FP16 直接转换而失败
    # 使用 PREFER 让 TRT 在精度约束和插件兼容性之间平衡，必要时插入 reformat 层
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # 校准器
    calibrator = HeadRealCalibrator(CALIB_NPZ, CALIB_CACHE)
    config.int8_calibrator = calibrator
    print(f"\nINT8 校准器已设置 (真实数据, {len(calibrator.valid_frame_keys)} 帧)")

    # 关键修复 2: 强制 MatMul 为 FP16
    matmul_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type == trt.LayerType.MATRIX_MULTIPLY:
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                output = layer.get_output(j)
                if output is not None:
                    output.dtype = trt.float16
            matmul_count += 1
    print(f"修复 2: 强制 {matmul_count} 个 MatMul 层为 FP16")

    # FP32 敏感层
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
    print(f"标记 {fp32_count} 个敏感层为 FP32")

    # FP16 插件层
    fp16_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_name = layer.name
        for kw in FP16_PLUGIN_KEYWORDS:
            if kw in layer_name:
                try:
                    layer.precision = trt.float16
                    for j in range(layer.num_outputs):
                        output = layer.get_output(j)
                        if output is not None:
                            output.dtype = trt.float16
                    fp16_count += 1
                except Exception:
                    pass
                break
    print(f"标记 {fp16_count} 个插件层为 FP16")

    # 优化级别
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
    #     config.set_calibration_profile(profile)

    # 编译
    print(f"\n开始编译 INT8 Orin 引擎...")
    serialized = build_engine(builder, network, config)

    print(f"编译成功! 引擎大小: {serialized.nbytes / 1024 / 1024:.2f} MB")
    save_engine(serialized, ENGINE_PATH)

    # 与 FP16 引擎大小对比
    fp16_path = os.path.join(PIPELINE_ROOT, "arm64/trt_compile/output/head_fp16_orin.engine")
    if os.path.exists(fp16_path):
        fp16_size = os.path.getsize(fp16_path) / 1024 / 1024
        int8_size = os.path.getsize(ENGINE_PATH) / 1024 / 1024
        print(f"\n引擎大小对比:")
        print(f"  FP16: {fp16_size:.2f} MB")
        print(f"  INT8: {int8_size:.2f} MB")
        print(f"  压缩比: {int8_size / fp16_size:.2%}")

    print("完成!")


if __name__ == '__main__':
    main()
