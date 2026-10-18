"""Build VAD Head INT8 Mixed-Precision TRT 引擎编译脚本。

在 build_head_v2.py 的 FP16 混合精度基础上，进一步加入 INT8 量化：
  - INT8 + FP16 混合精度全局策略
  - 精度关键层（LayerNorm/Softmax/Sigmoid/Reduce/Rotate 等）保持 FP32 输出
  - MSDeformAttn / SelectAndPad / RotateTRT 插件层保持 FP16（避免插件量化问题）
  - 其余层使用 INT8 量化以获得更好的推理性能

校准模式（通过环境变量 BUILD_MODE 控制）：
  - int8_random: 使用随机数据校准（快速验证）
  - int8_real:   使用真实数据校准（从 pkl/npy 文件加载，最终部署）

Head 输入:
  - mlvl_feats.0: [1, 6, 256, 92, 160]
  - mlvl_feats.1: [1, 6, 256, 46, 80]
  - mlvl_feats.2: [1, 6, 256, 23, 40]
  - mlvl_feats.3: [1, 6, 256, 12, 20]
  - img_metas.0[shift]: [1, 2]
  - img_metas.0[can_bus]: [1, 18]
  - img_metas.0[lidar2img]: [6, 4, 4]
  - prev_bev: [num_bev, 1, 256] (动态形状)
"""
import os
import sys
import ctypes
import glob

import numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# ======================== 路径配置 ========================
PIPELINE_ROOT = os.environ.get('VAD_DEPLOY_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ONNX_PATH = os.path.join(PIPELINE_ROOT, "onnx_export/output/vad_head_withprev.onnx")
ENGINE_PATH = os.path.join(PIPELINE_ROOT, "x86/trt_compile/output/head_withprev_int8_mixed.engine")
PLUGIN_LIB = os.path.join(PIPELINE_ROOT, "plugins/build/libplugins.so")
CALIB_CACHE = os.path.join(PIPELINE_ROOT, "data/calib_data/head_int8_mixed.calib")
EVAL_DATA_DIR = os.path.join(PIPELINE_ROOT, "data")

# ======================== 精度控制关键字 ========================
# 这些层的输出保持 FP32（精度关键操作，INT8 量化会导致精度严重下降）
FP32_OUTPUT_LAYER_KEYWORDS = [
    "rotate",
    "layernorm",
    "layer_norm",
    "softmax",
    "sigmoid",
    "reduce",
    "pow",
    "sqrt",
    "erf",
    "sin",
    "cos",
]

# 插件层保持 FP16（避免插件内部不支持 INT8 导致的量化问题）
# MSDeformAttn / SelectAndPad / RotateTRT 等自定义插件
FP16_PLUGIN_LAYER_KEYWORDS = [
    "msdeformattn",       # MSDeformAttn 插件
    "select_and_pad",     # SelectAndPad 插件
    "selectandpad",       # SelectAndPad (可能的小写形式)
    "rotate_trt",         # RotateTRT 插件
    "rotatetrt",          # RotateTRT (小写)
    "efficientscatter",   # EfficientScatter 插件 (如有)
]

# ======================== Head 输入配置 ========================
# 从 ONNX 获取的固定输入 shape
HEAD_INPUT_SHAPES = {
    "mlvl_feats.0": (1, 6, 256, 92, 160),
    "mlvl_feats.1": (1, 6, 256, 46, 80),
    "mlvl_feats.2": (1, 6, 256, 23, 40),
    "mlvl_feats.3": (1, 6, 256, 12, 20),
    "img_metas.0[shift]": (1, 2),
    "img_metas.0[can_bus]": (1, 18),
    "img_metas.0[lidar2img]": (6, 4, 4),
}

# prev_bev 动态形状范围 (已废弃 — prev_bev 现为静态 [40000, 1, 256])
# 保留 PREV_BEV_OPT 供校准器随机数据生成使用 (形状值不变,只是不再作为 profile)。
PREV_BEV_MIN = (40000, 1, 256)  # 已改为静态: min=opt=max
PREV_BEV_OPT = (40000, 1, 256)
PREV_BEV_MAX = (40000, 1, 256)

CALIB_BATCHES = 50  # 校准批次数

# ======================== 构建模式 ========================
MODE = os.environ.get("BUILD_MODE", "int8_random")  # int8_random 或 int8_real


# ======================== INT8 校准器 ========================
class HeadCalibrator(trt.IInt8EntropyCalibrator2):
    """Head INT8 校准器，支持随机数据和真实数据两种模式。

    Head 有多个输入（mlvl_feats, shift, can_bus, lidar2img, prev_bev），
    校准时需要为每个输入提供正确 shape 的数据。
    """

    def __init__(self, mode: str, cache_file: str, input_names: list):
        super().__init__()
        self.cache_file = cache_file
        self.current_idx = 0
        self.mode = mode
        self.input_names = input_names

        if mode == "int8_real":
            self.batches = self._load_real_data()
        else:
            self.batches = self._generate_random_data()

        print(f"[Calibrator] 模式: {mode}, 校准批次数: {len(self.batches)}")

    def _generate_random_data(self):
        """为 Head 的每个输入生成随机校准数据。"""
        batches = []
        for _ in range(CALIB_BATCHES):
            batch = {}
            for name, shape in HEAD_INPUT_SHAPES.items():
                batch[name] = np.random.randn(*shape).astype(np.float32) * 0.1
            # prev_bev 使用 opt shape
            batch["prev_bev"] = np.random.randn(*PREV_BEV_OPT).astype(np.float32) * 0.1
            batches.append(batch)
        return batches

    def _load_real_data(self):
        """从评估数据目录加载真实校准数据。

        优先查找 .pkl / .npy 格式的预处理数据文件。
        如果未找到真实数据，回退到随机数据。
        """
        # 搜索预处理数据文件
        data_patterns = [
            os.path.join(EVAL_DATA_DIR, "**", "*head_input*.pkl"),
            os.path.join(EVAL_DATA_DIR, "**", "*head_input*.npy"),
            os.path.join(EVAL_DATA_DIR, "**", "*calib*.pkl"),
            os.path.join(EVAL_DATA_DIR, "**", "*calib*.npy"),
        ]
        data_files = []
        for pattern in data_patterns:
            data_files.extend(glob.glob(pattern, recursive=True))

        if not data_files:
            print(f"[Calibrator] 警告: 未在 {EVAL_DATA_DIR} 找到预处理数据文件，回退到随机数据")
            return self._generate_random_data()

        print(f"[Calibrator] 找到 {len(data_files)} 个数据文件")

        batches = []
        for fpath in data_files[:CALIB_BATCHES]:
            try:
                if fpath.endswith('.pkl'):
                    import pickle
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    # 期望 data 是 dict: {input_name: numpy_array}
                    if isinstance(data, dict):
                        batches.append(data)
                elif fpath.endswith('.npy'):
                    data = np.load(fpath, allow_pickle=True)
                    if isinstance(data, np.ndarray) and data.dtype == object:
                        batches.append(data.item())
            except Exception as e:
                print(f"[Calibrator] 跳过文件 {fpath}: {e}")
                continue

        if not batches:
            print("[Calibrator] 警告: 无法加载任何数据文件，回退到随机数据")
            return self._generate_random_data()

        return batches

    def get_batch_size(self):
        """校准 batch size = 1 (Head 的 batch 维度固定为 1)。"""
        return 1

    def get_batch(self, names):
        """获取下一个校准批次。names 是输入张量名称列表。"""
        if self.current_idx >= len(self.batches):
            return None

        batch_data = self.batches[self.current_idx]
        self.current_idx += 1

        # get_batch 需要返回 GPU 指针列表
        # 使用 PyTorch 管理 GPU 内存
        ptrs = []
        self._gpu_refs = []  # 保持引用防止 GC
        for name in names:
            if isinstance(batch_data, dict) and name in batch_data:
                arr = np.ascontiguousarray(batch_data[name], dtype=np.float32)
            else:
                # 回退：使用随机数据
                if name in HEAD_INPUT_SHAPES:
                    shape = HEAD_INPUT_SHAPES[name]
                elif "prev_bev" in name.lower():
                    shape = PREV_BEV_OPT
                else:
                    shape = (1,)  # fallback
                arr = np.random.randn(*shape).astype(np.float32) * 0.1
            t = torch.from_numpy(arr).float().cuda()
            self._gpu_refs.append(t)
            ptrs.append(int(t.data_ptr()))

        return ptrs

    def read_calibration_cache(self):
        """读取校准缓存。"""
        if os.path.isfile(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                cache = f.read()
            print(f"[Calibrator] 读取缓存: {self.cache_file} ({len(cache)} bytes)")
            return cache
        return None

    def write_calibration_cache(self, cache):
        """写入校准缓存。"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"[Calibrator] 写入缓存: {self.cache_file} ({len(cache)} bytes)")


# ======================== 主编译流程 ========================
def main():
    print(f"构建模式: {MODE}")
    print(f"ONNX: {ONNX_PATH}")
    print(f"Engine: {ENGINE_PATH}")
    print(f"Plugin: {PLUGIN_LIB}")
    print(f"校准缓存: {CALIB_CACHE}")
    print()

    # 检查文件
    if not os.path.exists(ONNX_PATH):
        print(f"错误: ONNX 文件不存在: {ONNX_PATH}")
        sys.exit(1)
    if not os.path.exists(PLUGIN_LIB):
        print(f"错误: 插件库不存在: {PLUGIN_LIB}")
        sys.exit(1)

    # 加载插件库
    ctypes.CDLL(PLUGIN_LIB, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    print("插件库加载完成")

    # 创建 Builder 和 Network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 解析 ONNX
    print(f"解析 ONNX: {ONNX_PATH}")
    with open(ONNX_PATH, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  解析错误 {i}: {parser.get_error(i)}")
            sys.exit(1)

    print(f"网络信息: {network.num_inputs} 输入, {network.num_outputs} 输出, "
          f"{network.num_layers} 层")

    # 打印输入/输出信息
    input_names = []
    print("\n输入:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        input_names.append(inp.name)
        print(f"  {inp.name}: shape={tuple(inp.shape)}, dtype={inp.dtype}")
    print("\n输出:")
    output_names = []
    for i in range(network.num_outputs):
        out = network.get_output(i)
        output_names.append(out.name)
        print(f"  {out.name}: shape={tuple(out.shape)}, dtype={out.dtype}")

    # ======================== 精度控制 ========================
    # 1. 标记精度关键层的输出为 FP32（使用 strong typing）
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
                except Exception:
                    pass
                break
    print(f"\n已标记 {fp32_output_count} 个层为 FP32 输出精度 (关键字: {FP32_OUTPUT_LAYER_KEYWORDS})")

    # 2. 标记插件层保持 FP16（避免 INT8 量化干扰插件计算）
    fp16_plugin_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_name = layer.name.lower()
        for kw in FP16_PLUGIN_LAYER_KEYWORDS:
            if kw in layer_name:
                try:
                    # 设置层精度和输出类型为 FP16
                    layer.precision = trt.float16
                    for j in range(layer.num_outputs):
                        layer.set_output_type(j, trt.float16)
                    fp16_plugin_count += 1
                except Exception:
                    pass
                break
    print(f"已标记 {fp16_plugin_count} 个插件层为 FP16 精度 (关键字: {FP16_PLUGIN_LAYER_KEYWORDS})")

    # ======================== Builder 配置 ========================
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 * 1024 * 1024)  # 8GB

    # INT8 + FP16 混合精度
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    # 设置 INT8 校准器
    calibrator = HeadCalibrator(MODE, CALIB_CACHE, input_names)
    config.int8_calibrator = calibrator
    print(f"\nINT8 校准器已设置 (模式: {MODE})")

    # 精度约束策略
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # prev_bev 已改为静态 shape [40000, 1, 256] (bs=1),无需 optimization profile。
    # 以下旧动态 profile 代码保留作历史记录:
    # # prev_bev 动态形状的 optimization profile
    # profile = builder.create_optimization_profile()
    # for i in range(network.num_inputs):
    #     name = network.get_input(i).name
    #     if "prev_bev" in name.lower():
    #         profile.set_shape(name, PREV_BEV_MIN, PREV_BEV_OPT, PREV_BEV_MAX)
    #         config.add_optimization_profile(profile)
    #         print(f"已添加 optimization profile: {name} "
    #               f"min={PREV_BEV_MIN}, opt={PREV_BEV_OPT}, max={PREV_BEV_MAX}")
    #         break

    # 优化级别
    config.builder_optimization_level = 3

    # 确保输出目录存在
    os.makedirs(os.path.dirname(ENGINE_PATH), exist_ok=True)

    # ======================== 编译引擎 ========================
    print(f"\n开始编译 INT8 Mixed-Precision 引擎 (模式={MODE}, 这可能需要较长时间)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("编译失败!")
        sys.exit(1)

    # 保存引擎
    print(f"编译成功! 引擎大小: {serialized.nbytes / 1024 / 1024:.2f} MB")
    with open(ENGINE_PATH, 'wb') as f:
        f.write(serialized)
    print(f"已保存到: {ENGINE_PATH}")

    # 与 FP16 引擎大小对比
    fp16_path = os.path.join(os.path.dirname(ENGINE_PATH), "head_withprev_fp16_fixed_v2.engine")
    if os.path.exists(fp16_path):
        fp16_size = os.path.getsize(fp16_path) / 1024 / 1024
        int8_size = os.path.getsize(ENGINE_PATH) / 1024 / 1024
        print(f"\n引擎大小对比:")
        print(f"  FP16: {fp16_size:.2f} MB")
        print(f"  INT8 Mixed: {int8_size:.2f} MB")
        print(f"  压缩比: {int8_size / fp16_size:.2%}")


if __name__ == "__main__":
    main()
