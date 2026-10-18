"""Build VAD Backbone INT8 TRT engine for AGX Orin (ARM64).

在 Orin 设备上运行，使用 Orin 编译的插件库和共享的 ONNX 模型。
支持两种校准模式：
  - int8_random: 使用随机数据校准（快速验证，Backbone 是 CNN 对校准数据不敏感）
  - int8_real:   使用真实评估数据校准（最终部署）

通过环境变量 BUILD_MODE 控制，默认 int8_random。

Backbone 输入: img [6, 3, 736, 1280] (6 摄像头, 3 通道, 736x1280)
Backbone 输出: 4 个多尺度特征图 (FPN)
"""
import os
import sys
import ctypes
import glob

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
ONNX_PATH = os.path.join(PIPELINE_ROOT, "onnx_export/output/vad_backbone.onnx")
ENGINE_PATH = os.path.join(PIPELINE_ROOT, "arm64/trt_compile/output/backbone_int8_orin.engine")
PLUGIN_LIB = os.path.join(PIPELINE_ROOT, "arm64/plugins_build/build/libplugins_orin.so")
CALIB_CACHE = os.path.join(PIPELINE_ROOT, "data/calib_data/backbone_int8_orin.calib")
EVAL_DATA_DIR = os.path.join(PIPELINE_ROOT, "data")

# ======================== Backbone 输入配置 ========================
# 从 ONNX 文件获取: img: [6, 3, 736, 1280]
BATCH_SIZE = 6       # 6 个摄像头
CHANNELS = 3
HEIGHT = 736
WIDTH = 1280
CALIB_BATCHES = 10    # 校准批次数 (Backbone CNN 对校准数据不敏感, 10 批足够)

# ======================== 构建模式 ========================
MODE = os.environ.get("BUILD_MODE", "int8_random")  # int8_random 或 int8_real


# ======================== INT8 校准器 ========================
class BackboneCalibrator(trt.IInt8EntropyCalibrator2):
    """Backbone INT8 校准器，支持随机数据和真实数据两种模式。"""

    def __init__(self, mode: str, cache_file: str):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = BATCH_SIZE
        self.current_idx = 0
        self.mode = mode

        if mode == "int8_real":
            self.batches = self._load_real_data()
        else:
            # 随机数据校准：使用 ImageNet 均值/方差归一化的随机数据
            # Backbone (ResNet/EfficientNet 等 CNN) 对校准数据不太敏感
            self.batches = self._generate_random_data()

        print(f"[Calibrator] 模式: {mode}, 校准批次数: {len(self.batches)}, "
              f"每批 shape: ({BATCH_SIZE}, {CHANNELS}, {HEIGHT}, {WIDTH})")

    def _generate_random_data(self):
        """生成随机校准数据（模拟 ImageNet 归一化后的输入）。"""
        # ImageNet 均值 [0.485, 0.456, 0.406], 方差 [0.229, 0.224, 0.225]
        # 这里使用 [0, 1] 范围随机数，实际部署时精度影响很小
        batches = []
        for _ in range(CALIB_BATCHES):
            data = np.random.rand(self.batch_size, CHANNELS, HEIGHT, WIDTH).astype(np.float32)
            batches.append(data)
        return batches

    def _load_real_data(self):
        """从评估数据目录加载真实校准图像。"""
        batches = []

        # 搜索图像文件 (支持常见格式)
        img_patterns = [
            os.path.join(EVAL_DATA_DIR, "**", "*.jpg"),
            os.path.join(EVAL_DATA_DIR, "**", "*.png"),
            os.path.join(EVAL_DATA_DIR, "**", "*.npy"),
        ]
        img_files = []
        for pattern in img_patterns:
            img_files.extend(glob.glob(pattern, recursive=True))

        if not img_files:
            print(f"[Calibrator] 警告: 未在 {EVAL_DATA_DIR} 找到图像文件，回退到随机数据")
            return self._generate_random_data()

        print(f"[Calibrator] 找到 {len(img_files)} 个图像文件")

        # 按 BATCH_SIZE 分组加载
        # 注意：Backbone 输入是 [6, 3, H, W]，即一次需要 6 张图
        num_complete_batches = len(img_files) // self.batch_size
        for batch_idx in range(min(num_complete_batches, CALIB_BATCHES)):
            batch_imgs = []
            start = batch_idx * self.batch_size
            for j in range(self.batch_size):
                fpath = img_files[start + j]
                if fpath.endswith('.npy'):
                    img = np.load(fpath)
                else:
                    # 使用 OpenCV/PIL 加载并预处理
                    try:
                        import cv2
                        img = cv2.imread(fpath)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (WIDTH, HEIGHT))
                        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                    except ImportError:
                        # 没有 cv2 则用 PIL
                        from PIL import Image
                        img = Image.open(fpath).convert('RGB')
                        img = img.resize((WIDTH, HEIGHT))
                        img = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
                batch_imgs.append(img)

            batch_data = np.stack(batch_imgs, axis=0).astype(np.float32)
            batches.append(batch_data)

        if not batches:
            print("[Calibrator] 警告: 无法组成完整批次，回退到随机数据")
            return self._generate_random_data()

        return batches

    def get_batch_size(self):
        """返回校准批次大小。"""
        return self.batch_size

    def get_batch(self, names):
        """获取下一个校准批次。返回指向 GPU 内存的指针列表。

        使用 PyTorch 直接管理 GPU 内存（共享 TRT 的 CUDA 上下文）。
        """
        if self.current_idx >= len(self.batches):
            return None

        batch = self.batches[self.current_idx]
        self.current_idx += 1

        # 用 torch 拷贝到 GPU（与 TRT 共享主 CUDA 上下文，避免 pycuda 上下文冲突）
        t = torch.from_numpy(batch).float().cuda()
        self._device_mem = t  # 保持引用防止 GC
        return [int(t.data_ptr())]

    def read_calibration_cache(self):
        """读取校准缓存文件。"""
        if os.path.isfile(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                cache = f.read()
            print(f"[Calibrator] 读取缓存: {self.cache_file} ({len(cache)} bytes)")
            return cache
        return None

    def write_calibration_cache(self, cache):
        """写入校准缓存文件。"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"[Calibrator] 写入缓存: {self.cache_file} ({len(cache)} bytes)")


# ======================== 主编译流程 ========================
def main():
    print("=" * 60)
    print("VAD Backbone INT8 Orin 编译")
    print("=" * 60)
    print(f"TRT 版本: {trt.__version__} (major={TRT_MAJOR})")
    print(f"构建模式: {MODE}")
    print(f"ONNX:     {ONNX_PATH}")
    print(f"Engine:   {ENGINE_PATH}")
    print(f"Plugin:   {PLUGIN_LIB}")
    print(f"校准缓存: {CALIB_CACHE}")
    print("=" * 60)
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
    print("\n输入:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  {inp.name}: shape={tuple(inp.shape)}, dtype={inp.dtype}")
    print("\n输出:")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {out.name}: shape={tuple(out.shape)}, dtype={out.dtype}")

    # 配置 Builder
    config = builder.create_builder_config()
    set_workspace(config, 8 * 1024 * 1024 * 1024)  # 8GB

    # INT8 + FP16 混合精度
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    # 设置 INT8 校准器
    calibrator = BackboneCalibrator(MODE, CALIB_CACHE)
    config.int8_calibrator = calibrator
    print(f"\nINT8 校准器已设置 (模式: {MODE})")

    # 精度约束
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # 优化级别
    config.builder_optimization_level = 3

    # 编译引擎
    print(f"\n开始编译 INT8 引擎 (模式={MODE}, 这可能需要较长时间)...")
    serialized = build_engine(builder, network, config)

    # 保存引擎
    print(f"编译成功! 引擎大小: {serialized.nbytes / 1024 / 1024:.2f} MB")
    save_engine(serialized, ENGINE_PATH)

    # 与 FP16 引擎大小对比
    fp16_path = os.path.join(PIPELINE_ROOT, "arm64/trt_compile/output/backbone_fp16_orin.engine")
    if os.path.exists(fp16_path):
        fp16_size = os.path.getsize(fp16_path) / 1024 / 1024
        int8_size = os.path.getsize(ENGINE_PATH) / 1024 / 1024
        print(f"\n引擎大小对比:")
        print(f"  FP16: {fp16_size:.2f} MB")
        print(f"  INT8: {int8_size:.2f} MB")
        print(f"  压缩比: {int8_size / fp16_size:.2%}")

    print("完成!")


if __name__ == "__main__":
    main()
