"""Export backbone (ResNet50 + FPN) ONNX with precision verification.

产物 (两份契约, 分别服务不同下游):
  1. output/vad_backbone.onnx     输入 [6,3,736,1280]  -> 输出 [6,1,256,H,W]
     旧契约 (无 batch 维, 相机维当 batch), 与 x86/trt_compile 的
     build_backbone_int8.py 等脚本保持兼容.
  2. output/vad_backbone_b1.onnx  输入 [1,6,3,736,1280] -> 输出 [1,6,256,H,W]
     新契约 (带 batch 维), 与 vad_head_withprev.onnx / hous_reproduce 链条对齐,
     无需下游 reshape 桥接.

验证: 3 帧真实数据, PyTorch(GPU, TF32 off) vs ORT(CPU) 逐输出 max_abs_err.
注意: GridMask 为训练期增广, 导出前显式关闭 (use_grid_mask=False).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_ROOT = os.environ.get('VAD_DEPLOY_ROOT', os.environ.get('DEPLOY_ROOT', os.path.normpath(os.path.join(SCRIPT_DIR, '../..'))))
PROJECT_ROOT = os.environ.get('VAD_PROJECT_ROOT', os.environ.get('PROJECT_ROOT', os.path.normpath(os.path.join(DEPLOY_ROOT, '../../Bench2DriveZoo_uniad_vad'))))

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from torch.utils.data import DataLoader
from mmcv.parallel import collate
from mmcv.utils import Config, load_checkpoint
from mmcv.datasets import build_dataset
from mmcv.models import build_model

OUT_DIR = os.path.join(DEPLOY_ROOT, 'onnx_export/output')
OUT_LEGACY = os.path.join(OUT_DIR, 'vad_backbone.onnx')
OUT_B1 = os.path.join(OUT_DIR, 'vad_backbone_b1.onnx')

OPSET = 17


def build_model_loader():
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile('adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py')
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model)
    load_checkpoint(model, 'ckpts/vad_b2d_base.pth', map_location='cpu')
    model.cuda().eval()
    model.use_grid_mask = False
    dataset = build_dataset(cfg.data.test)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    return model, loader


class BackboneLegacy(torch.nn.Module):
    """旧契约: 输入 [6,3,H,W] -> 输出 [6,1,256,H,W] (extract_feat 的 dim=4 分支)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        return tuple(self.model.extract_feat(img))


class BackboneB1(torch.nn.Module):
    """新契约: 输入 [1,6,3,H,W] -> 输出 [1,6,256,H,W] (squeeze 分支, head 直连)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        return tuple(self.model.extract_feat(img))


def export_and_verify(model, loader, wrapper, sample_img, onnx_path, tag, expect_out0):
    torch.onnx.export(wrapper, sample_img, onnx_path,
                      input_names=['img'],
                      output_names=['out.0', 'out.1', 'out.2', 'out.3'],
                      opset_version=OPSET, do_constant_folding=True)
    print(f'[{tag}] saved: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f}MB)')

    import onnx
    g = onnx.load(onnx_path, load_external_data=False).graph
    print(f'[{tag}] input: {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input]}')

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    worst_all = 0.0
    with torch.no_grad():
        for idx, data in enumerate(loader):
            img = data['img'][0]
            if tag == 'legacy':
                img = img.squeeze(0)
            img = img.cuda()
            feed = {'img': img.cpu().numpy()}   # 快照在前: extract_feat 的 squeeze_ 会原地改输入
            with torch.no_grad():
                pt_outs = wrapper(img)
            ort_outs = sess.run(None, feed)
            for i, (po, oo) in enumerate(zip(pt_outs, ort_outs)):
                err = np.abs(po.cpu().numpy() - oo).max()
                worst_all = max(worst_all, err)
            print(f'  frame{idx} worst max_abs_err = {worst_all:.3e}')
            if idx >= 2:
                break
    assert worst_all < 1e-2, f'[{tag}] 精度超阈值'
    print(f'[{tag}] 精度验收通过 (阈值 1e-2)')


def main():
    model, loader = build_model_loader()

    sample = None
    for idx, data in enumerate(loader):
        sample = data['img'][0].cuda()
        break
    print('sample img:', tuple(sample.shape))

    export_and_verify(model, loader, BackboneLegacy(model).cuda().eval(),
                      sample.squeeze(0), OUT_LEGACY, 'legacy', (6, 1, 256, 92, 160))
    export_and_verify(model, loader, BackboneB1(model).cuda().eval(),
                      sample, OUT_B1, 'b1', (1, 6, 256, 92, 160))

    print('全部完成: legacy 供 step3 使用; b1 与 head 链条对齐')


if __name__ == '__main__':
    main()
