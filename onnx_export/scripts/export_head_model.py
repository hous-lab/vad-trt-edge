"""Export head ONNX with prev_bev. Used by export_head_onnx.py"""
import sys, os, torch, json, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_ROOT = os.environ.get('VAD_DEPLOY_ROOT', os.environ.get('DEPLOY_ROOT', os.path.normpath(os.path.join(SCRIPT_DIR, '../..'))))
PROJECT_ROOT = os.environ.get('VAD_PROJECT_ROOT', os.environ.get('PROJECT_ROOT', os.path.normpath(os.path.join(DEPLOY_ROOT, '../../Bench2DriveZoo_uniad_vad'))))
ONNX_ROOT = os.environ.get('ONNX_ROOT', os.path.join(PROJECT_ROOT, 'vad_b2d_onnx'))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ONNX_ROOT)
os.chdir(PROJECT_ROOT)

DATA_DIR = os.environ.get('DATA_DIR', os.path.join(DEPLOY_ROOT, 'onnx_export/demo_data/b2d_vadv1_prev.pts_bbox_head.forward'))

def load_data():
    def load(n,s): return torch.from_numpy(np.fromfile(f'{DATA_DIR}/{n}.bin',dtype=np.float32).reshape(s)).cuda()
    x0=load('mlvl_feats.0',(1,6,256,92,160));x1=load('mlvl_feats.1',(1,6,256,46,80))
    x2=load('mlvl_feats.2',(1,6,256,23,40));x3=load('mlvl_feats.3',(1,6,256,12,20))
    sft=load('img_metas.0[shift]',(1,2));can=load('img_metas.0[can_bus]',(1,18));l2i=load('img_metas.0[lidar2img]',(6,4,4))
    ish=torch.from_numpy(np.fromfile(f'{DATA_DIR}/img_metas.0[img_shape].bin',dtype=np.float32)).reshape(1,-1).cuda()
    return x0,x1,x2,x3,sft,can,l2i,ish

def build_model():
    from mmcv.utils import Config, load_checkpoint as lc
    from mmcv.models import build_model as bm
    cfg=Config.fromfile('adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py');cfg.model.train_cfg=None;cfg.model.pretrained=None
    m=bm(cfg.model,test_cfg=cfg.get('test_cfg'));lc(m,'ckpts/vad_b2d_base.pth',map_location='cpu')
    m=m.cuda().eval(); head=m.pts_bbox_head
    return head

def patch_head(head):
    from bev_deploy.patch.bevformer.patch import patch_spatial_cross_attn_forward as psca,patch_point_sampling as pps,patch_bevformer_encoder_forward as pbef
    from patch.patch_head import patch_VADHead_forward as pvhf,patch_VADPerceptionTransformer_get_bev_features as pvpgb
    from patch.patch_head import patch_VADHead_select_and_pad_query as pvhsq
    from patch.patch_head import patch_VADHead_select_and_pad_pred_map as pvhsapm
    for lyr in range(6):
        mod=head.transformer.encoder.layers[lyr].attentions[1]
        head.transformer.encoder.layers[lyr].attentions[1].forward=lambda *a,_m=mod,**kw:psca(_m,*a,**kw)
    head.transformer.encoder.point_sampling=lambda *a,**kw:pps(head.transformer.encoder,*a,**kw)
    head.transformer.encoder.forward=lambda *a,**kw:pbef(head.transformer.encoder,*a,**kw)
    head.transformer.get_bev_features=lambda *a,**kw:pvpgb(head.transformer,*a,**kw)
    head.select_and_pad_query=lambda q,p,c,**kw:pvhsq(head,q,p,c,**kw)
    head.select_and_pad_pred_map=lambda *a,**kw:pvhsapm(head,*a,**kw)
    head.patched_forward=lambda *a,**kw:pvhf(head,*a,**kw)
    return head

class HeadExportWrapper(torch.nn.Module):
    def __init__(self,head,ish):
        super().__init__()
        self.head=head
        self.register_buffer('img_shape_val',ish)
    def forward(self,x0,x1,x2,x3,sft,can,l2i,prev_bev=None):
        r=self.head.patched_forward([x0,x1,x2,x3],[{'lidar2img':l2i,'shift':sft,'can_bus':can,'img_shape':self.img_shape_val}],
                                     prev_bev=prev_bev,only_bev=False)
        return(r['bev_embed'],r['all_cls_scores'],r['all_bbox_preds'],r['all_traj_preds'],
               r['all_traj_cls_scores'],r['map_all_cls_scores'],r['map_all_bbox_preds'],
               r['map_all_pts_preds'],r['ego_fut_preds'])
