"""Script to export head ONNX with prev_bev."""
import sys, os, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_ROOT = os.environ.get('VAD_DEPLOY_ROOT', os.environ.get('DEPLOY_ROOT', os.path.normpath(os.path.join(SCRIPT_DIR, '../..'))))
PROJECT_ROOT = os.environ.get('VAD_PROJECT_ROOT', os.environ.get('PROJECT_ROOT', os.path.normpath(os.path.join(DEPLOY_ROOT, '../../Bench2DriveZoo_uniad_vad'))))
ONNX_ROOT = os.environ.get('ONNX_ROOT', os.path.join(PROJECT_ROOT, 'vad_b2d_onnx'))

# CRITICAL: Patch Tensor.__eq__ to convert bool to tensor before comparison.
# The JIT tracer crashes on eq(Tensor, bool). This ensures isinstance check.
def _patch_tensor_comparisons():
    for op_name in ['__eq__', '__ne__']:
        orig_name = f'__orig_{op_name}'
        if not hasattr(torch.Tensor, orig_name):
            setattr(torch.Tensor, orig_name, getattr(torch.Tensor, op_name))
            def make_patch(op):
                def patch(self, other):
                    if isinstance(other, bool):
                        other = float(other)
                    return getattr(torch.Tensor, f'__orig_{op}')(self, other)
                return patch
            setattr(torch.Tensor, op_name, make_patch(op_name))
    print("  [PATCH] Tensor comparisons patched for bool safety")
_patch_tensor_comparisons()

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ONNX_ROOT)
os.chdir(PROJECT_ROOT)

from export_head_model import load_data, build_model, patch_head, HeadExportWrapper

# Note: VAD_head.py line 1817 was modified (~valid_map_inst).to(float32) instead of == False

x0,x1,x2,x3,sft,can,l2i,ish = load_data()
head = patch_head(build_model())
# NOTE: rotate_prev_bev=True is the CORRECT behavior for temporal BEV alignment.
# The original model has rotate_prev_bev=True by default.
# Was previously set to False to avoid GridSamplePlugin errors, 
# but the custom RotatePlugin handles this correctly.
print(f'  rotate_prev_bev: {head.transformer.rotate_prev_bev} (using default)')
wr = HeadExportWrapper(head,ish).cuda().eval()

# Generate a REAL prev_bev from model forward (not zeros) to avoid shape mismatch
with torch.no_grad():
    out = wr(x0,x1,x2,x3,sft,can,l2i,None)  # Run with no prev first
    prev_bev = out[0]  # Use bev_embed as prev_bev (same shape: [40000, 1, 256])
    # Re-run with actual prev_bev for proper trace
    out = wr(x0,x1,x2,x3,sft,can,l2i,prev_bev)
print(f'Test forward OK, prev_bev shape: {prev_bev.shape}')

ONNX=os.path.join(DEPLOY_ROOT, 'onnx_export/output/vad_head_withprev.onnx')
os.makedirs(os.path.dirname(ONNX),exist_ok=True)

# Common parameters
all_inputs = (x0,x1,x2,x3,sft,can,l2i,prev_bev)
input_names = ['mlvl_feats.0','mlvl_feats.1','mlvl_feats.2','mlvl_feats.3',
               'img_metas.0[shift]','img_metas.0[can_bus]','img_metas.0[lidar2img]',
               'prev_bev']
output_names = ['out.bev_embed','out.all_cls_scores','out.all_bbox_preds','out.all_traj_preds',
                'out.all_traj_cls_scores','out.map_all_cls_scores','out.map_all_bbox_preds',
                'out.map_all_pts_preds','out.ego_fut_preds']

# Dynamic axes for variable-size dimensions.
# NOTE: prev_bev (input) must stay STATIC ([40000, 1, 256] at bs=1). If it is
# exported with dynamic axes, TensorRT refuses to build the engine without an
# optimization profile ("Network has dynamic or shape inputs, but no
# optimization profile has been defined") — the INT8 builders set none, by
# design. Outputs may stay dynamic (TRT only constrains inputs).
dynamic_axes = {
    'out.bev_embed': {0: 'num_bev', 1: 'bev_dim1', 2: 'bev_dim2'},
    'out.all_cls_scores': {2: 'num_query'},
    'out.all_bbox_preds': {2: 'num_query'},
    'out.all_traj_preds': {2: 'num_query', 4: 'traj_dim'},
    'out.all_traj_cls_scores': {2: 'num_query'},
    'out.map_all_cls_scores': {2: 'num_map'},
    'out.map_all_bbox_preds': {2: 'num_map'},
    'out.map_all_pts_preds': {2: 'num_map', 3: 'pts_dim'},
    'out.ego_fut_preds': {1: 'fut_mode'},
}

# Try jit.script first
print('Trying jit.script...')
export_success = False
try:
    traced = torch.jit.script(wr)
    print('jit.script OK!')
    torch.onnx.export(traced,all_inputs,ONNX,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=15,do_constant_folding=False)
    export_success = True
except Exception as e:
    print(f'jit.script failed: {type(e).__name__}')

if not export_success:
    print('Falling back to torch.onnx.export...')
    torch.onnx.export(wr,all_inputs,ONNX,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=15,do_constant_folding=False,verbose=False)

print(f'ONNX saved: {ONNX} ({os.path.getsize(ONNX)/1024/1024:.1f}MB)' if os.path.exists(ONNX) else 'ONNX NOT CREATED')
