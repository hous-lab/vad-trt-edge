#!/usr/bin/env python3
"""
VAD Module-Level Profiling Script
=================================
Profiles each module of VAD end-to-end model with CUDA events for precise latency breakdown.

Usage:
    # PyTorch profiling (no TRT needed)
    python profile_vad_modules.py --num-samples 10
    
    # TRT engine profiling (requires compiled engines)
    python profile_vad_modules.py --num-samples 10 --trt-backbone-engine <path> --trt-head-engine <path>

Output:
    - Per-module latency breakdown (ms)
    - Percentage of total time
    - Bottleneck identification
"""

import os, sys, time, json, copy, ctypes, argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# ============ Configuration ============
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.environ.get("VAD_PROJECT_ROOT",
    os.path.normpath(os.path.join(PIPELINE_ROOT, "..", "..", "Bench2DriveZoo_uniad_vad")))
DEPLOY_ROOT = os.environ.get("VAD_DEPLOY_ROOT", PIPELINE_ROOT)

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# ============ Parse args ============
parser = argparse.ArgumentParser(description="VAD Module-Level Profiling")
parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to profile")
parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
parser.add_argument("--output", type=str, default=None, help="Output JSON path")
parser.add_argument("--trt-backbone-engine", type=str, default=None,
                    help="Path to TRT backbone engine (for TRT profiling)")
parser.add_argument("--trt-head-engine", type=str, default=None,
                    help="Path to TRT head engine (for TRT profiling)")
parser.add_argument("--profile-trt", action="store_true", default=False,
                    help="Also profile with TRT engines if available")
parser.add_argument("--detail", action="store_true", default=True,
                    help="Enable per-layer profiling (encoder/decoder layers)")
args = parser.parse_args()

# ============ CUDA Event Timer ============
class CudaTimer:
    """CUDA event-based timer for precise GPU kernel timing."""
    def __init__(self, name="", enabled=True):
        self.name = name
        self.enabled = enabled
        self.events = []
        self.times = []
    
    def start(self, name=None):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events.append(("start", name or self.name, ev))
    
    def stop(self, name=None):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events.append(("stop", name or self.name, ev))
    
    def get_results(self):
        """Process events and return {name: total_ms} dict."""
        stack = {}
        totals = {}
        for ev_type, name, ev in self.events:
            if ev_type == "start":
                stack[name] = ev
            else:
                if name in stack:
                    elapsed = stack[name].elapsed_time(ev)  # ms
                    totals[name] = totals.get(name, 0) + elapsed
                    del stack[name]
        # Convert to ms
        return totals


# ============ Model Loading ============
def load_model():
    from mmcv.utils import Config
    from mmcv.models import build_model
    from mmcv.utils import load_checkpoint
    
    CONFIG_PATH = "adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py"
    CKPT_PATH = os.path.join(DEPLOY_ROOT, "ckpts/vad_b2d_base.pth")
    
    print(f"Loading model from {CONFIG_PATH}...")
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CKPT_PATH, map_location='cpu')
    model = model.cuda().eval()
    model.video_test_mode = True
    print("Model loaded successfully.")
    return model


# ============ Profiling Wrappers ============
def create_profiled_model(model, timer):
    """Patch model with profiling timers."""
    head = model.pts_bbox_head
    trans = head.transformer
    
    # --- 1. Backbone (extract_img_feat) ---
    _orig_extract = model.extract_img_feat
    def _profiled_extract(img, img_metas, len_queue=None):
        timer.start("1_backbone")
        result = _orig_extract(img, img_metas, len_queue)
        timer.stop("1_backbone")
        return result
    model.extract_img_feat = _profiled_extract
    
    # --- 2. BEVFormer Encoder (per-layer) ---
    if args.detail and hasattr(trans.encoder, 'layers'):
        for i, layer in enumerate(trans.encoder.layers):
            attns = layer.attentions
            for j, attn in enumerate(attns):
                attn_name = {0: "temporal_self_attn", 1: "spatial_cross_attn"}.get(j, f"attn_{j}")
                _orig_attn = attn.forward
                # Use default args to capture loop variables by value (fix closure bug)
                def _make_attn_prof(orig_fn, layer_idx, attn_label):
                    def _profiled_inner(*args, **kw):
                        timer.start(f"2b_encoder_L{layer_idx}_{attn_label}")
                        r = orig_fn(*args, **kw)
                        timer.stop(f"2b_encoder_L{layer_idx}_{attn_label}")
                        return r
                    return _profiled_inner
                attn.forward = _make_attn_prof(_orig_attn, i, attn_name)
    
    # --- 3. Detection Decoder (per-layer) ---
    if args.detail and hasattr(trans, 'decoder') and trans.decoder is not None:
        for i, layer in enumerate(trans.decoder.layers):
            _orig_dec_fwd = layer.forward
            def _make_dec_prof(orig_fn, idx):
                def _profiled_inner(*args, **kw):
                    timer.start(f"3_det_decoder_L{idx}")
                    r = orig_fn(*args, **kw)
                    timer.stop(f"3_det_decoder_L{idx}")
                    return r
                return _profiled_inner
            layer.forward = _make_dec_prof(_orig_dec_fwd, i)
    
    # --- 4. Map Decoder (per-layer) ---
    if args.detail and hasattr(trans, 'map_decoder') and trans.map_decoder is not None:
        for i, layer in enumerate(trans.map_decoder.layers):
            _orig_map_fwd = layer.forward
            def _make_map_prof(orig_fn, idx):
                def _profiled_inner(*args, **kw):
                    timer.start(f"4_map_decoder_L{idx}")
                    r = orig_fn(*args, **kw)
                    timer.stop(f"4_map_decoder_L{idx}")
                    return r
                return _profiled_inner
            layer.forward = _make_map_prof(_orig_map_fwd, i)
    
    # --- 5. Post-processing (detection cls/reg) ---
    _orig_decoder_post = head.forward
    # We'll use the head's forward method to time coarse sections
    
    # --- 6. Motion Decoder ---
    if head.motion_decoder is not None:
        _orig_motion_fwd = head.motion_decoder.forward
        def _make_motion_prof(orig_fn):
            def _profiled(*args, **kw):
                timer.start("6_motion_decoder")
                r = orig_fn(*args, **kw)
                timer.stop("6_motion_decoder")
                return r
            return _profiled
        head.motion_decoder.forward = _make_motion_prof(_orig_motion_fwd)
    
    # --- 7. Motion-Map Decoder ---
    if head.motion_map_decoder is not None:
        _orig_mm_fwd = head.motion_map_decoder.forward
        def _make_mm_prof(orig_fn):
            def _profiled(*args, **kw):
                timer.start("7_motion_map_decoder")
                r = orig_fn(*args, **kw)
                timer.stop("7_motion_map_decoder")
                return r
            return _profiled
        head.motion_map_decoder.forward = _make_mm_prof(_orig_mm_fwd)
    
    # --- 8. Ego-Agent Decoder ---
    if head.ego_agent_decoder is not None:
        _orig_ea_fwd = head.ego_agent_decoder.forward
        def _make_ea_prof(orig_fn):
            def _profiled(*args, **kw):
                timer.start("8_ego_agent_decoder")
                r = orig_fn(*args, **kw)
                timer.stop("8_ego_agent_decoder")
                return r
            return _profiled
        head.ego_agent_decoder.forward = _make_ea_prof(_orig_ea_fwd)
    
    # --- 9. Ego-Map Decoder ---
    if head.ego_map_decoder is not None:
        _orig_em_fwd = head.ego_map_decoder.forward
        def _make_em_prof(orig_fn):
            def _profiled(*args, **kw):
                timer.start("9_ego_map_decoder")
                r = orig_fn(*args, **kw)
                timer.stop("9_ego_map_decoder")
                return r
            return _profiled
        head.ego_map_decoder.forward = _make_em_prof(_orig_em_fwd)
    
    return model


# ============ TRT Engine Profiling ============
def load_trt_engines(bb_path, head_path):
    """Load TRT engines for profiling."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    # Load plugins
    plugin_libs = [
        os.path.join(DEPLOY_ROOT, "plugins/build/libplugins.so"),
        os.path.join(DEPLOY_ROOT, "arm64/plugins_build/build/libplugins_orin.so"),
    ]
    for lib in plugin_libs:
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            print(f"Loaded plugin: {lib}")
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), '')
    
    # Backbone engine
    bb_engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open(bb_path, 'rb').read())
    bb_ctx = bb_engine.create_execution_context()
    bb_stream = cuda.Stream()
    
    bb_io_names = []
    bb_out_shapes = []
    for i in range(bb_engine.num_io_tensors):
        n = bb_engine.get_tensor_name(i)
        m = bb_engine.get_tensor_mode(n)
        s = tuple(bb_engine.get_tensor_shape(n))
        if m == trt.TensorIOMode.OUTPUT:
            bb_out_shapes.append(s)
        bb_io_names.append(n)
    
    # Head engine
    head_engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open(head_path, 'rb').read())
    head_ctx = head_engine.create_execution_context()
    head_stream = cuda.Stream()
    
    head_io_names = []
    head_in_shapes = []
    head_out_shapes = []
    for i in range(head_engine.num_io_tensors):
        n = head_engine.get_tensor_name(i)
        m = head_engine.get_tensor_mode(n)
        s = tuple(head_engine.get_tensor_shape(n))
        if m == trt.TensorIOMode.INPUT:
            head_in_shapes.append((n, s))
        else:
            head_out_shapes.append((n, s))
        head_io_names.append(n)
    
    return (bb_engine, bb_ctx, bb_stream, bb_io_names, bb_out_shapes,
            head_engine, head_ctx, head_stream, head_io_names, head_in_shapes, head_out_shapes)


# ============ Main Profiling ============
def main():
    print("=" * 60)
    print("VAD Module-Level Profiling")
    print("=" * 60)
    print(f"Num samples: {args.num_samples}, Warmup: {args.warmup}")
    print(f"Detail profiling: {args.detail}")
    print()
    
    # Load model
    model = load_model()
    head = model.pts_bbox_head
    trans = head.transformer
    
    # Model architecture summary
    print(f"\nModel Architecture:")
    print(f"  BEV: {head.bev_h}x{head.bev_w} = {head.bev_h * head.bev_w} queries")
    print(f"  Encoder: {len(trans.encoder.layers)} layers")
    print(f"  Det Decoder: {len(trans.decoder.layers)} layers" if trans.decoder else "  Det Decoder: None")
    print(f"  Map Decoder: {len(trans.map_decoder.layers)} layers" if trans.map_decoder else "  Map Decoder: None")
    print(f"  Motion Decoder: {head.motion_decoder is not None}")
    print(f"  Motion-Map Decoder: {head.motion_map_decoder is not None}")
    print(f"  Ego-Agent Decoder: {head.ego_agent_decoder is not None}")
    print(f"  Ego-Map Decoder: {head.ego_map_decoder is not None}")
    print(f"  FPN Levels: {trans.num_feature_levels}")
    print(f"  Temporal Queue: {len(head.transformer.encoder.layers) > 0} (with_prev)" if hasattr(model, 'video_test_mode') else "")
    print()
    
    # Build dataset
    from mmcv.utils import Config
    from mmcv.datasets import build_dataset
    from torch.utils.data import DataLoader as TorchDataLoader
    from mmcv.parallel import collate
    
    CONFIG_PATH = "adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py"
    cfg = Config.fromfile(CONFIG_PATH)
    dataset = build_dataset(cfg.data.test)
    data_loader = TorchDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    
    # ========== PyTorch Profiling ==========
    print("=" * 60)
    print("PyTorch Module-Level Profiling")
    print("=" * 60)
    
    # Create timer and profiled model
    pytorch_timer = CudaTimer(enabled=True)
    profiled_model = create_profiled_model(model, pytorch_timer)
    
    # Warmup
    print(f"Warming up for {args.warmup} iterations...")
    warmup_count = 0
    for batch_data in data_loader:
        if warmup_count >= args.warmup:
            break
        cuda_batch = {}
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                cuda_batch[k] = v.cuda()
            elif isinstance(v, list):
                cuda_batch[k] = [e.cuda() if isinstance(e, torch.Tensor) else e for e in v]
        with torch.no_grad():
            profiled_model(cuda_batch, return_loss=False, rescale=True)
        warmup_count += 1
        if warmup_count % 5 == 0:
            print(f"  Warmup {warmup_count}/{args.warmup}")
    
    # Also measure coarse head latency
    head_coarse_timer = CudaTimer(enabled=True)
    _orig_head_fwd = head.forward
    def _profiled_head_fwd(*args, **kw):
        head_coarse_timer.start("head_total")
        r = _orig_head_fwd(*args, **kw)
        head_coarse_timer.stop("head_total")
        return r
    head.forward = _profiled_head_fwd
    
    # Also measure encoder total
    _orig_enc_fwd = trans.encoder.forward
    def _profiled_enc_fwd(*args, **kw):
        head_coarse_timer.start("encoder_total")
        r = _orig_enc_fwd(*args, **kw)
        head_coarse_timer.stop("encoder_total")
        return r
    trans.encoder.forward = _profiled_enc_fwd
    
    # Also measure decoder total
    if trans.decoder is not None:
        _orig_dec_fwd = trans.decoder.forward
        def _profiled_dec_fwd(*args, **kw):
            head_coarse_timer.start("det_decoder_total")
            r = _orig_dec_fwd(*args, **kw)
            head_coarse_timer.stop("det_decoder_total")
            return r
        trans.decoder.forward = _profiled_dec_fwd
    
    # Also measure map decoder total
    if trans.map_decoder is not None:
        _orig_map_dec_fwd = trans.map_decoder.forward
        def _profiled_map_dec_fwd(*args, **kw):
            head_coarse_timer.start("map_decoder_total")
            r = _orig_map_dec_fwd(*args, **kw)
            head_coarse_timer.stop("map_decoder_total")
            return r
        trans.map_decoder.forward = _profiled_map_dec_fwd
    
    # Run profiling
    print(f"Profiling {args.num_samples} samples...")
    all_latencies = []
    all_module_times = []
    processed = 0
    
    # Track first frame separately (no prev_bev)
    first_frame_latency = None
    first_frame_modules = {}
    
    for batch_data in data_loader:
        if processed >= args.num_samples:
            break
        
        # Reset timers
        pytorch_timer.events = []
        head_coarse_timer.events = []
        
        cuda_batch = {}
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                cuda_batch[k] = v.cuda()
            elif isinstance(v, list):
                cuda_batch[k] = [e.cuda() if isinstance(e, torch.Tensor) else e for e in v]
        
        torch.cuda.synchronize()
        with torch.no_grad():
            t0 = time.perf_counter()
            result = profiled_model(cuda_batch, return_loss=False, rescale=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        
        total_ms = (t1 - t0) * 1000
        all_latencies.append(total_ms)
        
        # Get module-level times
        fine_times = pytorch_timer.get_results()
        coarse_times = head_coarse_timer.get_results()
        
        # Merge
        module_times = {**fine_times, **coarse_times}
        all_module_times.append(module_times)
        
        if processed == 0:
            first_frame_latency = total_ms
            first_frame_modules = module_times
        
        processed += 1
        if processed % 5 == 0:
            print(f"  {processed}/{args.num_samples}, E2E: {total_ms:.1f}ms")
    
    # Compute averages
    avg_total = np.mean(all_latencies)
    print(f"\n{'='*60}")
    print(f"PyTorch Profiling Results ({args.num_samples} samples, {args.warmup} warmup)")
    print(f"{'='*60}")
    print(f"  End-to-End Avg: {avg_total:.1f}ms")
    if first_frame_latency is not None:
        print(f"  First Frame (no prev_bev): {first_frame_latency:.1f}ms")
    print(f"  Subsequent Frames Avg: {np.mean(all_latencies[1:]):.1f}ms" if len(all_latencies) > 1 else "")
    
    # Aggregate module times
    module_keys = set()
    for mt in all_module_times:
        module_keys.update(mt.keys())
    
    module_avgs = {}
    for k in sorted(module_keys):
        vals = [mt.get(k, 0) for mt in all_module_times]
        module_avgs[k] = np.mean(vals)
    
    print(f"\n  Module-Level Latency Breakdown:")
    print(f"  {'Module':<40s} {'Avg (ms)':<12s} {'% of E2E':<12s}")
    print(f"  {'-'*40} {'-'*12} {'-'*12}")
    
    # Sort by time descending
    sorted_modules = sorted(module_avgs.items(), key=lambda x: -x[1])
    for name, avg_ms in sorted_modules:
        if avg_ms < 0.1:
            continue
        pct = avg_ms / avg_total * 100 if avg_total > 0 else 0
        print(f"  {name:<40s} {avg_ms:<12.1f} {pct:<11.1f}%")
    
    # ========== Summary Analysis ==========
    print(f"\n{'='*60}")
    print(f"Bottleneck Analysis")
    print(f"{'='*60}")
    
    # Group by coarse category
    categories = {
        "Backbone (ResNet50+FPN)": ["1_backbone"],
        "BEVFormer Encoder": [k for k in module_avgs.keys() if "encoder" in k.lower()],
        "Detection Decoder": [k for k in module_avgs.keys() if "det_decoder" in k.lower() or "3_det" in k.lower()],
        "Map Decoder": [k for k in module_avgs.keys() if "map_decoder" in k.lower() or "4_map" in k.lower()],
        "Motion Decoder": [k for k in module_avgs.keys() if "motion" in k.lower()],
        "Ego Decoder": [k for k in module_avgs.keys() if "ego" in k.lower()],
        "Head Total": ["head_total"],
    }
    
    for cat_name, keys in categories.items():
        cat_time = sum(module_avgs.get(k, 0) for k in keys if k in module_avgs)
        if cat_time > 0:
            pct = cat_time / avg_total * 100
            bar = "█" * int(pct / 2)
            print(f"  {cat_name:<25s} {cat_time:>8.1f}ms ({pct:>5.1f}%) {bar}")
    
    # ========== TRT Profiling ==========
    if args.profile_trt and args.trt_backbone_engine and args.trt_head_engine:
        print(f"\n{'='*60}")
        print(f"TRT Engine Profiling")
        print(f"{'='*60}")
        
        (bb_engine, bb_ctx, bb_stream, bb_io_names, bb_out_shapes,
         head_engine, head_ctx, head_stream, head_io_names, head_in_shapes, head_out_shapes) = \
            load_trt_engines(args.trt_backbone_engine, args.trt_head_engine)
        
        trt_latencies = []
        trt_bb_latencies = []
        trt_head_latencies = []
        
        # Reload model for TRT patching
        model2 = load_model()
        
        def trt_feat_infer(inp_tensor):
            out_tensors = [torch.zeros(s, dtype=torch.float32, device='cuda:0') for s in bb_out_shapes]
            all_tensors = [inp_tensor] + out_tensors
            for i in range(bb_engine.num_io_tensors):
                bb_ctx.set_tensor_address(bb_io_names[i], int(all_tensors[i].data_ptr()))
            t_bb_start = torch.cuda.Event(enable_timing=True)
            t_bb_end = torch.cuda.Event(enable_timing=True)
            t_bb_start.record()
            bb_ctx.execute_async_v3(stream_handle=bb_stream.handle)
            bb_stream.synchronize()
            t_bb_end.record()
            torch.cuda.synchronize()
            bb_ms = t_bb_start.elapsed_time(t_bb_end)
            return out_tensors, bb_ms
        
        def trt_infer(engine, ctx, stream, inp_tensors, out_shapes):
            out_tensors = [torch.zeros(s, dtype=torch.float32, device='cuda:0') for _, s in out_shapes]
            all_io = list(inp_tensors) + out_tensors
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    engine_shape = tuple(engine.get_tensor_shape(name))
                    actual_shape = tuple(all_io[i].shape)
                    if -1 in engine_shape:
                        ctx.set_input_shape(name, actual_shape)
                ctx.set_tensor_address(name, int(all_io[i].data_ptr()))
            ctx.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()
            return out_tensors
        
        def _feat_trt_wrapper(img, img_metas, len_queue=None):
            B, N, C, H, W = img.shape
            img_flat = img.reshape(B * N, C, H, W).contiguous()
            trt_out, bb_ms = trt_feat_infer(img_flat)
            trt_bb_latencies.append(bb_ms)
            return [t.transpose(0, 1).contiguous() for t in trt_out]
        
        def _head_trt_wrapper(mlvl_feats, img_metas, prev_bev=None, only_bev=False,
                               ego_his_trajs=None, ego_lcf_feat=None):
            meta = img_metas[0]
            
            def to_cuda_f32(arr, shape):
                if isinstance(arr, np.ndarray):
                    return torch.from_numpy(arr.copy()).float().reshape(shape).cuda()
                elif isinstance(arr, torch.Tensor):
                    return arr.clone().float().reshape(shape).cuda()
                elif isinstance(arr, list):
                    arr = np.array(arr, dtype=np.float32)
                    return torch.from_numpy(arr).float().reshape(shape).cuda()
                else:
                    return torch.tensor(arr, dtype=torch.float32).reshape(shape).cuda()
            
            can_arr = meta['can_bus']
            if isinstance(can_arr, torch.Tensor):
                can_arr = can_arr.cpu().numpy()
            can_arr = np.asarray(can_arr, dtype=np.float32).flatten()
            
            delta_x = can_arr[0]; delta_y = can_arr[1]
            ego_angle = can_arr[-2] / np.pi * 180
            grid_length_y = 60.0 / 200; grid_length_x = 30.0 / 200
            translation_length = np.sqrt(delta_x**2 + delta_y**2)
            translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
            bev_angle = ego_angle - translation_angle
            shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / 200
            shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / 200
            shift = torch.tensor([[shift_x, shift_y]], dtype=torch.float32, device='cuda:0')
            
            can_bus = to_cuda_f32(can_arr, (1, 18))
            lidar2img = to_cuda_f32(meta['lidar2img'], (6, 4, 4))
            
            inp_tensors = [f.contiguous() for f in mlvl_feats] + [shift, can_bus, lidar2img]
            if prev_bev is not None:
                inp_tensors.append(prev_bev.contiguous())
            else:
                prev_shape = head_in_shapes[-1][1]
                if -1 in prev_shape:
                    prev_bev_zeros = torch.zeros(40000, 1, 256, dtype=torch.float32, device='cuda:0').contiguous()
                else:
                    prev_bev_zeros = torch.zeros(*prev_shape, dtype=torch.float32, device='cuda:0').contiguous()
                inp_tensors.append(prev_bev_zeros)
            
            t_head_start = torch.cuda.Event(enable_timing=True)
            t_head_end = torch.cuda.Event(enable_timing=True)
            t_head_start.record()
            out_tensors = trt_infer(head_engine, head_ctx, head_stream, inp_tensors, head_out_shapes)
            t_head_end.record()
            torch.cuda.synchronize()
            head_ms = t_head_start.elapsed_time(t_head_end)
            trt_head_latencies.append(head_ms)
            
            out_map = {}
            for (n, _), t in zip(head_out_shapes, out_tensors):
                out_map[n] = t
            name_map = {
                'out.bev_embed': 'bev_embed', 'out.all_cls_scores': 'all_cls_scores',
                'out.all_bbox_preds': 'all_bbox_preds', 'out.all_traj_preds': 'all_traj_preds',
                'out.all_traj_cls_scores': 'all_traj_cls_scores',
                'out.map_all_cls_scores': 'map_all_cls_scores',
                'out.map_all_bbox_preds': 'map_all_bbox_preds',
                'out.map_all_pts_preds': 'map_all_pts_preds',
                'out.ego_fut_preds': 'ego_fut_preds',
            }
            result = {}
            for trt_full, py_key in name_map.items():
                if trt_full in out_map:
                    result[py_key] = out_map[trt_full]
            result.update({
                'enc_cls_scores': None, 'enc_bbox_preds': None,
                'map_enc_cls_scores': None, 'map_enc_bbox_preds': None, 'map_enc_pts_preds': None,
            })
            return result
        
        model2.extract_img_feat = lambda img, img_metas, len_queue=None: _feat_trt_wrapper(img, img_metas, len_queue)
        model2.pts_bbox_head.forward = lambda mlvl_feats, img_metas, prev_bev=None, only_bev=False, ego_his_trajs=None, ego_lcf_feat=None: \
            _head_trt_wrapper(mlvl_feats, img_metas, prev_bev, only_bev, ego_his_trajs, ego_lcf_feat)
        model2.video_test_mode = True
        
        # TRT warmup
        print(f"TRT warmup {args.warmup} iterations...")
        wc = 0
        for batch_data in data_loader:
            if wc >= args.warmup:
                break
            cuda_batch = {}
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    cuda_batch[k] = v.cuda()
                elif isinstance(v, list):
                    cuda_batch[k] = [e.cuda() if isinstance(e, torch.Tensor) else e for e in v]
            with torch.no_grad():
                model2(cuda_batch, return_loss=False, rescale=True)
            wc += 1
        
        # TRT profiling
        trt_bb_latencies = []
        trt_head_latencies = []
        trt_total_latencies = []
        processed = 0
        for batch_data in data_loader:
            if processed >= args.num_samples:
                break
            cuda_batch = {}
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    cuda_batch[k] = v.cuda()
                elif isinstance(v, list):
                    cuda_batch[k] = [e.cuda() if isinstance(e, torch.Tensor) else e for e in v]
            torch.cuda.synchronize()
            with torch.no_grad():
                t0 = time.perf_counter()
                model2(cuda_batch, return_loss=False, rescale=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            trt_total_latencies.append((t1 - t0) * 1000)
            processed += 1
        
        avg_trt_total = np.mean(trt_total_latencies)
        avg_trt_bb = np.mean(trt_bb_latencies) if trt_bb_latencies else 0
        avg_trt_head = np.mean(trt_head_latencies) if trt_head_latencies else 0
        
        print(f"\n  TRT Profiling Results ({args.num_samples} samples):")
        print(f"  {'Module':<40s} {'Avg (ms)':<12s} {'% of E2E':<12s}")
        print(f"  {'-'*40} {'-'*12} {'-'*12}")
        print(f"  {'Backbone (TRT)':<40s} {avg_trt_bb:<12.1f} {avg_trt_bb/avg_trt_total*100:<11.1f}%")
        print(f"  {'Head (TRT)':<40s} {avg_trt_head:<12.1f} {avg_trt_head/avg_trt_total*100:<11.1f}%")
        print(f"  {'End-to-End (TRT)':<40s} {avg_trt_total:<12.1f} {'100.0%':<11s}")
        print(f"  {'Other (PyTorch overhead)':<40s} {avg_trt_total-avg_trt_bb-avg_trt_head:<12.1f} {(avg_trt_total-avg_trt_bb-avg_trt_head)/avg_trt_total*100:<11.1f}%")
    
    # ========== Orin Latency Estimation ==========
    # Known ratios from existing data:
    # x86 (RTX 4090): Backbone FP16 ~7ms, Head FP16 ~88ms, Total ~95ms
    # Orin: Backbone FP16 ~178ms, Head FP16 ~7799ms, Total ~7977ms
    # Ratio: Backbone ~25x, Head ~88x (due to transformer memory bandwidth bottleneck)
    
    print(f"\n{'='*60}")
    print(f"Orin Latency Estimation")
    print(f"{'='*60}")
    print(f"  Based on known ratios from existing benchmarks:")
    print(f"  x86 (RTX 4090) FP16: Backbone ~7ms, Head ~88ms, Total ~95ms")
    print(f"  Orin FP16: Backbone ~178ms, Head ~7799ms, Total ~7977ms")
    print(f"  Orin/x86 ratio: Backbone ~25x, Head ~89x")
    print()
    print(f"  Estimated Breakdown on Orin FP16 (based on x86 PyTorch profiling):")
    print(f"  {'Module':<40s} {'x86 (ms)':<12s} {'Orin Est. (ms)':<16s}")
    print(f"  {'-'*40} {'-'*12} {'-'*16}")
    
    # Use x86 PyTorch profiling to estimate
    for name, avg_ms in sorted_modules:
        if avg_ms < 0.5 or name in ["head_total", "encoder_total", "det_decoder_total", "map_decoder_total"]:
            continue
        # Head modules use ~89x ratio, backbone uses ~25x
        if "backbone" in name.lower():
            orin_est = avg_ms * 25
        else:
            orin_est = avg_ms * 89
        if orin_est > 1:
            print(f"  {name:<40s} {avg_ms:<12.1f} {orin_est:<16.1f}")
    
    print(f"\n  Orin Target: 100ms")
    print(f"  Current gap: ", end="")
    # From README: Orin INT8 OptLevel=2: ~1754ms
    print(f"17.5x (1754ms -> 100ms)")
    print(f"  VAD-Small estimated speedup: 8-16x (BEV 4x + layers 2x + FPN levels 4x)")
    print(f"  VAD-Small estimated Orin latency: ~110-220ms INT8")
    print(f"  Still need additional optimization to reach 100ms")
    
    # Save results
    output_path = args.output or os.path.join(os.path.dirname(__file__), "../results/vad_module_profile.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = {
        "config": {
            "num_samples": args.num_samples,
            "warmup": args.warmup,
            "detail": args.detail,
            "profile_trt": args.profile_trt,
        },
        "architecture": {
            "bev_h": head.bev_h,
            "bev_w": head.bev_w,
            "bev_queries": head.bev_h * head.bev_w,
            "encoder_layers": len(trans.encoder.layers),
            "det_decoder_layers": len(trans.decoder.layers) if trans.decoder else 0,
            "map_decoder_layers": len(trans.map_decoder.layers) if trans.map_decoder else 0,
            "fpn_levels": trans.num_feature_levels,
            "temporal_queue": 4,
        },
        "pytorch_profiling": {
            "avg_e2e_ms": float(avg_total),
            "first_frame_ms": float(first_frame_latency) if first_frame_latency else 0,
            "module_breakdown_ms": {k: float(v) for k, v in sorted_modules},
        },
        "orin_estimation": {
            "fp16_head_ms": 7799,  # from README
            "int8_opt0_ms": 2630,
            "int8_opt2_ms": 1754,
            "target_ms": 100,
            "gap_x": 17.5,
            "vad_small_estimated_ms": "110-220",
        },
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()