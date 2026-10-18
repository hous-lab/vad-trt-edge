#!/usr/bin/env python3
"""Power-mode latency benchmark for the Orin deployment engines.

Per nvpmodel mode: load each engine once, 3 warm-up + N timed runs on fixed
inputs; per-run latencies + p50/p90. tegrastats (100 ms) samples clocks and
power during the timed window. One JSON + one tegrastats log per mode.
"""
import argparse, json, os, re, subprocess, sys, time
import numpy as np
import torch
import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_orin_lite import load_engine, trt_run

MODE_NAMES = {0: "MAXN", 1: "MODE_15W", 2: "MODE_30W", 3: "MODE_50W"}

def verify_mode(mode_id):
    """Assert the requested nvpmodel mode took effect (exit on mismatch)."""
    out = subprocess.check_output(["nvpmodel", "-q"], text=True)
    name = out.splitlines()[0].replace("NV Power Mode:", "").strip()
    expected = MODE_NAMES.get(mode_id)
    if name != expected:
        sys.exit(f"FATAL: requested mode {mode_id} ({expected}) but device reports "
                 f"'{name}' - switch did not apply (reboot-required mode?), aborting.")
    return name


def start_tegrastats(path):
    log = open(path, "w")
    proc = subprocess.Popen(["sudo", "-n", "tegrastats", "--interval", "100"],
                            stdout=log, stderr=subprocess.DEVNULL)
    time.sleep(1.0)  # let the first samples arrive
    return proc, log


def stop_tegrastats(proc, log):
    proc.terminate()
    log.close()

def stats(xs, scale=1.0):
    if not xs:
        return None
    return {"mean": round(float(np.mean(xs)) * scale, 1),
            "p95": round(float(np.percentile(xs, 95)) * scale, 1),
            "min": round(min(xs) * scale, 1),
            "max": round(max(xs) * scale, 1)}


def parse_tegrastats(path):
    """Extract GPU clock / power statistics from a tegrastats log."""
    gpu, emc, vin, vdd = [], [], [], []
    pat_gpu = re.compile(r"GR3D_FREQ \d+%@\[([\d, ]+)\]")
    pat_emc = re.compile(r"EMC_FREQ \d+%@(\d+)")
    pat_vin = re.compile(r"VIN_SYS_5V0 (\d+)mW")
    pat_vdd = re.compile(r"VDD_GPU_SOC (\d+)mW")
    with open(path) as f:
        for line in f:
            m = pat_gpu.search(line)
            if m:
                gpu.append(max(int(x) for x in m.group(1).split(",")))
            m = pat_emc.search(line)
            if m:
                emc.append(int(m.group(1)))
            m = pat_vin.search(line)
            if m:
                vin.append(int(m.group(1)))
            m = pat_vdd.search(line)
            if m:
                vdd.append(int(m.group(1)))
    return {
        "gpu_clock_mhz": stats(gpu), "emc_clock_mhz": stats(emc),
        "power_vin_w": stats(vin, 1e-3), "power_gpu_soc_rail_w": stats(vdd, 1e-3),
    }

def bench_engine(engine, ctx, stream, make_inputs, reps):
    """3 warm-up + reps timed runs; returns per-run latencies (ms)."""
    for _ in range(3):
        trt_run(engine, ctx, stream, make_inputs())
    torch.cuda.synchronize()
    lats = []
    for _ in range(reps):
        t0 = time.perf_counter()
        trt_run(engine, ctx, stream, make_inputs())
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000.0)
    return lats


def pct(xs):
    return {"p50": round(float(np.percentile(xs, 50)), 1),
            "p90": round(float(np.percentile(xs, 90)), 1),
            "mean": round(float(np.mean(xs)), 1),
            "runs": [round(x, 1) for x in xs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-engine", required=True)
    ap.add_argument("--head-engine", required=True)
    ap.add_argument("--plugin-lib", required=True)
    ap.add_argument("--calib-npz", required=True)
    ap.add_argument("--random-head-inputs", action="store_true",
                    help="skip the npz; random tensors at the engine's input shapes "
                         "(valid for latency-only benching: static engines have data-independent latency)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--mode-id", type=int, required=True)
    ap.add_argument("--mode-label", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    name = verify_mode(args.mode_id)
    print(f"[bench] mode verified: {name} (id {args.mode_id})")
    print(f"[bench] TRT {trt.__version__}, torch {torch.__version__}, reps={args.reps}")

    bb_engine = load_engine(args.backbone_engine, args.plugin_lib)
    bb_ctx = bb_engine.create_execution_context()
    head_engine = load_engine(args.head_engine, args.plugin_lib)
    head_ctx = head_engine.create_execution_context()
    stream = torch.cuda.Stream()

    # Backbone input: random fixed tensor at the engine's input shape.
    bb_in = None
    for i in range(bb_engine.num_io_tensors):
        n = bb_engine.get_tensor_name(i)
        if bb_engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            shape = tuple(d if d != -1 else 1 for d in bb_engine.get_tensor_shape(n))
            bb_in = {n: torch.randn(shape, dtype=torch.float32, device="cuda:0")}
            print(f"[bench] backbone input {n} {shape} (random fixed tensor)")
            break

    # Head inputs: real frame_000 of the calib npz by default; or random tensors
    # at the engine's input shapes with --random-head-inputs (valid for
    # latency-only benching: static engines have data-independent latency).
    if args.random_head_inputs:
        head_in = {}
        for i in range(head_engine.num_io_tensors):
            n = head_engine.get_tensor_name(i)
            if head_engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                shape = tuple(d if d != -1 else 1 for d in head_engine.get_tensor_shape(n))
                head_in[n] = torch.randn(shape, dtype=torch.float32, device="cuda:0")
        print(f"[bench] head inputs: random at engine shapes, {len(head_in)} tensors")
    else:
        calib = np.load(args.calib_npz, allow_pickle=True, mmap_mode="r")
        head_in_names = [head_engine.get_tensor_name(i) for i in range(head_engine.num_io_tensors)
                         if head_engine.get_tensor_mode(head_engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        head_in = {}
        for n in head_in_names:
            arr = np.ascontiguousarray(calib[f"frame_000__{n}"], dtype=np.float32)
            head_in[n] = torch.from_numpy(arr).float().cuda()
        print(f"[bench] head inputs loaded: {list(head_in.keys())}")

    def bb_inputs():
        return bb_in

    def head_inputs():
        return head_in

    # Warm the backbone once outside the tegrastats window (context init).
    trt_run(bb_engine, bb_ctx, stream, bb_inputs())
    torch.cuda.synchronize()

    # ---- timed section: tegrastats samples this window only ----
    ts_log = os.path.splitext(args.output)[0] + ".tegrastats.log"
    proc, log = start_tegrastats(ts_log)
    bb_lats = bench_engine(bb_engine, bb_ctx, stream, bb_inputs, args.reps)
    head_lats = bench_engine(head_engine, head_ctx, stream, head_inputs, args.reps)
    stop_tegrastats(proc, log)

    bb, hd = pct(bb_lats), pct(head_lats)
    results = {
        "mode_id": args.mode_id, "mode_name": name,
        "trt_version": trt.__version__, "torch_version": torch.__version__,
        "backbone_engine": os.path.basename(args.backbone_engine),
        "head_engine": os.path.basename(args.head_engine),
        "reps": args.reps, "warmup": 3,
        "workload": "backbone: fixed random input; head: "
                    + ("random tensors at engine shapes" if args.random_head_inputs
                       else "frame_000 of " + os.path.basename(args.calib_npz)),
        "backbone": bb, "head": hd,
        "e2e": {"p50": round(bb["p50"] + hd["p50"], 1),
                "p90": round(bb["p90"] + hd["p90"], 1)},
        "tegrastats": parse_tegrastats(ts_log),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"[bench] saved {args.output}")


if __name__ == "__main__":
    main()
