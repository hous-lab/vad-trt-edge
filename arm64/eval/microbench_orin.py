#!/usr/bin/env python3
"""Peak-throughput microbenchmarks anchoring the roofline endpoints (Orin GPU).

- DRAM: large FP16 copy (2 x 1 GB buffers -> 2 GB traffic per rep).
- FP16 dense GEMM: 8192^3, TFLOPS.
- INT8 dense GEMM: 8192^3 via torch._int_mm, TOPS.
- FP16 kernel-loop: small kernel launches back-to-back (launch/latency anchor).
"""
import sys
import torch

def bench(fn, reps=10, warmup=3):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out

def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}")

    # --- DRAM bandwidth: FP16 copy, 2 x 1 GB buffers (>> L2 4 MB) ---
    n = 512 * 1024 * 1024  # 1 GiB of fp16
    x = torch.empty(n, dtype=torch.float16, device="cuda")
    y = torch.empty(n, dtype=torch.float16, device="cuda")
    t = min(bench(lambda: y.copy_(x)))
    bw = 2 * n * 2 / (t * 1e-3) / 1e9
    print(f"DRAM copy (fp16, 1GiB x2): {t:.2f} ms  -> {bw:.0f} GB/s")

    # --- FP16 dense GEMM peak ---
    a = torch.randn(8192, 8192, dtype=torch.float16, device="cuda")
    b = torch.randn(8192, 8192, dtype=torch.float16, device="cuda")
    t = min(bench(lambda: a @ b))
    print(f"FP16 GEMM 8192^3: {t:.2f} ms -> {2 * 8192**3 / (t * 1e-3) / 1e12:.1f} TFLOPS")
    del a, b
    torch.cuda.empty_cache()

    # --- INT8 dense GEMM peak ---
    ai = torch.randint(-127, 127, (8192, 8192), dtype=torch.int8, device="cuda")
    bi = torch.randint(-127, 127, (8192, 8192), dtype=torch.int8, device="cuda")
    t = min(bench(lambda: torch._int_mm(ai, bi)))
    print(f"INT8 GEMM 8192^3: {t:.2f} ms -> {2 * 8192**3 / (t * 1e-3) / 1e12:.1f} TOPS")
    del ai, bi
    torch.cuda.empty_cache()

    # --- launch/latency anchor: many tiny kernel launches back-to-back ---
    s_small = torch.empty(1024, dtype=torch.float16, device="cuda")
    d_small = torch.empty(1024, dtype=torch.float16, device="cuda")
    def tiny():
        for _ in range(1000):
            d_small.add_(1.0)
    t = min(bench(tiny))
    print(f"1000 tiny kernels: {t:.2f} ms -> {t:.3f} us/kernel (launch+execute)")
    print("microbench done")
if __name__ == "__main__":
    main()
