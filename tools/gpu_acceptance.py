"""New/used GPU acceptance test: VRAM integrity + compute sanity.

Fill (nearly) all VRAM with patterned tensors, read back and verify —
catches degraded memory; then a sustained bf16 matmul burst for thermal
sanity. Run inside the card's venv:  python3 tools/gpu_acceptance.py

Exit 0 = keep the card.
"""
import sys
import time

import torch

assert torch.cuda.is_available(), "no GPU visible"
props = torch.cuda.get_device_properties(0)
total_gib = props.total_memory / 2**30
print(f"device: {props.name}  arch: {getattr(props, 'gcnArchName', '?')}  "
      f"vram: {total_gib:.1f} GiB")

# --- VRAM integrity: pattern-fill ~90% in 1 GiB chunks, verify each
chunk_elems = 2**28  # 1 GiB of fp32
n_chunks = int(total_gib * 0.90)
chunks, bad = [], 0
for i in range(n_chunks):
    t = torch.full((chunk_elems,), float(i % 251), device="cuda",
                   dtype=torch.float32)
    chunks.append(t)
torch.cuda.synchronize()
for i, t in enumerate(chunks):
    if not bool((t == float(i % 251)).all()):
        bad += 1
        print(f"CHUNK {i}: PATTERN MISMATCH")
del chunks
torch.cuda.empty_cache()
print(f"vram integrity: {n_chunks - bad}/{n_chunks} chunks clean")

# --- sustained compute: 60 s of bf16 matmul, watch for NaN/slowdown
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
t0 = time.perf_counter()
iters = 0
rates = []
while time.perf_counter() - t0 < 60:
    s = time.perf_counter()
    for _ in range(50):
        a = a @ a
        a = a / a.norm() * 4096  # keep values bounded
    torch.cuda.synchronize()
    rates.append(50 / (time.perf_counter() - s))
    iters += 50
    if not bool(torch.isfinite(a).all()):
        print("NON-FINITE VALUES UNDER LOAD")
        sys.exit(1)
tflops = rates[-1] * 2 * 4096**3 * 1 / 1e12
drift = (rates[0] - rates[-1]) / rates[0] * 100
print(f"sustained bf16 matmul: {tflops:.1f} TFLOPS effective, "
      f"thermal drift {drift:+.1f}% over 60s ({iters} iters)")

ok = bad == 0 and drift < 15
print("ACCEPTANCE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
