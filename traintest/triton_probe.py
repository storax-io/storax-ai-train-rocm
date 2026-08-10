"""Triton-on-ROCm probe: compile and run real Triton kernels on the GPU and
check numerics against eager torch. This is the load-bearing test for the
Instinct pathfinding goal — if Triton can't compile for this gfx target,
torch.compile/Inductor can't either. Prints one JSON object."""
import json
import time

result = {
    "triton_version": None,
    "vector_add_ok": None,
    "softmax_ok": None,
    "softmax_speedup_vs_eager": None,
    "compile_time_s": None,
    "error": None,
}


def main():
    import torch
    import triton
    import triton.language as tl

    result["triton_version"] = triton.__version__

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    @triton.jit
    def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        out = num / tl.sum(num, axis=0)
        tl.store(out_ptr + row * n_cols + offs, out, mask=mask)

    n = 1 << 20
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    t0 = time.perf_counter()
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    result["compile_time_s"] = round(time.perf_counter() - t0, 2)
    result["vector_add_ok"] = bool(torch.allclose(out, x + y))

    rows, cols = 4096, 1024
    m = torch.randn(rows, cols, device="cuda")
    sm = torch.empty_like(m)
    softmax_kernel[(rows,)](m, sm, cols, BLOCK=1024)
    torch.cuda.synchronize()
    result["softmax_ok"] = bool(
        torch.allclose(sm, torch.softmax(m, dim=1), atol=1e-5)
    )

    # Rough perf sanity: warmed-up triton softmax vs eager torch softmax.
    for _ in range(3):
        softmax_kernel[(rows,)](m, sm, cols, BLOCK=1024)
        torch.softmax(m, dim=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        softmax_kernel[(rows,)](m, sm, cols, BLOCK=1024)
    torch.cuda.synchronize()
    t_triton = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(50):
        torch.softmax(m, dim=1)
    torch.cuda.synchronize()
    t_eager = time.perf_counter() - t0
    result["softmax_speedup_vs_eager"] = round(t_eager / t_triton, 2)


try:
    main()
except Exception as e:  # noqa: BLE001
    result["error"] = repr(e)

print(json.dumps(result))
