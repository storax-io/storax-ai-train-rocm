"""Probe the Windows ROCm PyTorch environment; print one JSON object.
Runs under the Windows venv python. Never raises — failures are reported
in the JSON so the WSL-side smoke test can render a diagnosis."""
import json
import platform
import sys

info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": None,
    "torch_error": None,
    "hip": None,
    "gpu_available": False,
    "device_name": None,
    "gcn_arch": None,
    "vram_total_gib": None,
    "bf16_supported": None,
    "triton": None,
    "triton_error": None,
    "matmul_bf16_ok": None,
    "sdpa_ok": None,
}

try:
    import torch
    info["torch"] = torch.__version__
    info["hip"] = getattr(torch.version, "hip", None)
    info["gpu_available"] = torch.cuda.is_available()
    if info["gpu_available"]:
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["gcn_arch"] = getattr(props, "gcnArchName", None)
        info["vram_total_gib"] = round(props.total_memory / 2**30, 2)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        try:
            a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
            c = a @ b
            torch.cuda.synchronize()
            info["matmul_bf16_ok"] = bool(torch.isfinite(c.float()).all().item())
        except Exception as e:  # noqa: BLE001
            info["matmul_bf16_ok"] = f"error: {e}"
        try:
            q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
            o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            info["sdpa_ok"] = bool(torch.isfinite(o.float()).all().item())
        except Exception as e:  # noqa: BLE001
            info["sdpa_ok"] = f"error: {e}"
except Exception as e:  # noqa: BLE001
    info["torch_error"] = repr(e)

try:
    import triton
    info["triton"] = triton.__version__
except Exception as e:  # noqa: BLE001
    info["triton_error"] = repr(e)

print(json.dumps(info))
