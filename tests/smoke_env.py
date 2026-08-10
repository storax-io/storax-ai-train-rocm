#!/usr/bin/env python3
"""Smoke: Windows ROCm environment health — torch sees the 7800 XT, bf16
math works, SDPA works, and Triton compiles + runs real kernels on gfx11.
Run directly from WSL: python3 tests/smoke_env.py  (no venv needed here;
the GPU work happens in the Windows venv via scripts/run_win.sh)."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAILS = []


def run_probe(script):
    p = subprocess.run([str(REPO / "scripts" / "run_win.sh"), script],
                       capture_output=True, text=True, timeout=600)
    line = next((l for l in reversed(p.stdout.splitlines())
                 if l.startswith("{")), None)
    if line is None:
        print(f"--- {script} produced no JSON ---")
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        sys.exit(1)
    return json.loads(line)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


env = run_probe("env_probe.py")
print(json.dumps(env, indent=2))
check("torch imports", env["torch"] is not None, env.get("torch_error") or env["torch"])
check("GPU visible", env["gpu_available"] is True, env.get("device_name"))
check("gfx11 target", bool(env.get("gcn_arch")) and "gfx11" in env["gcn_arch"],
      env.get("gcn_arch"))
check("VRAM >= 15 GiB", (env.get("vram_total_gib") or 0) >= 15,
      f"{env.get('vram_total_gib')} GiB")
check("bf16 supported", env.get("bf16_supported") is True)
check("bf16 matmul finite", env.get("matmul_bf16_ok") is True,
      str(env.get("matmul_bf16_ok")))
check("SDPA runs", env.get("sdpa_ok") is True, str(env.get("sdpa_ok")))
check("triton importable", env.get("triton") is not None,
      env.get("triton_error") or env.get("triton"))

if env.get("gpu_available") and env.get("triton"):
    tr = run_probe("triton_probe.py")
    print(json.dumps(tr, indent=2))
    check("triton vector-add kernel", tr.get("vector_add_ok") is True,
          tr.get("error") or f"compile {tr.get('compile_time_s')}s")
    check("triton softmax kernel numerics", tr.get("softmax_ok") is True,
          f"speedup vs eager: {tr.get('softmax_speedup_vs_eager')}x")

print()
if FAILS:
    print(f"SMOKE ENV: FAIL ({len(FAILS)}): {', '.join(FAILS)}")
    sys.exit(1)
print("SMOKE ENV: PASS")
