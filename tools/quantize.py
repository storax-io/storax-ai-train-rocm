"""Quantize a checkpoint for serving (Henri 2026-08-25: local tool,
quantize only — judging a quant is the eval battery's job, not this
tool's).

    python3 tools/quantize.py --model <hf-ckpt-dir> --out <dir> \
        [--quants q4_k_m,q5_k_m,q8_0]

llama.cpp is vendored under tools/llama.cpp on first run (pinned tag)
and built with the HIP backend when hipcc is present, CPU otherwise —
quantization is CPU-bound either way. Output: one GGUF per quant plus
quantize-report.json (sizes, sha256s) beside them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
LLAMA = TOOLS / "llama.cpp"
PIN = "b6200"   # known-good tag; bump deliberately


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def ensure_llama():
    if not LLAMA.exists():
        sh(["git", "clone", "--depth", "1", "--branch", PIN,
            "https://github.com/ggml-org/llama.cpp", str(LLAMA)])
    bin_dir = LLAMA / "build/bin"
    if not (bin_dir / "llama-quantize").exists():
        hip = shutil.which("hipcc") is not None
        args = ["cmake", "-S", str(LLAMA), "-B", str(LLAMA / "build"),
                "-DCMAKE_BUILD_TYPE=Release"]
        if hip:
            args += ["-DGGML_HIP=ON", "-DAMDGPU_TARGETS=gfx1101"]
        sh(args)
        sh(["cmake", "--build", str(LLAMA / "build"), "-j", "12",
            "--target", "llama-quantize"])
        print(f"llama.cpp built ({'HIP' if hip else 'CPU'} backend)")
    return bin_dir


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quants", default="q4_k_m,q5_k_m,q8_0")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bin_dir = ensure_llama()
    name = Path(a.model).name
    f16 = out / f"{name}-f16.gguf"
    if not f16.exists():
        sh([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"),
            a.model, "--outfile", str(f16), "--outtype", "f16"])
    report = {"model": a.model, "quants": {}}
    for q in (s.strip() for s in a.quants.split(",")):
        gguf = out / f"{name}-{q}.gguf"
        if not gguf.exists():
            sh([str(bin_dir / "llama-quantize"), str(f16), str(gguf),
                q.upper()])
        report["quants"][q] = {"bytes": gguf.stat().st_size,
                               "sha256": sha256(gguf)}
        print(q, report["quants"][q]["bytes"] // (1 << 20), "MiB")
    (out / "quantize-report.json").write_text(json.dumps(report, indent=1))
    print(f"report -> {out / 'quantize-report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
