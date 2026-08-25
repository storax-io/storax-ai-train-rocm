"""Quantize a checkpoint for serving (Henri 2026-08-25: local tool,
quantize only — judging a quant is the eval battery's job, not this
tool's).

    python3 tools/quantize.py --model <hf-ckpt-dir> --out <dir> \
        [--quants q4_k_m,q5_k_m,q8_0] [--calib-pack <trainpack-dir>]
        [--no-imatrix] [--output-tensor-type q8_0]
        [--token-embedding-type q8_0] [--gfx gfx1101]

Best-of-breed path: an IMPORTANCE MATRIX is computed first
(llama-imatrix) and fed to every quantization — the difference between
a competent low-bit quant and a lobotomized one. Calibration text is
sampled from OUR OWN trainpack (--calib-pack): the deployment
distribution, not wikitext. --no-imatrix falls back to plain quanting.

llama.cpp is vendored under tools/llama.cpp on first run (pinned tag),
HIP backend when hipcc is present, CPU otherwise (imatrix on CPU for a
14B is slow — hours; fine overnight, fast once hipcc lands). Output:
one GGUF per quant + imatrix.dat + quantize-report.json (sizes,
sha256s, calibration provenance).
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
PIN = "b10622"   # verified live tag 2026-08-25; bump deliberately


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def _real_vulkan_device() -> bool:
    """llvmpipe-only Vulkan (stock Ubuntu on WSL: no dozen ICD) is
    SLOWER than the native CPU backend — only count real devices."""
    try:
        out = subprocess.run(["vulkaninfo", "--summary"],
                             capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return False
    devs = [l for l in out.splitlines() if "deviceName" in l]
    return any("llvmpipe" not in d for d in devs) if devs else False


def ensure_llama(gfx="gfx1101"):
    if not LLAMA.exists():
        sh(["git", "clone", "--depth", "1", "--branch", PIN,
            "https://github.com/ggml-org/llama.cpp", str(LLAMA)])
    bin_dir = LLAMA / "build/bin"
    if not (bin_dir / "llama-quantize").exists():
        # backend ladder: HIP (hipcc) > Vulkan (in-distro, works on WSL
        # via Mesa d3d12 — Ubuntu 26.04 predates AMD's apt packaging) > CPU
        args = ["cmake", "-S", str(LLAMA), "-B", str(LLAMA / "build"),
                "-DCMAKE_BUILD_TYPE=Release"]
        if shutil.which("hipcc"):
            backend = "HIP"
            args += ["-DGGML_HIP=ON", f"-DAMDGPU_TARGETS={gfx}"]
        elif shutil.which("glslc") and _real_vulkan_device():
            backend = "Vulkan"
            args += ["-DGGML_VULKAN=ON"]
        else:
            backend = "CPU"
        sh(args)
        sh(["cmake", "--build", str(LLAMA / "build"), "-j", "12",
            "--target", "llama-quantize", "llama-imatrix"])
        print(f"llama.cpp built ({backend} backend)")
    return bin_dir


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build_calib(pack: Path, out_txt: Path, n_per_stream=40,
                max_chars=4000):
    """Calibration text from the trainpack itself: a few dozen records
    per stream, truncated — the serve-time distribution in miniature."""
    import random
    rng = random.Random(0)
    chunks = []
    for f in sorted(pack.glob("*.jsonl")):
        rows = []
        for ln in f.open():
            if ln.strip():
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                s = str(r.get("source") or r.get("edited") or "")
                if len(s) > 200:
                    rows.append(s)
        for s in rng.sample(rows, min(n_per_stream, len(rows))):
            chunks.append(s[:max_chars])
    rng.shuffle(chunks)
    out_txt.write_text("\n\n".join(chunks))
    return len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quants", default="q4_k_m,q5_k_m,q8_0")
    ap.add_argument("--calib-pack", default=None,
                    help="trainpack dir to sample calibration text from")
    ap.add_argument("--calib-file", default=None,
                    help="ready-made calibration text file")
    ap.add_argument("--no-imatrix", action="store_true")
    ap.add_argument("--output-tensor-type", default=None,
                    help="keep the output tensor at this type (e.g. q8_0/f16)")
    ap.add_argument("--token-embedding-type", default=None,
                    help="keep token embeddings at this type (e.g. q8_0/f16)")
    ap.add_argument("--gfx", default="gfx1101",
                    help="AMDGPU target for the HIP build")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bin_dir = ensure_llama(a.gfx)
    name = Path(a.model).name
    f16 = out / f"{name}-f16.gguf"
    if not f16.exists():
        sh([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"),
            a.model, "--outfile", str(f16), "--outtype", "f16"])
    report = {"model": a.model, "quants": {}}
    imatrix = None
    if not a.no_imatrix:
        calib = Path(a.calib_file) if a.calib_file else out / "calib.txt"
        if a.calib_file is None:
            if not a.calib_pack:
                sys.exit("imatrix needs --calib-pack or --calib-file "
                         "(or pass --no-imatrix)")
            n = build_calib(Path(a.calib_pack), calib)
            print(f"calibration: {n} chunks from {a.calib_pack}")
        imatrix = out / "imatrix.dat"
        if not imatrix.exists():
            sh([str(bin_dir / "llama-imatrix"), "-m", str(f16),
                "-f", str(calib), "-o", str(imatrix)])
        report["imatrix"] = {"calib": str(calib),
                             "sha256": sha256(imatrix)}
    for q in (s.strip() for s in a.quants.split(",")):
        gguf = out / f"{name}-{q}.gguf"
        if not gguf.exists():
            cmd = [str(bin_dir / "llama-quantize")]
            if imatrix:
                cmd += ["--imatrix", str(imatrix)]
            if a.output_tensor_type:
                cmd += ["--output-tensor-type", a.output_tensor_type]
            if a.token_embedding_type:
                cmd += ["--token-embedding-type", a.token_embedding_type]
            cmd += [str(f16), str(gguf), q.upper()]
            sh(cmd)
        report["quants"][q] = {"bytes": gguf.stat().st_size,
                               "sha256": sha256(gguf)}
        print(q, report["quants"][q]["bytes"] // (1 << 20), "MiB")
    (out / "quantize-report.json").write_text(json.dumps(report, indent=1))
    print(f"report -> {out / 'quantize-report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
