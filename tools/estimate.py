#!/usr/bin/env python3
"""Project measured training throughput to Instinct hardware.

Usage: python3 tools/estimate.py [path/to/result.json] [--models 3,7,30,70]

Method: take the measured MFU (achieved FLOPs / peak bf16 FLOPS) from a
real run on this machine, then for each (GPU, model size) compute
  tok/s = MFU * peak_tflops * 1e12 / (8 * params)
The 8*N FLOPs/token assumes training with activation recompute, matching
how the measurement was taken.

Caveats printed with the table: MFU on Instinct is usually HIGHER than on
consumer RDNA3 (CDNA matrix cores, HBM bandwidth, mature Linux ROCm with
flash-attention + hipBLASLt), so these are conservative floors. Multi-GPU
scaling adds communication overhead not modeled here.
"""
import argparse
import json
from pathlib import Path

# Vendor peak dense bf16 TFLOPS (no sparsity).
GPUS = {
    "RX 7800 XT (measured here)": 74.65,
    "MI250X module (LUMI-G)": 383.0,  # 2 GCDs; LUMI bills per module
    "MI300X": 1307.4,
    "MI355X": 2500.0,
}
DEFAULT_RESULT = "/mnt/c/Users/hs/storax-ai-train-test-win/runs/smoke/result.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", nargs="?", default=DEFAULT_RESULT)
    ap.add_argument("--models", default="3,7,30,70",
                    help="model sizes in B params")
    ap.add_argument("--tokens", type=float, default=1e9,
                    help="corpus size (tokens) for wall-clock estimate")
    ap.add_argument("--mfu", type=float, default=None,
                    help="override measured MFU (scenario analysis)")
    ap.add_argument("--gpu-hours", action="store_true",
                    help="print GPU-hours per 1B tokens instead of days")
    args = ap.parse_args()

    r = json.loads(Path(args.result).read_text())
    mfu = args.mfu if args.mfu is not None else r["mfu"]
    print(f"measured: {r['params_b']}B model, {r['tok_per_s_avg']} tok/s, "
          f"{r['achieved_tflops']} TFLOPS achieved, MFU={mfu:.1%} "
          f"on {r['device']}\n")

    sizes = [float(s) for s in args.models.split(",")]
    hdr = "model".ljust(8) + "".join(g.ljust(28) for g in GPUS)
    print(hdr)
    for nb in sizes:
        flops_tok = 8 * nb * 1e9
        row = f"{nb:g}B".ljust(8)
        for gpu, peak in GPUS.items():
            tok_s = mfu * peak * 1e12 / flops_tok
            if args.gpu_hours:
                gpuh = args.tokens / tok_s / 3600
                row += f"{tok_s:10,.0f} tok/s {gpuh:8,.0f} GPUh/Btok".ljust(28)
            else:
                days = args.tokens / tok_s / 86400
                row += f"{tok_s:10,.0f} tok/s {days:6.1f} d/Btok".ljust(28)
        print(row)
    print(f"\n(wall-clock column = days per {args.tokens:.0e} tokens, single GPU, "
          f"at measured MFU={mfu:.1%}; Instinct MFU will likely be 1.5-3x better)")


if __name__ == "__main__":
    main()
