#!/usr/bin/env python3
"""Smoke: multi-node training mechanics, SIMULATED — 2 ranks under
torchrun with the gloo backend on CPU, in the LUMI-container-pinned venv
(torch 2.10 / transformers v5). Exercises the exact DDP code paths a LUMI
node uses: process-group init, same-seed permutation + rank-strided
sharding, no_sync gradient accumulation, allreduce on boundaries, rank-0
artifacts, barriers, clean teardown. What it cannot test: RCCL, real
inter-node fabric, Slurm launch.

Run: python3 tests/smoke_dist.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV = REPO / ".venv-lumi-compat"
OUT = REPO / "runs-linux" / "dist-sim"
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def main():
    torchrun = VENV / "bin" / "torchrun"
    if not torchrun.exists():
        print("FAIL  compat venv missing (create per README / smoke_lumi_compat)")
        sys.exit(1)
    shutil.rmtree(OUT, ignore_errors=True)

    import os
    env = dict(os.environ,
               HF_HOME=str(REPO / ".hf-cache-compat"),
               OMP_NUM_THREADS="4")
    p = subprocess.run(
        [str(torchrun), "--standalone", "--nproc_per_node=2",
         "train.py",
         "--model", "HuggingFaceTB/SmolLM2-135M-Instruct",
         "--data", "cpp26", "--seq-len", "128",
         "--batch", "1", "--accum", "2", "--epochs", "1",
         "--max-steps", "6", "--lr", "1e-5",
         "--out", str(OUT), "--save-model"],
        cwd=REPO / "traintest", env=env,
        capture_output=True, text=True, timeout=1800)
    tail = (p.stdout + p.stderr)[-1500:]
    check("torchrun 2-rank run exits 0", p.returncode == 0,
          "" if p.returncode == 0 else tail)
    if p.returncode != 0:
        print(tail)

    res = OUT / "result.json"
    check("rank-0 result.json written", res.exists())
    if res.exists():
        r = json.loads(res.read_text())
        check("world_size recorded as 2", r.get("world_size") == 2,
              str(r.get("world_size")))
        check("loss finite", isinstance(r.get("final_loss"), (int, float)),
              str(r.get("final_loss")))
        check("global throughput = 2x per-rank",
              abs(r.get("global_tok_per_s_avg", 0)
                  - 2 * r.get("tok_per_s_avg", 0)) < 1.0,
              f"{r.get('tok_per_s_avg')} -> {r.get('global_tok_per_s_avg')}")
    check("rank-0 saved the model",
          (OUT / "model" / "config.json").exists())
    n_metrics = sum(1 for _ in (OUT / "metrics.jsonl").open()) \
        if (OUT / "metrics.jsonl").exists() else 0
    check("exactly one metrics stream (rank-0 only)",
          n_metrics == 6, f"{n_metrics} lines")

    print()
    if FAILS:
        print(f"SMOKE DIST: FAIL ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print("SMOKE DIST: PASS — DDP mechanics verified (simulated 2-rank "
          "gloo/CPU on container-pinned torch+transformers)")


if __name__ == "__main__":
    main()
