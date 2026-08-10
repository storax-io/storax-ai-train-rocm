#!/usr/bin/env python3
"""Smoke: end-to-end full fine-tune of SmolLM3-3B on ROCm and knowledge
verification. Run directly from WSL: python3 tests/smoke_train.py [--quick]

Full mode (default, ~30-60 min on the 7800 XT):
  1. baseline eval  — model must NOT know the facts (train+control ~0)
  2. full fine-tune — bf16 + Adafactor + grad checkpointing
  3. post eval      — train facts learned, control still unknown,
                      retention (real-world QA) preserved
--quick: 8 training steps, no learning assertions — proves the pipeline
and memory recipe only.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import os

REPO = Path(__file__).resolve().parent.parent
STAGE = Path(os.environ.get("TRAINTEST_STAGE_WSL",
                            "/mnt/c/Users/hs/storax-ai-train-test-win"))
RUN_WIN = os.environ.get("TRAINTEST_STAGE_WIN",
                         r"C:\Users\hs\storax-ai-train-test-win")
FAILS = []


POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def kill_win_orphans():
    """Kill Windows-side venv pythons. A WSL-side timeout/kill never
    reaches across the interop boundary, so a timed-out run leaves a
    python.exe grinding on the GPU forever — which then silently slows
    every later run into paging (full3/full4 post-mortem)."""
    subprocess.run(
        [POWERSHELL, "-NoProfile", "-Command",
         "Get-Process python -ErrorAction SilentlyContinue | "
         "Where-Object {$_.Path -like '*storax-ai-train-test-win*'} | "
         "Stop-Process -Force"],
        capture_output=True, timeout=60)


def run(script, *args, timeout=7200):
    cmd = [str(REPO / "scripts" / "run_win.sh"), script, *args]
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_win_orphans()
        print(f"SMOKE TRAIN: FAIL — {script} timed out after {timeout}s "
              f"(windows-side process killed)")
        sys.exit(1)
    print(f"=== {script} exit={p.returncode} ({time.time()-t0:.0f}s)", flush=True)
    if p.returncode != 0:
        print(f"SMOKE TRAIN: FAIL — {script} exited {p.returncode}")
        sys.exit(1)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--run-name", default="smoke")
    ap.add_argument("--compile", action="store_true",
                    help="also exercise torch.compile/Inductor/Triton")
    args = ap.parse_args()

    run_dir = STAGE / "runs" / args.run_name
    run_dir_win = rf"{RUN_WIN}\runs\{args.run_name}"

    kill_win_orphans()  # stale GPU holders make every stage page & crawl

    # 1. Baseline. Facts are real people (Finnish presidents), so the base
    # model knows a few — we record the baseline and assert on the learned
    # delta afterwards, not on a zero baseline.
    run("evaluate.py", "--model", "HuggingFaceTB/SmolLM3-3B",
        "--out", rf"{run_dir_win}\eval_before.json", timeout=3600)
    before = json.loads((run_dir / "eval_before.json").read_text())["sets"]
    check("baseline train-facts mostly unknown",
          before["train"]["accuracy"] <= 0.5,
          f"acc={before['train']['accuracy']}")
    check("baseline control unknown", before["control"]["accuracy"] <= 0.2,
          f"acc={before['control']['accuracy']}")
    check("baseline retention sane", before["retention"]["accuracy"] >= 0.6,
          f"acc={before['retention']['accuracy']}")
    print(f"INFO  baseline paraphrase={before['paraphrase']['accuracy']} "
          f"composition={before['composition']['accuracy']} "
          f"multihop={before['multihop']['accuracy']} "
          f"adjacent={before['adjacent']['accuracy']}")
    run("evaluate.py", "--model", "HuggingFaceTB/SmolLM3-3B", "--think",
        "--sets", "multihop",
        "--out", rf"{run_dir_win}\eval_before_think.json", timeout=3600)
    bthink = json.loads(
        (run_dir / "eval_before_think.json").read_text())["sets"]
    print(f"INFO  baseline multihop THINK={bthink['multihop']['accuracy']}")

    # 2. Train (full fine-tune).
    train_args = ["--out", run_dir_win, "--save-model"]
    if args.quick:
        train_args += ["--max-steps", "8"]
    if args.compile:
        train_args += ["--compile"]
    run("train.py", *train_args, timeout=10800)
    result = json.loads((run_dir / "result.json").read_text())
    print(json.dumps(result, indent=2))
    check("peak VRAM within 16 GiB card", result["peak_vram_gib"] <= 15.5,
          f"{result['peak_vram_gib']} GiB")
    check("throughput recorded", result["tok_per_s_avg"] > 0,
          f"{result['tok_per_s_avg']} tok/s")

    # 3. Post-training verification.
    if not args.quick:
        run("evaluate.py", "--model", rf"{run_dir_win}\model",
            "--out", rf"{run_dir_win}\eval_after.json", timeout=3600)
        after = json.loads((run_dir / "eval_after.json").read_text())["sets"]
        check("facts learned", after["train"]["accuracy"] >= 0.75,
              f"acc={after['train']['accuracy']}")
        check("learned delta >= 0.25",
              after["train"]["accuracy"] - before["train"]["accuracy"] >= 0.25,
              f"{before['train']['accuracy']} -> {after['train']['accuracy']}")
        check("knowledge survives rephrasing (not format matching)",
              after["paraphrase"]["accuracy"] >= 0.6,
              f"{before['paraphrase']['accuracy']} -> {after['paraphrase']['accuracy']}")
        check("composition over stored facts (incl. reversed relations)",
              after["composition"]["accuracy"] >= 0.75,
              f"{before['composition']['accuracy']} -> {after['composition']['accuracy']}")
        check("adjacent knowledge not damaged",
              after["adjacent"]["accuracy"]
              >= before["adjacent"]["accuracy"] - 0.13,
              f"{before['adjacent']['accuracy']} -> {after['adjacent']['accuracy']}")
        check("control still unknown (no eval leak)",
              after["control"]["accuracy"] <= 0.2,
              f"acc={after['control']['accuracy']}")
        check("retention preserved", after["retention"]["accuracy"] >= 0.9,
              f"acc={after['retention']['accuracy']}")
        run("evaluate.py", "--model", rf"{run_dir_win}\model", "--think",
            "--sets", "multihop",
            "--out", rf"{run_dir_win}\eval_after_think.json", timeout=3600)
        athink = json.loads(
            (run_dir / "eval_after_think.json").read_text())["sets"]
        # Measured, not gated: first thinking-mode run. Gate once we know
        # what the mode is worth at 3B.
        print(f"INFO  multihop no-think {before['multihop']['accuracy']} -> "
              f"{after['multihop']['accuracy']}; "
              f"THINK {bthink['multihop']['accuracy']} -> "
              f"{athink['multihop']['accuracy']}")

    print()
    if FAILS:
        print(f"SMOKE TRAIN: FAIL ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print("SMOKE TRAIN: PASS")


if __name__ == "__main__":
    main()
