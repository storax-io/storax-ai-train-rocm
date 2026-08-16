"""The eval as 128 users: what did each one feel?

Every eval task is one user's transaction: they wait for every generated
token (and again per repair round). This tool converts an eval.json into
the user-experience distribution — wait percentiles at a given local
decode speed, how many needed repairs, how many were never served.

  python3 tools/user_pain.py RUN_DIR... [--tok-s 30]

Needs evals recorded with gen_tokens (2026-08-16+).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path


def pain(eval_json: Path, tok_s: float):
    d = json.loads(eval_json.read_text())
    waits, repaired, unserved, no_tok = [], 0, 0, 0
    for r in d["results"]:
        t = r.get("gen_tokens")
        if t is None:
            no_tok += 1
            continue
        waits.append(t / tok_s)
        if r.get("repair_rounds_used", 0) > 0:
            repaired += 1
        if not r.get("ok"):
            unserved += 1
    return d, waits, repaired, unserved, no_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs containing eval/eval.json")
    ap.add_argument("--tok-s", type=float, default=30.0,
                    help="local decode speed per user (tok/s), default 30")
    args = ap.parse_args()
    print(f"{'run':24} {'p50':>6} {'p95':>6} {'max':>6}  repaired unserved  (waits in s @ %g tok/s)"
          % args.tok_s)
    for rd in args.runs:
        ej = Path(rd) / "eval/eval.json"
        if not ej.exists():
            print(f"{Path(rd).name:24} — no eval.json")
            continue
        d, waits, repaired, unserved, no_tok = pain(ej, args.tok_s)
        if not waits:
            print(f"{Path(rd).name:24} — eval predates gen_tokens recording"
                  f" ({no_tok} tasks without token counts)")
            continue
        waits.sort()
        p = lambda q: waits[min(len(waits) - 1, int(q * len(waits)))]
        n = len(waits)
        print(f"{Path(rd).name:24} {p(.5):6.0f} {p(.95):6.0f} {waits[-1]:6.0f}"
              f"  {repaired:4d}/{n:<4d} {unserved:4d}/{n:<4d}")
    print("\nunserved = compile/run still failing after repairs: that user "
          "got nothing, however long they waited.")


if __name__ == "__main__":
    main()
