"""Cluster eval failures by normalized compiler-error signature.

    python3 tools/error_clusters.py ~/storax/runs/lumi

Reads every runs/**/eval/eval.json, groups failing tasks by template
family and by a normalized first_error (identifiers, numbers and paths
stripped), and prints ranked clusters with counts and sample tasks —
the generator work order, straight from the compiler's mouth.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def family(task_id):
    parts = task_id.split("-")
    return "-".join(parts[1:-1]) or task_id


def normalize(err):
    if not err:
        return "(no compiler error: run/timeout/truncation)"
    e = err
    e = re.sub(r"^[^:]*:\d+:\d+:\s*", "", e)          # file:line:col
    e = re.sub(r"'[^']*'", "'_'", e)                  # quoted identifiers
    e = re.sub(r"\d+", "N", e)
    return e.strip()[:160]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    files = sorted(root.glob("**/eval/eval.json"))
    print(f"{len(files)} eval files under {root}")
    clusters = defaultdict(Counter)
    samples = defaultdict(dict)
    totals = Counter()
    fails = Counter()
    for f in files:
        run = f.parent.parent.name
        for r in json.loads(f.read_text())["results"]:
            fam = family(r["id"])
            totals[fam] += 1
            if r["ok"]:
                continue
            fails[fam] += 1
            sig = ("TRUNCATED" if r.get("truncated")
                   else normalize(r.get("first_error", "")))
            clusters[fam][sig] += 1
            samples[fam].setdefault(sig, (run, r["id"],
                                          r.get("code_head", "")[:100]))
    for fam in sorted(fails, key=lambda k: -fails[k]):
        print(f"\n=== {fam}: {fails[fam]}/{totals[fam]} failing "
              f"({fails[fam] / max(1, totals[fam]):.0%}) ===")
        for sig, n in clusters[fam].most_common(8):
            run, tid, head = samples[fam][sig]
            print(f"  {n:4}x  {sig}")
            print(f"         e.g. {run}/{tid}  code: {head!r}")


if __name__ == "__main__":
    main()
