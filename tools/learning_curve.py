"""The campaign's headline: numbers of learning, one row per generation.

    python3 tools/learning_curve.py ~/storax-runs/lumi
"""
import json
import statistics
import sys
from pathlib import Path


def load(d):
    try:
        e = json.loads((d / "eval/eval.json").read_text())
        g = json.loads((d / "eval/guard.json").read_text())
    except Exception:
        return None
    res = e["results"]
    return {"rate": e["rate"], "guard": g["rate"],
            "first": sum(1 for r in res if r["ok"] and not r.get("repair_rounds_used")) / max(1, e["total"]),
            "config": (json.loads((d / "round.json").read_text()).get("round")
                       if (d / "round.json").exists() else d.name)}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    print("%-9s %6s %6s %6s %6s %7s %6s %s" %
          ("stage", "best", "mean*", "1shot", "guard", "seeds", "mh", "note"))
    b = load(root / "base-ref2")
    if b:
        print("%-9s %6.3f %6s %6.3f %6.3f %7s %6s %s" %
              ("base", b["rate"], "-", b["first"], b["guard"], "-", "0",
               "untrained reference"))
    for gen_dir in sorted(root.glob("gen*")):
        runs = [r for r in (load(d) for d in sorted(gen_dir.iterdir())
                            if d.is_dir() and not d.name.endswith("-salvage"))
                if r]
        if not runs:
            continue
        by = {}
        for r in runs:
            by.setdefault(r["config"], []).append(r)
        ok_cfgs = {c: rs for c, rs in by.items()
                   if all(x["guard"] >= 0.9 for x in rs)}
        pool = ok_cfgs or by
        best_cfg = max(pool, key=lambda c: statistics.mean(x["rate"] for x in pool[c]))
        rs = pool[best_cfg]
        best_run = max(runs, key=lambda r: (r["guard"] >= 0.9, r["rate"]))
        guard_ok = sum(1 for r in runs if r["guard"] >= 0.9)
        mh = len(runs) * 9
        print("%-9s %6.3f %6.3f %6.3f %6.3f %4d/%-3d %6d %s" %
              (gen_dir.name, best_run["rate"],
               statistics.mean(x["rate"] for x in rs),
               statistics.mean(x["first"] for x in rs),
               best_run["guard"], guard_ok, len(runs), mh,
               "winner %s%s" % (best_cfg, "" if ok_cfgs else " (NO guard-clean cfg)")))
    print("\nmean* = best guard-clean config, multi-seed mean. mh ~= rounds*9 incl eval.")


if __name__ == "__main__":
    main()
