"""Generation report: per-config multi-seed statistics + family triage.

    python3 tools/gen_report.py /flash/.../runs/gen1 [/flash/.../runs/base-ref]

Reads each round dir's round.json + eval/eval.json + eval/guard.json,
groups seeds by config, applies the guard filter (hard, per plan), ranks
configs by multi-seed mean suite rate, and aggregates failing tasks by
template family across the whole generation — the strata input for the
next generation's manifest and generator feedback.
"""
import json
import statistics
import sys
from pathlib import Path


def load_round(d):
    try:
        meta = json.loads((d / "round.json").read_text())
        ev = json.loads((d / "eval" / "eval.json").read_text())
        gd = json.loads((d / "eval" / "guard.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"dir": d.name, "error": repr(e)}
    return {"dir": d.name, "config": meta.get("round"),
            "mix": meta.get("mix"), "drill": meta.get("drill_share"),
            "seed": meta.get("seed"), "rate": ev["rate"],
            "pass": ev["compile_pass"], "total": ev["total"],
            "guard": gd["rate"], "guard_ok": gd["rate"] >= 0.9,
            "fails": [r["id"] for r in ev["results"] if not r["ok"]],
            "truncated": sum(1 for r in ev["results"]
                             if r.get("truncated") and not r["ok"]),
            "repairs": sum(r.get("repair_rounds_used", 0)
                           for r in ev["results"] if r["ok"])}


def family(task_id):
    parts = task_id.split("-")
    return "-".join(parts[1:-1]) or task_id


def main():
    gen_dir = Path(sys.argv[1])
    rounds = [load_round(d) for d in sorted(gen_dir.iterdir()) if d.is_dir()
              and not d.name.endswith(("-salvage",))]
    ok = [r for r in rounds if "error" not in r]
    bad = [r for r in rounds if "error" in r]
    for r in bad:
        print(f"SKIP {r['dir']}: {r['error']}")

    by_cfg = {}
    for r in ok:
        by_cfg.setdefault(r["config"], []).append(r)

    print(f"\n{'config':8} {'mix':10} {'drill':5} {'n':2} "
          f"{'mean':6} {'min-max':11} {'sigma':5} {'guard-min':9} verdict")
    ranked = []
    for cfg, rs in by_cfg.items():
        rates = [r["rate"] for r in rs]
        guards = [r["guard"] for r in rs]
        mean = statistics.mean(rates)
        sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
        guard_fail = any(not r["guard_ok"] for r in rs)
        trunc = sum(r.get("truncated", 0) for r in rs)
        ranked.append((mean, cfg, rs[0], rates, sd, min(guards),
                       guard_fail, trunc))
    ranked.sort(reverse=True)
    for mean, cfg, r0, rates, sd, gmin, gfail, trunc in ranked:
        note = "GUARD-FAIL" if gfail else ("DEGENERATE" if trunc > 20 else "ok")
        print(f"{cfg:8} {r0['mix']:10} {r0['drill']:<5} {len(rates):<2} "
              f"{mean:.3f}  {min(rates):.3f}-{max(rates):.3f} "
              f"{sd:.3f} {gmin:9.3f} {note}  trunc={trunc}")

    fails = {}
    for r in ok:
        for t in r["fails"]:
            fails.setdefault(family(t), 0)
        for t in r["fails"]:
            fails[family(t)] += 1
    n_rounds = max(1, len(ok))
    print("\nfailures by family (avg per round, across generation):")
    for fam, n in sorted(fails.items(), key=lambda kv: -kv[1]):
        print(f"  {fam:24} {n / n_rounds:5.1f}")

    if len(sys.argv) > 2:
        ref = load_round(Path(sys.argv[2]))
        if "error" not in ref:
            print(f"\nbase reference: rate {ref['rate']:.3f} "
                  f"({ref['pass']}/{ref['total']}), guard {ref['guard']:.3f}")

    best = ranked[0]
    print(f"\nBEST: {best[1]} mean {best[0]:.3f} over {len(best[3])} seeds"
          f"{' (GUARD-FAIL — excluded from selection!)' if best[6] else ''}")


if __name__ == "__main__":
    main()
