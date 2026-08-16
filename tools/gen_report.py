"""Generation report v2 — prints the decision contract, not raw data.

    python3 tools/gen_report.py <gen-dir> [base-ref-dir] [prev-gen-dir]

Per config: seeds, suite mean/min-max/sigma, guard-min, first-shot vs
repaired rate, truncations, verdict. Then the gate answers (vs base, vs
previous gen, guard stability, sigma trend), family deltas for the
winner, best single checkpoint, provenance, and a machine verdict.
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def family(task_id):
    parts = task_id.split("-")
    return "-".join(parts[1:-1]) or task_id


def load_run(d):
    partial = False
    try:
        ev = json.loads((d / "eval" / "eval.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # salvage fallback: merge whatever shards survived (checkpoint may
        # be pruned — this partial is all this run will ever say)
        shards = sorted((d / "eval-salvage").glob("shard*.json"))
        if not shards:
            return None
        merged = {}
        for sh in shards:
            for r in json.loads(sh.read_text())["results"]:
                merged[r["id"]] = r
        res = list(merged.values())
        ev = {"rate": round(sum(r["ok"] for r in res) / max(1, len(res)), 3),
              "compile_pass": sum(r["ok"] for r in res),
              "total": len(res), "results": res}
        partial = True
    try:
        gd = json.loads((d / "eval" / "guard.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        gd = {"rate": None}
    meta = {}
    rj = d / "round.json"
    if rj.exists():
        meta = json.loads(rj.read_text())
    res = ev["results"]
    return {"dir": d.name, "partial": partial,
            "config": meta.get("round", d.name),
            "meta": meta, "rate": ev["rate"],
            "first_shot": sum(1 for r in res if r["ok"]
                              and not r.get("repair_rounds_used")) / max(1, ev["total"]),
            "guard": gd["rate"] if gd["rate"] is not None else -1.0,
            "trunc": sum(1 for r in res if r.get("truncated") and not r["ok"]),
            "fails_by_family": Counter(family(r["id"]) for r in res if not r["ok"]),
            "ok_ids": {r["id"] for r in res if r["ok"]}}


def load_gen(gen_dir):
    runs, gaps = [], []
    for d in sorted(Path(gen_dir).iterdir()):
        if not d.is_dir() or d.name.endswith("-salvage"):
            continue
        r = load_run(d)
        (runs if r else gaps).append(r or d.name)
    return runs, gaps


def config_rows(runs):
    by = {}
    for r in runs:
        by.setdefault(r["config"], []).append(r)
    rows = []
    for cfg, rs in by.items():
        rates = [r["rate"] for r in rs]
        rows.append({
            "config": cfg, "n": len(rs),
            "mix": rs[0]["meta"].get("mix", "?"),
            "drill": rs[0]["meta"].get("drill_share", "?"),
            "mean": statistics.mean(rates),
            "min": min(rates), "max": max(rates),
            "sigma": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "guard_min": min((r["guard"] for r in rs if r["guard"] >= 0), default=-1),
            "guard_pass": sum(1 for r in rs if r["guard"] >= 0.9),
            "guarded": sum(1 for r in rs if r["guard"] >= 0),
            "partials": sum(1 for r in rs if r.get("partial")),
            "first_shot": statistics.mean(r["first_shot"] for r in rs),
            "trunc": sum(r["trunc"] for r in rs),
            "runs": rs})
    rows.sort(key=lambda r: -r["mean"])
    return rows


def verdict_of(row):
    if row["guard_pass"] < row.get("guarded", row["n"]):
        return "GUARD-FAIL"
    if row["trunc"] > 20 * row["n"]:
        return "DEGENERATE"
    return "ok"


def main():
    gen_dir = sys.argv[1]
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    prev = sys.argv[3] if len(sys.argv) > 3 else None

    runs, gaps = load_gen(gen_dir)
    rows = config_rows(runs)

    print(f"{'config':8} {'mix':10} {'drill':5} {'n':2} {'mean':6} "
          f"{'min-max':12} {'sigma':6} {'grd-min':7} {'grd-n':5} "
          f"{'1shot':6} {'trunc':5} verdict")
    for r in rows:
        print(f"{r['config']:8} {r['mix']:10} {str(r['drill']):5} {r['n']:<2} "
              f"{r['mean']:.3f}  {r['min']:.3f}-{r['max']:.3f} {r['sigma']:.3f}  "
              f"{r['guard_min']:.3f}   {r['guard_pass']}/{r['guarded']:<3} "
              f"{r['first_shot']:.3f}  {r['trunc']:<5} {verdict_of(r)}"
              + (f" [{r['partials']} partial]" if r.get("partials") else ""))

    ok_rows = [r for r in rows if verdict_of(r) == "ok"]
    winner = ok_rows[0] if ok_rows else None
    best_run = max(runs, key=lambda r: (r["guard"] >= 0.9, r["rate"]),
                   default=None)

    base_rate = None
    if base and (base / "eval" / "eval.json").exists():
        base_rate = json.loads((base / "eval" / "eval.json").read_text())["rate"]

    prev_best = prev_sigma = None
    if prev:
        pruns, _ = load_gen(prev)
        prows = config_rows(pruns)
        pok = [r for r in prows if verdict_of(r) == "ok"]
        if pok:
            prev_best = pok[0]["mean"]
            prev_sigma = statistics.mean(r["sigma"] for r in prows)

    print("\n== gates ==")
    if winner and base_rate is not None:
        print(f"vs base:      winner {winner['mean']:.3f} vs base {base_rate:.3f} "
              f"(delta {winner['mean'] - base_rate:+.3f})")
    if winner and prev_best is not None:
        d = winner["mean"] - prev_best
        print(f"vs prev gen:  {d:+.3f} ({'MEETS' if d >= 0.02 else 'BELOW'} the +0.02 rule)")
    if prev_sigma is not None and rows:
        cur_sigma = statistics.mean(r["sigma"] for r in rows)
        print(f"sigma trend:  {prev_sigma:.3f} -> {cur_sigma:.3f} "
              f"({'COLLAPSED' if cur_sigma < prev_sigma * 0.5 else 'persists'})")
    all_guard = sum(r["guard_pass"] for r in rows), sum(r["n"] for r in rows)
    print(f"guard stability: {all_guard[0]}/{all_guard[1]} seeds pass 0.9")

    if winner and prev:
        pruns, _ = load_gen(prev)
        prev_fails = Counter()
        for r in pruns:
            prev_fails.update(r["fails_by_family"])
        cur_fails = Counter()
        for r in winner["runs"]:
            cur_fails.update(r["fails_by_family"])
        pn, cn = max(1, len(pruns)), max(1, len(winner["runs"]))
        print("\n== family deltas (winner vs prev gen, avg fails/round) ==")
        for fam in sorted(set(prev_fails) | set(cur_fails)):
            print(f"  {fam:24} {prev_fails[fam]/pn:6.1f} -> {cur_fails[fam]/cn:6.1f}")

    if best_run:
        print(f"\nbest checkpoint: {Path(gen_dir).name}/{best_run['dir']} "
              f"rate {best_run['rate']:.3f} guard {best_run['guard']:.3f}")
    packs = {r["meta"].get("trainpack", "?") for r in runs}
    print(f"trainpack(s): {packs}")

    print("\n== verdict ==")
    if gaps:
        print(f"PAUSED-GAPS: missing evals for {gaps}")
    elif winner and winner["mean"] >= 0.9:
        print("CLOSE-STAGE: winner clears the release bar")
    elif winner:
        print(f"ITERATE: winner {winner['config']} mean {winner['mean']:.3f} "
              f"— next lever per family deltas above")
    else:
        print("ITERATE: no config passed guards — retention is the blocker")


if __name__ == "__main__":
    main()
