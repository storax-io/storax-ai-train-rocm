"""Merge sharded oracle_eval reports (--shard I/M) into one.

    python3 tools/merge_eval.py merged.json shard0.json shard1.json ...

Recomputes pass counts and rate; keeps per-task results deduped by id
(later files win, matching oracle_eval's own rerun-merge semantics).
"""
import json
import sys
from pathlib import Path


def main():
    out, *parts = sys.argv[1:]
    merged, meta = {}, None
    for p in parts:
        rep = json.loads(Path(p).read_text())
        meta = meta or rep
        for r in rep["results"]:
            merged[r["id"]] = r
    results = list(merged.values())
    ok = sum(1 for r in results if r["ok"])
    report = {k: meta[k] for k in
              ("model", "suite", "run", "repair", "max_new")}
    report.update({"compile_pass": ok, "total": len(results),
                   "rate": round(ok / max(1, len(results)), 3),
                   "shards": len(parts), "results": results})
    Path(out).write_text(json.dumps(report, indent=2))
    print("RESULT " + json.dumps({"compile_rate": report["rate"],
                                  "pass": ok, "total": len(results),
                                  "shards": len(parts)}), flush=True)


if __name__ == "__main__":
    main()
