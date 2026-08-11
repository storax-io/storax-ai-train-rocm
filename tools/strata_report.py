#!/usr/bin/env python3
"""Strata triage report: turn oracle-eval results into generator feedback.

Joins an oracle_eval output JSON with the eval suite's task metadata
(family, grammar_cells), clusters failures by normalized compiler-error
signature, and emits a markdown report where each cluster is one
actionable item: either a generator coverage gap (fix upstream) or a
model capability gap (worth GPU).

Usage:
  python3 tools/strata_report.py <eval.json> <tasks.jsonl> [-o report.md]
"""
import argparse
import collections
import json
import re
from pathlib import Path


def normalize_error(stderr_head, first_error=""):
    """First error line with file/line/col and identifiers scrubbed, so
    'parse_Currency' and 'parse_Biome' failures cluster together."""
    if (stderr_head or "").startswith("TRUNCATED-GENERATION"):
        return "TRUNCATED-GENERATION (raise --max-new)"
    if first_error:
        line = re.sub(r"^[^:]*:\d+:\d+:\s*", "", first_error)
        return re.sub(r"'[^']*'", "'_'", line).strip()
    line = next((l for l in (stderr_head or "").splitlines()
                 if "error:" in l), "")
    line = re.sub(r"^[^:]*:\d+:\d+:\s*", "", line)
    line = re.sub(r"'[^']*'", "'_'", line)
    return line.strip() or "(no compiler error captured)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json")
    ap.add_argument("tasks_jsonl")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    r = json.loads(Path(args.eval_json).read_text())
    tasks = {t["id"]: t for t in
             (json.loads(l) for l in
              Path(args.tasks_jsonl).read_text().splitlines() if l.strip())}

    fam_tot, fam_ok = collections.Counter(), collections.Counter()
    fam_repaired = collections.Counter()
    clusters = collections.defaultdict(list)
    for x in r["results"]:
        t = tasks.get(x["id"], {})
        fam = t.get("family", x["id"].rsplit("-", 1)[0])
        fam_tot[fam] += 1
        if x["ok"]:
            fam_ok[fam] += 1
            if x.get("repair_rounds_used"):
                fam_repaired[fam] += 1
        else:
            clusters[normalize_error(x.get("stderr_head"),
                                     x.get("first_error", ""))].append((x, t))

    lines = [f"# Strata triage — {Path(args.eval_json).name}",
             "",
             f"model: `{r.get('model')}`  ·  repair: {r.get('repair', 0)}  ·  "
             f"total: **{r.get('compile_pass')}/{r.get('total')}**",
             "", "## Per-family", "",
             "| family | pass | via repair |", "|---|---|---|"]
    for fam in sorted(fam_tot):
        lines.append(f"| {fam} | {fam_ok[fam]}/{fam_tot[fam]} "
                     f"| {fam_repaired[fam]} |")

    lines += ["", "## Failure clusters (one item each)", ""]
    for sig, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        fams = collections.Counter(t.get("family", "?") for _, t in members)
        cells = collections.Counter(
            c for _, t in members for c in t.get("grammar_cells", []))
        lines += [f"### {len(members)}× `{sig}`",
                  f"- families: {dict(fams)}",
                  f"- grammar cells involved: "
                  f"{[c for c, _ in cells.most_common(6)]}",
                  f"- sample: `{members[0][0]['id']}` — "
                  f"`{(members[0][0].get('code_head') or '')[:120]}`", ""]

    out = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(out)
        print(f"wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
