"""Compiler-verified evaluation: the model generates C++ from prompts and
the g++ oracle (GCC 16.1, C++26 reflection + contracts) judges each answer
by compiling it. No substring matching — the compiler is the ground truth.

  scripts/run_linux.sh oracle_eval.py --model <id-or-dir> \
      --suite ../data/cpp26_probes.jsonl --url http://<oracle-host>:8950 \
      --out <out.json>

Suite format: JSONL with {id, prompt}; generation is greedy; the first
```-fenced code block is compiled (whole output if no fence).
"""
import argparse
import json
import re
from pathlib import Path

import torch

import hfcompat
from oracle_client import Oracle


def extract_code(text):
    m = re.search(r"```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


@torch.no_grad()
def generate(model, tok, msgs, system, max_new):
    """Returns (text, truncated) — truncated means the generation hit the
    token cap, so a compile failure would be an artifact, not a verdict."""
    ids = hfcompat.chat_prompt_ids(
        tok, msgs, thinking=False, system=system).unsqueeze(0).cuda()
    out = model.generate(ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen_len = out.shape[1] - ids.shape[1]
    return (tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True),
            gen_len >= max_new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=None, help="oracle URL(s), else ORACLE_URL")
    ap.add_argument("--system", default=None)
    # Cap only binds on runaway generations (EOS ends normal answers).
    # 4096 covers STL-scale programs while bounding a local runaway at
    # ~80s; on LUMI use --max-new 16000 (faster GPUs, thinking traces).
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--run", action="store_true", help="also execute a.out")
    ap.add_argument("--rerun-truncated", default=None, metavar="PREV_JSON",
                    help="re-evaluate ONLY tasks that failed truncated in a "
                         "previous result file (at the current --max-new) "
                         "and merge verdicts into --out")
    ap.add_argument("--repair", type=int, default=0,
                    help="repair rounds: on compile failure, feed the "
                         "compiler error back and regenerate (matches the "
                         "storax pipeline's max_repair_rounds semantics)")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--shard", default="", metavar="I/M",
                    help="evaluate tasks[i::m] only — run M instances (one "
                         "per GCD) and merge with tools/merge_eval.py")
    args = ap.parse_args()

    oracle = Oracle(args.url) if args.url else Oracle()
    print("oracle:", json.dumps(oracle.health()), flush=True)

    tok = hfcompat.load_tokenizer(args.model)
    model = hfcompat.load_causal_model(args.model, torch.bfloat16, args.attn)
    model.cuda().eval()

    tasks = [json.loads(l) for l in Path(args.suite).read_text().splitlines()
             if l.strip()]
    prev = None
    if args.rerun_truncated:
        prev = json.loads(Path(args.rerun_truncated).read_text())
        redo = {x["id"] for x in prev["results"]
                if x.get("truncated") and not x["ok"]}
        tasks = [t for t in tasks if t["id"] in redo]
        print(f"rerun-truncated: {len(tasks)} task(s) from "
              f"{args.rerun_truncated} at max_new={args.max_new}", flush=True)
        if not tasks:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(prev, indent=2))
            print("RESULT " + json.dumps({"compile_rate": prev["rate"],
                                          "pass": prev["compile_pass"],
                                          "total": prev["total"],
                                          "rerun": 0}), flush=True)
            return
    if args.limit:
        # Stratify across template families — suites are grouped, so a
        # head-slice samples a single family (measured: 24/24 same family,
        # gated by one shared syntax form).
        by_fam = {}
        for t in tasks:
            by_fam.setdefault(t.get("family", ""), []).append(t)
        picked, i = [], 0
        while len(picked) < args.limit and any(by_fam.values()):
            for fam in sorted(by_fam):
                if by_fam[fam] and len(picked) < args.limit:
                    picked.append(by_fam[fam].pop(0))
            i += 1
        tasks = picked
    if args.shard:
        i, m = (int(x) for x in args.shard.split("/"))
        tasks = tasks[i::m]
        print(f"shard {i}/{m}: {len(tasks)} tasks", flush=True)

    results = []
    ok_count = 0
    for t in tasks:
        msgs = [{"role": "user", "content": t["prompt"]}]
        ok = False
        rounds_used = 0
        verdict = {}
        code = ""
        truncated = False
        for attempt in range(args.repair + 1):
            gen, truncated = generate(model, tok, msgs, args.system,
                                      args.max_new)
            code = extract_code(gen)
            try:
                verdict = oracle.compile(code, run=args.run)
            except Exception as e:  # noqa: BLE001 — oracle/network failure
                verdict = {"ok": False, "error": repr(e)}
            ok = bool(verdict.get("ok")) and (not args.run
                                              or verdict.get("run_rc") == 0)
            rounds_used = attempt
            if ok or attempt == args.repair:
                break
            msgs += [{"role": "assistant", "content": gen},
                     {"role": "user", "content":
                      "That does not compile. Compiler output:\n"
                      + (verdict.get("stderr") or verdict.get("error", ""))[:1200]
                      + "\nFix the program. Only output the corrected code."}]
        ok_count += ok
        full_err = verdict.get("stderr") or ""
        first_error = next((l for l in full_err.splitlines()
                            if "error:" in l), "")
        results.append({"id": t["id"], "ok": ok,
                        "repair_rounds_used": rounds_used,
                        "truncated": truncated,
                        "rc": verdict.get("rc"),
                        "ms": verdict.get("ms"),
                        # template cascades bury the error line beyond any
                        # head-truncation — extract it before truncating
                        "first_error": first_error[:300],
                        "stderr_head": ("TRUNCATED-GENERATION\n"
                                        if truncated and not ok else "")
                                       + full_err[:400],
                        "code_head": code[:200]})
        tag = "PASS" if ok else "FAIL"
        if truncated and not ok:
            tag += " (truncated)"
        if ok and rounds_used:
            tag += f" (repair {rounds_used})"
        print(f"{tag}  {t['id']}", flush=True)

    if prev is not None:
        merged = {x["id"]: x for x in prev["results"]}
        merged.update({x["id"]: x for x in results})
        results = list(merged.values())
        ok_count = sum(1 for x in results if x["ok"])
    report = {"model": args.model, "suite": args.suite, "run": args.run,
              "repair": args.repair, "max_new": args.max_new,
              "compile_pass": ok_count, "total": len(results),
              "rate": round(ok_count / max(1, len(results)), 3),
              "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("RESULT " + json.dumps({"compile_rate": report["rate"],
                                  "pass": ok_count, "total": len(tasks)}),
          flush=True)


if __name__ == "__main__":
    main()
