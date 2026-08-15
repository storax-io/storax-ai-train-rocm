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


def generate_batch(model, tok, batch_msgs, system, max_new):
    """Left-padded batched greedy generate — one forward pass serves the
    whole wave instead of single-stream decodes (~5-10x per-GCD eval
    throughput; single-stream uses a few percent of an MI250X GCD)."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        return _generate_batch_once(model, tok, batch_msgs, system, max_new,
                                    pad)
    except torch.OutOfMemoryError:
        # 16 long-winded sequences fragment the growing KV cache (base-model
        # eval OOMed at 53 GiB with 8-10 GiB reserved-unallocated). Rows are
        # independent under left-padded greedy — split and retry.
        if len(batch_msgs) == 1:
            raise
        torch.cuda.empty_cache()
        mid = len(batch_msgs) // 2
        print(f"generate_batch: OOM at batch {len(batch_msgs)} — splitting",
              flush=True)
        return (generate_batch(model, tok, batch_msgs[:mid], system, max_new)
                + generate_batch(model, tok, batch_msgs[mid:], system,
                                 max_new))


def _generate_batch_once(model, tok, batch_msgs, system, max_new, pad):
    seqs = [hfcompat.chat_prompt_ids(tok, m, thinking=False, system=system)
            for m in batch_msgs]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, width - len(s):] = s
        attn[r, width - len(s):] = 1
    ids, attn = ids.cuda(), attn.cuda()
    out = model.generate(ids, attention_mask=attn, max_new_tokens=max_new,
                         do_sample=False, pad_token_id=pad)
    res = []
    for r in range(len(seqs)):
        gen_ids = out[r, width:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        # per-row truncation: a row that never emitted EOS ran to the cap
        hit_cap = (len(gen_ids) >= max_new
                   and tok.eos_token_id not in gen_ids.tolist())
        res.append((text, hit_cap))
    return res


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
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"],
                    help="vllm: continuous-batched generation (whole shard "
                         "in flight) — ~10x eval throughput; hf: sequential "
                         "transformers generate (reference semantics)")
    ap.add_argument("--shard", default="", metavar="I/M",
                    help="evaluate tasks[i::m] only — run M instances (one "
                         "per GCD) and merge with tools/merge_eval.py")
    args = ap.parse_args()

    oracle = Oracle(args.url) if args.url else Oracle()
    print("oracle:", json.dumps(oracle.health()), flush=True)

    tok = hfcompat.load_tokenizer(args.model)
    model = None
    if args.backend == "hf":
        model = hfcompat.load_causal_model(args.model, torch.bfloat16,
                                           args.attn)
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

    # Wave-based evaluation: generate for every unresolved task, verify
    # against the oracle, and carry failures (with compiler feedback
    # appended) into the next wave. Repair semantics identical to the old
    # per-task loop; waves are what let the vllm backend batch an entire
    # shard through continuous batching instead of single-stream decode.
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams, TokensPrompt
        # max_model_len: the checkpoint advertises 262k context, for which
        # vLLM would reserve a 40GiB KV cache that does not fit beside the
        # weights (LUMI job 21151918). Worst real conversation here is
        # prompt + repair rounds carrying 16k generations ~= 35k tokens.
        llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=False,
                  max_model_len=65536)

        def batch_generate(batch_msgs):
            prompts = [TokensPrompt(prompt_token_ids=hfcompat.chat_prompt_ids(
                tok, m, thinking=False, system=args.system).tolist())
                for m in batch_msgs]
            # detokenize=False: the container's HF tokenizer garbles
            # byte-level decode (see hfcompat.load_tokenizer) — decode
            # returned ids with our sanity-gated tokenizer instead
            sp = SamplingParams(temperature=0, max_tokens=args.max_new,
                                stop_token_ids=[tok.eos_token_id],
                                detokenize=False)
            outs = llm.generate(prompts, sp)
            return [(tok.decode(list(o.outputs[0].token_ids),
                                skip_special_tokens=True),
                     o.outputs[0].finish_reason == "length") for o in outs]
    else:
        def batch_generate(batch_msgs):
            return generate_batch(model, tok, batch_msgs, args.system,
                                  args.max_new)

    states = [{"task": t, "msgs": [{"role": "user", "content": t["prompt"]}],
               "ok": False, "rounds_used": 0, "verdict": {}, "code": "",
               "truncated": False} for t in tasks]
    active = states
    for attempt in range(args.repair + 1):
        if not active:
            break
        gens = batch_generate([s["msgs"] for s in active])
        # a 0/N wave with universal truncation is indistinguishable from a
        # broken prompt path without seeing output — always show one head
        print(f"wave {attempt} sample [{active[0]['task']['id']}] "
              f"truncated={gens[0][1]} head: {gens[0][0][:200]!r}", flush=True)
        nxt = []
        for s, (gen, truncated) in zip(active, gens):
            s["truncated"] = truncated
            s["code"] = extract_code(gen)
            try:
                s["verdict"] = oracle.compile(s["code"], run=args.run)
            except Exception as e:  # noqa: BLE001 — oracle/network failure
                s["verdict"] = {"ok": False, "error": repr(e)}
            s["ok"] = bool(s["verdict"].get("ok")) and (
                not args.run or s["verdict"].get("run_rc") == 0)
            s["rounds_used"] = attempt
            if not s["ok"] and attempt < args.repair:
                s["msgs"] += [{"role": "assistant", "content": gen},
                              {"role": "user", "content":
                               "That does not compile. Compiler output:\n"
                               + (s["verdict"].get("stderr")
                                  or s["verdict"].get("error", ""))[:1200]
                               + "\nFix the program. Only output the "
                                 "corrected code."}]
                nxt.append(s)
        print(f"wave {attempt}: {sum(1 for s in states if s['ok'])}"
              f"/{len(states)} passing, {len(nxt)} to repair", flush=True)
        active = nxt

    results = []
    ok_count = 0
    for s in states:
        t, verdict, ok = s["task"], s["verdict"], s["ok"]
        ok_count += ok
        full_err = verdict.get("stderr") or ""
        first_error = next((l for l in full_err.splitlines()
                            if "error:" in l), "")
        results.append({"id": t["id"], "ok": ok,
                        "repair_rounds_used": s["rounds_used"],
                        "truncated": s["truncated"],
                        "rc": verdict.get("rc"),
                        "ms": verdict.get("ms"),
                        # template cascades bury the error line beyond any
                        # head-truncation — extract it before truncating
                        "first_error": first_error[:300],
                        "stderr_head": ("TRUNCATED-GENERATION\n"
                                        if s["truncated"] and not ok else "")
                                       + full_err[:400],
                        "code_head": s["code"][:200]})
        tag = "PASS" if ok else "FAIL"
        if s["truncated"] and not ok:
            tag += " (truncated)"
        if ok and s["rounds_used"]:
            tag += f" (repair {s['rounds_used']})"
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
