"""Expert-iteration HARVEST: the model samples best-of-N answers to FRESH
oracle-checkable prompts (never the eval suite — that stays a clean test
set); the compiler keeps the winners. Verified winners become the next
trainpack's expert band — the model teaching itself whatever it can
already sample but not yet rank first.

  harvest.py --model <dir> --prompts prompts.jsonl --out winners.jsonl \
             --samples 8 --temperature 0.8 --shard I/M

Prompts: JSONL {id, prompt[, family]}. Winners: JSONL {id, prompt,
source, family, sample, gen_tokens} — only answers that COMPILED, RAN,
and HELD their assertions. Dedup by code hash: N samples of one idea
count once.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import hfcompat
from oracle_client import Oracle
from oracle_eval import extract_code, _free_cuda, EVAL_BATCH_MAX


@torch.no_grad()
def sample_batch(model, tok, batch_msgs, system, max_new, temperature, top_p):
    """Left-padded batched SAMPLED generation (oracle_eval's batcher is
    greedy by design — the judge must be deterministic; the harvester
    must not be)."""
    if len(batch_msgs) > EVAL_BATCH_MAX:
        out = []
        for i in range(0, len(batch_msgs), EVAL_BATCH_MAX):
            out += sample_batch(model, tok, batch_msgs[i:i + EVAL_BATCH_MAX],
                                system, max_new, temperature, top_p)
            _free_cuda(model)
        return out
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = [hfcompat.chat_prompt_ids(tok, m, thinking=False, system=system)
            for m in batch_msgs]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, width - len(s):] = s
        attn[r, width - len(s):] = 1
    ids, attn = ids.cuda(), attn.cuda()
    try:
        out = model.generate(ids, attention_mask=attn, max_new_tokens=max_new,
                             do_sample=True, temperature=temperature,
                             top_p=top_p, pad_token_id=pad)
    except torch.OutOfMemoryError:
        if len(batch_msgs) == 1:
            raise
        _free_cuda(model)
        mid = len(batch_msgs) // 2
        return (sample_batch(model, tok, batch_msgs[:mid], system, max_new,
                             temperature, top_p)
                + sample_batch(model, tok, batch_msgs[mid:], system, max_new,
                               temperature, top_p))
    res = []
    for r in range(len(seqs)):
        gen_ids = out[r, width:]
        lst = gen_ids.tolist()
        n_tok = (lst.index(tok.eos_token_id) + 1
                 if tok.eos_token_id in lst else len(lst))
        res.append((tok.decode(gen_ids, skip_special_tokens=True), n_tok))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--shard", default="", metavar="I/M")
    args = ap.parse_args()

    oracle = Oracle(args.url) if args.url else Oracle()
    print("oracle:", json.dumps(oracle.health()), flush=True)
    tok = hfcompat.load_tokenizer(args.model)
    model = hfcompat.load_causal_model(args.model, torch.bfloat16, args.attn)
    model.cuda().eval()

    tasks = [json.loads(l) for l in Path(args.prompts).read_text().splitlines()
             if l.strip()]
    if args.shard:
        i, m = (int(x) for x in args.shard.split("/"))
        tasks = tasks[i::m]
    print(f"harvest: {len(tasks)} prompts x {args.samples} samples "
          f"@T={args.temperature}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # resume: prompts already harvested (any verdict) are skipped — a killed
    # shard loses only its unfinished prompts (every cycle contributes)
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            try:
                done.add(json.loads(ln)["id"])
            except Exception:
                pass
    marker = out_path.with_suffix(".done.jsonl")
    if marker.exists():
        done |= set(marker.read_text().split())
    tasks = [t for t in tasks if t["id"] not in done]
    print(f"resume: {len(done)} prompt(s) already decided, {len(tasks)} to go",
          flush=True)

    import time as _time
    solved = kept = 0
    with out_path.open("a") as wf, marker.open("a") as mf:
        # wave = one prompt-group chunk; each prompt replicated N times so
        # the batcher amortizes the whole group in one forward pass
        group = max(1, EVAL_BATCH_MAX // args.samples)
        for gi in range(0, len(tasks), group):
            wave = tasks[gi:gi + group]
            _t0 = _time.monotonic()
            msgs = [[{"role": "user", "content": t["prompt"]}]
                    for t in wave for _ in range(args.samples)]
            gens = sample_batch(model, tok, msgs, args.system,
                                args.max_new, args.temperature, args.top_p)
            for ti, t in enumerate(wave):
                seen_hash, hit = set(), 0
                for si in range(args.samples):
                    text, n_tok = gens[ti * args.samples + si]
                    code = extract_code(text)
                    h = hashlib.sha256(code.encode()).hexdigest()[:16]
                    if not code or h in seen_hash:
                        continue
                    seen_hash.add(h)
                    try:
                        v = oracle.compile(code, run=True)
                    except Exception as e:  # noqa: BLE001
                        v = {"ok": False, "error": repr(e)}
                    if v.get("ok") and v.get("run_rc") == 0:
                        hit += 1
                        wf.write(json.dumps({
                            "id": t["id"], "family": t.get("family", ""),
                            "prompt": t["prompt"], "source": code,
                            "sample": si, "gen_tokens": n_tok,
                            "model": args.model,
                            "temperature": args.temperature}) + "\n")
                mf.write(t["id"] + "\n")
                if hit:
                    solved += 1
                    kept += hit
            wf.flush(); mf.flush()
            dt = _time.monotonic() - _t0
            print(f"wave {gi // group}: {solved}/{gi + len(wave)} prompts "
                  f"solved, {kept} verified winners, {dt:.0f}s", flush=True)
    print("HARVEST " + json.dumps({"prompts": len(tasks) + len(done),
                                   "solved": solved, "winners": kept}),
          flush=True)


if __name__ == "__main__":
    main()
