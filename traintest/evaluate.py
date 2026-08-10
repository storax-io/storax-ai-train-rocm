"""Evaluate a model (base or fine-tuned) on the facts QA sets with greedy
decoding. A question counts as correct iff the distinctive answer key
appears (case-insensitive) in the generation. Writes one JSON object to
--out and prints it."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import facts
import hfcompat


@torch.no_grad()
def ask(model, tok, question, max_new_tokens=48, thinking=False, system=None):
    ids = hfcompat.chat_prompt_ids(
        tok, [{"role": "user", "content": question}],
        thinking=thinking, system=system).unsqueeze(0).cuda()
    out = model.generate(ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    if thinking:
        # </think> is a special token — decode raw to find the boundary,
        # then match keys only against the answer: traces recite rosters
        # and would false-positive nearly any surname key.
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        if "</think>" in gen:
            gen = gen.rsplit("</think>", 1)[1]
        for t in (tok.eos_token, "<|im_end|>", tok.pad_token or ""):
            if t:
                gen = gen.replace(t, "")
        return gen.strip()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local model dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets",
                    default="train,paraphrase,composition,multihop,adjacent,control,retention")
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--system", default=None,
                    help="system-prompt override — must match training")
    ap.add_argument("--think", action="store_true",
                    help="evaluate in thinking mode (slower; answers matched after </think>)")
    args = ap.parse_args()

    tok = hfcompat.load_tokenizer(args.model)
    model = hfcompat.load_causal_model(args.model, torch.bfloat16, args.attn)
    model.cuda().eval()

    all_sets = facts.eval_sets()
    report = {"model": args.model, "sets": {}}
    for name in args.sets.split(","):
        items = all_sets[name]
        details = []
        correct = 0
        for question, key in items:
            # Think budget fits the BASE model's natural ~650-token traces
            # (measured) — a truncated trace never reaches its answer and
            # would score an unfair 0.
            base_new = 256 if "|" in key else 48
            gen = ask(model, tok, question,
                      max_new_tokens=base_new + (760 if args.think else 0),
                      thinking=args.think, system=args.system)
            # "a|b|c" = multi-key: all parts must appear (enumerations).
            hit = all(k.lower() in gen.lower() for k in key.split("|"))
            correct += hit
            details.append({"q": question, "key": key, "hit": hit,
                            "gen": gen[:160]})
        report["sets"][name] = {
            "correct": correct, "total": len(items),
            "accuracy": round(correct / len(items), 3),
            "details": details,
        }
        print(f"{name}: {correct}/{len(items)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("RESULT " + json.dumps(
        {k: v["accuracy"] for k, v in report["sets"].items()}), flush=True)


if __name__ == "__main__":
    main()
