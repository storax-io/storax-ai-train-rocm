"""One-shot free-form chat probes for judging general chat quality drift
between base and fine-tuned models. Prints each answer delimited."""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROBES = [
    "Hi! What can you help me with?",
    "Write a two-line poem about autumn.",
    "Explain in one paragraph why the sky is blue.",
    "Which countries border Finland?",
    "What is 17 + 25?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.cuda().eval()
    for q in PROBES:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True,
            return_tensors="pt", enable_thinking=False).cuda()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=100, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        print(f"### {q}")
        print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()[:400])
        print()


if __name__ == "__main__":
    main()
