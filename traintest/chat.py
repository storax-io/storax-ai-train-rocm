"""Interactive chat with SmolLM3 on the ROCm GPU — for probing what the
model does and doesn't know before picking training facts.

Commands inside the REPL:
  /think     toggle SmolLM3 extended thinking (default off)
  /clear     drop conversation history
  /temp X    set sampling temperature (0 = greedy)
  /quit      exit
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

import hfcompat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    print(f"loading {args.model} ...", flush=True)
    tok = hfcompat.load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.cuda().eval()
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    print(f"ready on {torch.cuda.get_device_name(0)} — /quit to exit", flush=True)

    history = []
    thinking = False
    temp = 0.0
    while True:
        try:
            user = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/clear":
            history = []
            print("(history cleared)")
            continue
        if user == "/think":
            thinking = not thinking
            print(f"(thinking {'on' if thinking else 'off'})")
            continue
        if user.startswith("/temp"):
            temp = float(user.split()[1])
            print(f"(temperature {temp})")
            continue

        history.append({"role": "user", "content": user})
        ids = hfcompat.chat_prompt_ids(
            tok, history, thinking=thinking).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new, streamer=streamer,
                do_sample=temp > 0, temperature=temp if temp > 0 else None,
                pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
