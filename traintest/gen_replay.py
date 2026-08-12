"""Self-distillation replay: the BASE model answers diverse general
prompts; those (prompt, answer) pairs are mixed into fine-tuning so chat
style and general knowledge stay anchored to the base model instead of
drifting toward the domain corpus (full2 drifted: one-word answers,
corrupted neighbor-country knowledge).

Writes replay.json next to this script (the staging dir); the WSL side
copies it back to the repo's data/ so later runs reuse it.
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import hfcompat

TOPICS = [
    "Explain how photosynthesis works.",
    "What is the difference between RAM and a hard drive?",
    "Write a short story opening about a lighthouse keeper.",
    "How do vaccines work?",
    "What causes seasons on Earth?",
    "Give me three tips for improving sleep quality.",
    "Explain the rules of chess briefly.",
    "What is compound interest and why does it matter?",
    "Describe the water cycle.",
    "What is the difference between weather and climate?",
    "How does a refrigerator keep food cold?",
    "Write a haiku about the ocean.",
    "What are black holes?",
    "Explain supply and demand with an example.",
    "How do airplanes stay in the air?",
    "What is DNA and what does it do?",
    "Summarize the plot of a classic hero's journey.",
    "What is the Pythagorean theorem used for?",
    "How is chocolate made?",
    "What are the main renewable energy sources?",
    "Explain what an operating system does.",
    "Why is the ocean salty?",
    "What is inflation in economics?",
    "Describe how earthquakes happen.",
    "What is machine learning in simple terms?",
    "Give advice for a first job interview.",
    "How does the human immune system work?",
    "What is the difference between bacteria and viruses?",
    "Explain gravity to a ten-year-old.",
    "What makes a good cup of coffee?",
    "How do bees make honey?",
    "What is the greenhouse effect?",
    "Write a limerick about a cat.",
    "What is the difference between a star and a planet?",
    "How does GPS know where you are?",
    "What are the primary colors and how do they mix?",
    "Explain what a database is.",
    "Why do leaves change color in autumn?",
    "What is the speed of light and why is it important?",
    "How do muscles grow with exercise?",
    "What is the difference between an alligator and a crocodile?",
    "Explain the concept of time zones.",
    "How is glass made?",
    "What is a healthy breakfast?",
    "Describe how sound travels.",
    "What is the role of the United Nations?",
    "How do plants know when to bloom?",
    "What is quantum computing in simple terms?",
    "Give three tips for learning a new language.",
    "What causes rainbows?",
    "How does a battery store energy?",
    "What is the difference between fiction and nonfiction?",
    "Explain how tides work.",
    "What is cholesterol and why does it matter?",
    "How do cameras capture images?",
    "What was the Industrial Revolution?",
    "Explain what APIs are to a non-programmer.",
    "Why do we dream?",
    "What is the tallest mountain in the world and how was it formed?",
    "How does recycling paper work?",
    "What is the difference between a violin and a viola?",
    "Explain herd immunity.",
    "How do submarines dive and surface?",
    "What is a solar eclipse?",
    "Give tips for writing a clear email.",
    "What is the function of the liver?",
    "How does yeast make bread rise?",
    "What is the difference between latitude and longitude?",
    "Explain how magnets work.",
    "What is the largest desert in the world?",
    "How do birds navigate during migration?",
    "What is open-source software?",
    "Describe the layers of the Earth.",
    "What makes thunder and lightning?",
    "How is cheese made?",
    "What is the difference between a marathon and a triathlon?",
    "Explain the placebo effect.",
    "How do noise-cancelling headphones work?",
    "What is the boiling point of water at high altitude and why?",
    "Write two sentences describing a rainy morning.",
    # Drift-zone anchors: the fine-tune's domain (Finnish presidents)
    # bleeds into nearby European geography/institutions. These anchor
    # the CATEGORY with the base model's own knowledge — deliberately
    # not the adjacent-eval questions themselves.
    "Give a brief overview of the Nordic countries.",
    "What is the European Union and when was it formed?",
    "What currency do most European Union countries use?",
    "Describe the geography of Northern Europe.",
    "Which languages are spoken in Scandinavia?",
    "Tell me about the Baltic Sea region.",
    "How does a parliamentary democracy work?",
    "What are the capitals of the G7 countries?",
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B",
                    help="replay MUST come from the model being trained — "
                         "another model's outputs anchor the wrong distribution")
    ap.add_argument("--system", default=None,
                    help="system-prompt override; must match training")
    ap.add_argument("--think", action="store_true",
                    help="generate thinking-mode replay (trace + answer)")
    args = ap.parse_args()

    tok = hfcompat.load_tokenizer(args.model)
    model = hfcompat.load_causal_model(args.model, torch.bfloat16, "sdpa")
    model.cuda().eval()

    # Think traces are long; use fewer prompts and cap tightly so the
    # (prompt + trace + answer) still fits the training seq_len (320).
    topics = TOPICS[:40] if args.think else TOPICS
    max_new = 240 if args.think else 180

    out = []
    for i, q in enumerate(topics):
        ids = hfcompat.chat_prompt_ids(
            tok, [{"role": "user", "content": q}],
            thinking=args.think, system=args.system).unsqueeze(0).cuda()
        with torch.no_grad():
            gen = model.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        if args.think:
            # <think>/</think> are special tokens — skip_special_tokens
            # would strip them and lose the trace/answer boundary.
            ans = tok.decode(gen[0, ids.shape[1]:],
                             skip_special_tokens=False).strip()
            for t in (tok.eos_token, "<|im_end|>", tok.pad_token or ""):
                if t:
                    ans = ans.replace(t, "")
            ans = ans.strip()
            if "</think>" not in ans:
                continue  # trace truncated — useless as a training target
        else:
            ans = tok.decode(gen[0, ids.shape[1]:],
                             skip_special_tokens=True).strip()
        out.append({"prompt": q, "answer": ans})
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(topics)}", flush=True)

    dest = Path(os.environ.get("TRAINTEST_REPLAY_DIR",
                            Path(__file__).resolve().parent)) / (
        "replay_think.json" if args.think else "replay.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"wrote {dest} ({len(out)} pairs)")


if __name__ == "__main__":
    main()
