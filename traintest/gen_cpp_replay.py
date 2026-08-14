"""Ordinary-modern-C++ anchor replay: the BASE model answers everyday
C++17/20/23 tasks; every answer is compiled (and run) by the g++ oracle,
and only passing (prompt, code) pairs are kept. Anchors normal C++
competence while feature-dense C++26 corpora train the new constructs —
the domain analogue of chat replay anchoring, but compiler-verified.

Writes cpp_replay.json next to this script; cpp26dsdata picks it up.

  scripts/run_linux.sh gen_cpp_replay.py --model <base> --system "..."
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch

import hfcompat
from oracle_client import Oracle

TASKS = [
    "Write a C++ program that sorts a std::vector<int> and prints it.",
    "Write a C++ class Rectangle with width/height, an area() method, and a demo in main.",
    "Write a C++ program using std::map<std::string,int> to count word occurrences in a hardcoded list and print them.",
    "Write a C++ function template max3 returning the largest of three values; demonstrate with int and double.",
    "Write a C++ program that uses std::optional to represent integer parsing failure and handles both cases.",
    "Write a C++ RAII class that prints on construction and destruction; demonstrate scope-based lifetime.",
    "Write a C++ program using range-based for and structured bindings over a std::map.",
    "Write a C++ program that filters even numbers from a vector using std::ranges and prints them.",
    "Write a C++ struct Point with operator+ and operator== ; demonstrate both.",
    "Write a C++ program using std::string_view to trim leading spaces from a string.",
    "Write a C++ program with an enum class and a switch statement over it.",
    "Write a C++ program using std::variant<int,std::string> and std::visit to print either.",
    "Write a C++ program that computes the sum of squares of 1..10 with std::accumulate.",
    "Write a C++ class hierarchy: Shape with virtual area(), Circle and Square overriding it; print areas via base pointers.",
    "Write a C++ program using std::unique_ptr to manage a small object and transfer ownership.",
    "Write a C++ constexpr function factorial with a static_assert and runtime print.",
    "Write a C++ program that splits a comma-separated std::string into a vector of tokens.",
    "Write a C++ program using std::array and std::ranges::reverse, printing before and after.",
    "Write a C++ lambda that captures a local by reference and modifies it; print the result.",
    "Write a C++ program with a namespace, a free function in it, and a using-declaration in main.",
    "Write a C++ program that uses std::pair and std::tie to return and unpack two values.",
    "Write a C++ program defining a concept Numeric and a constrained template function that doubles a value.",
    "Write a C++ program using if constexpr in a template to handle integral vs floating-point differently.",
    "Write a C++ program with std::format (or std::print) formatting an int, a double and a string.",
    "Write a C++ program that reverses a std::string in place and prints it.",
    "Write a C++ struct with a default member initializer, aggregate initialization in main, and printed fields.",
    "Write a C++ program using std::span over a C array to sum elements.",
    "Write a C++ program with a static member counter incremented per constructed instance; print the count.",
    "Write a C++ program using std::sort with a custom comparator sorting strings by length.",
    "Write a C++ program that builds a 2D std::vector grid and prints it row by row.",
]


def extract_code(text):
    m = re.search(r"```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-new", type=int, default=1200)  # 600 truncated
    # every answer from the more verbose 14B (0/30 kept, LUMI job 21140120)
    args = ap.parse_args()

    oracle = Oracle()
    tok = hfcompat.load_tokenizer(args.model)
    model = hfcompat.load_causal_model(args.model, torch.bfloat16, "sdpa")
    model.cuda().eval()

    kept, rejected = [], 0
    for i, task in enumerate(TASKS):
        prompt = task + " Only output the code."
        ids = hfcompat.chat_prompt_ids(
            tok, [{"role": "user", "content": prompt}], thinking=False,
            system=args.system).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen_ids = out[0, ids.shape[1]:]
        truncated = gen_ids[-1].item() != tok.eos_token_id
        code = extract_code(tok.decode(gen_ids, skip_special_tokens=True))
        v = oracle.compile(code, run=True)
        if v.get("ok") and v.get("run_rc") == 0 and not truncated:
            kept.append({"prompt": prompt, "code": code})
        else:
            rejected += 1
            # a reject must explain itself — 0/30 with silent discards
            # cost a rerun to diagnose (LUMI job 21140120)
            err = (v.get("stderr") or v.get("run_stderr") or "")
            reason = ("TRUNCATED at max-new" if truncated else
                      "; ".join(err.splitlines()[:2]) or f"rc={v.get('run_rc')}")
            print(f"reject[{i}]: {reason}\n  head: {code[:120]!r}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(TASKS)} kept={len(kept)}", flush=True)

    dest = Path(os.environ.get("TRAINTEST_REPLAY_DIR",
                            Path(__file__).resolve().parent)) / "cpp_replay.json"
    dest.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    print(f"wrote {dest}: {len(kept)} verified anchors, {rejected} rejected",
          flush=True)


if __name__ == "__main__":
    main()
