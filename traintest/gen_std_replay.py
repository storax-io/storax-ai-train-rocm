"""STANDARD-SWEEP replay anchors: the BASE model answers everyday tasks
per language standard — C17, C++98, C++11, C++17, C++20 — and every
answer is compiled with the MATCHING -std= flag and run; only passing
(prompt, code, std) triples are kept. The standard is named in the
prompt at birth (Henri 2026-08-23: "standard should be in prompt";
forge eval: C and older-C++ file-based at 0, C++23 scoring under C++26
— the model was never told which standard it writes).

Standalone by design (stdlib + transformers + torch + gcc/g++): runs on
suite hosts without the harness or the oracle container. Writes
std_replay.json next to this script (or TRAINTEST_REPLAY_DIR);
cpp26dsdata._std_replay picks it up and format-wraps at train time.

  python3 gen_std_replay.py --model <base-model-dir> [--batch 8]
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

SYSTEM = "You are a helpful assistant."  # identical to training/eval

C_TASKS = [
    "prints the sum of the integers 1..100",
    "reverses a string in place and prints it",
    "counts the words in a hardcoded sentence and prints the count",
    "sorts an array of ints with qsort and prints it",
    "implements a singly linked list with push and print, then frees it",
    "copies a string into heap memory with malloc, prints it, and frees it",
    "reads characters from a hardcoded string and counts vowels",
    "computes gcd of two numbers with the Euclidean algorithm and prints it",
    "prints the binary representation of an unsigned int using bit operations",
    "defines a struct point, an array of three points, and prints their distances from origin",
    "implements bubble sort over an int array and prints before and after",
    "uses a function pointer to select between add and multiply and prints both results",
    "concatenates two strings into a malloc'd buffer safely and prints it",
    "computes the factorial of 10 iteratively and prints it",
    "prints a 5x5 multiplication table with aligned columns",
    "counts the occurrences of each letter in a hardcoded string and prints nonzero counts",
    "implements a fixed-size stack of ints with push/pop and demonstrates it",
    "swaps two variables through pointers and prints them",
    "finds the largest and smallest element of an int array in one pass and prints both",
    "checks whether a hardcoded string is a palindrome ignoring case and prints the verdict",
    "converts a decimal number to hexadecimal without printf %x and prints it",
    "implements strlen and strcmp by hand and demonstrates them",
    "uses a union to inspect the bytes of a float and prints them in hex",
    "defines an enum of weekdays and prints the name for each value via a switch",
    "tokenizes a comma-separated string with strtok and prints each token",
    "computes the average of an int array as a double and prints it to two decimals",
    "implements binary search over a sorted int array and prints the found index",
    "prints the sizes of the basic integer types using sizeof",
]

CPP_TASKS = [
    "sorts a vector of ints with std::sort and prints it",
    "defines a class Rectangle with width, height and an area() method, demonstrated in main",
    "counts word occurrences in a hardcoded list using std::map and prints them",
    "reverses a std::string in place and prints it",
    "defines a Shape base class with virtual area(), Circle and Square overriding it, printed via base pointers",
    "implements a template function max3 returning the largest of three values, demonstrated with int and double",
    "splits a comma-separated std::string into a vector of tokens and prints them",
    "defines a struct Point with operator+ and operator==, both demonstrated",
    "uses std::stringstream to parse three integers from a string and prints their sum",
    "implements a RAII class that prints on construction and destruction, demonstrating scope-based lifetime",
    "builds a 2D vector grid and prints it row by row",
    "sorts strings by length with std::sort and a comparator and prints them",
    "computes the sum of squares of 1..10 with a loop and prints it",
    "stores key-value pairs in a std::map and iterates printing both",
    "implements a simple Stack class over std::vector with push/pop/top, demonstrated",
    "counts vowels in a std::string and prints the count",
    "defines a class with a static instance counter incremented per construction, printing the count",
    "uses std::set to deduplicate a list of ints and prints the unique values",
    "implements fizzbuzz for 1..30",
    "overloads operator<< for a custom struct and prints it",
    "finds the min and max of a vector with std::min_element and std::max_element and prints them",
    "implements a simple bank account class with deposit/withdraw and prints the balance",
    "uses std::pair to return two values from a function and prints them",
    "builds a frequency histogram of digits in a hardcoded number and prints it",
]

STDS = [
    ("C17", "gcc", "-std=c17", ".c", C_TASKS),
    ("C++98", "g++", "-std=c++98", ".cpp", CPP_TASKS),
    ("C++11", "g++", "-std=c++11", ".cpp", CPP_TASKS),
    ("C++17", "g++", "-std=c++17", ".cpp", CPP_TASKS),
    ("C++20", "g++", "-std=c++20", ".cpp", CPP_TASKS),
]


def extract_code(text):
    m = re.search(r"```(?:c|cpp|c\+\+|C|C\+\+)?\s*\n(.*?)```", text, re.S)
    code = (m.group(1) if m else text).strip()
    if "```" in code:
        code = "\n".join(l for l in code.splitlines()
                         if not l.strip().startswith("```")).strip()
    return code


def verify(code, compiler, stdflag, suffix):
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / ("prog" + suffix)
        src.write_text(code, encoding="utf-8")
        exe = Path(td) / "prog"
        try:
            c = subprocess.run(
                [compiler, stdflag, "-O1", "-o", str(exe), str(src)],
                capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return False, "compile timeout"
        if c.returncode != 0:
            return False, (c.stderr or "compile failed").splitlines()[0][:160]
        try:
            r = subprocess.run([str(exe)], capture_output=True, text=True,
                               timeout=5, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False, "run timeout"
        return r.returncode == 0, f"rc={r.returncode}"


def ids_of(tok, msgs, system=SYSTEM):
    msgs = [{"role": "system", "content": system}] + msgs
    try:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      enable_thinking=False)
    except TypeError:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True)
    if hasattr(out, "input_ids"):
        out = out.input_ids
    return list(out[0]) if out and isinstance(out[0], (list, tuple)) \
        else list(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.model,
                                            fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(args.model)
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16,
            attn_implementation=args.attn)
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16,
            attn_implementation=args.attn)
    try:
        model.generation_config.max_length = None
    except Exception:
        pass
    model = model.to(args.device).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id

    @torch.no_grad()
    def gen(prompts):
        res = []
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            seqs = [ids_of(tok, [{"role": "user", "content": p}])
                    for p in chunk]
            width = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            attn = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, width - len(s):] = torch.tensor(s)
                attn[r, width - len(s):] = 1
            out = model.generate(
                ids.to(args.device), attention_mask=attn.to(args.device),
                max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=pad)
            for r in range(len(seqs)):
                lst = out[r, width:].tolist()
                truncated = tok.eos_token_id not in lst
                res.append((tok.decode(out[r, width:],
                                       skip_special_tokens=True), truncated))
        return res

    kept, rejected = [], 0
    for std, compiler, stdflag, suffix, tasks in STDS:
        prompts = [f"Write a {std} program that {t}." for t in tasks]
        gens = gen(prompts)
        n0 = len(kept)
        for prompt, (text, truncated) in zip(prompts, gens):
            code = extract_code(text)
            ok, why = (False, "TRUNCATED") if truncated else \
                verify(code, compiler, stdflag, suffix)
            if ok and code:
                kept.append({"prompt": prompt, "code": code, "std": std})
            else:
                rejected += 1
                if rejected <= 12:
                    print(f"reject [{std}] {why}\n  head: {code[:100]!r}",
                          flush=True)
        print(f"{std}: {len(kept) - n0}/{len(tasks)} kept", flush=True)

    dest = Path(os.environ.get(
        "TRAINTEST_REPLAY_DIR",
        Path(__file__).resolve().parent)) / "std_replay.json"
    dest.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    print(f"wrote {dest}: {len(kept)} verified anchors "
          f"({rejected} rejected)", flush=True)


if __name__ == "__main__":
    main()
