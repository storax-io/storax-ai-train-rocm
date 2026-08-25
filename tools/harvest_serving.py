"""Harvest burst against a SERVED model (Henri 2026-08-25: run the
reliability harvest BEFORE cutting the pack — and make it double as the
C-QA generation).

    python3 tools/harvest_serving.py --serve http://127.0.0.1:8091 \
        --oracle http://127.0.0.1:8950 --out winners-harvest.jsonl \
        [--tasks tasks.jsonl] [--samples 16]

Two stages:
  1. Task bank: if --tasks is absent, DeepSeek (ToS-cleared teacher)
     authors fresh whole-program tasks per standard — C17 heavy (the
     eval's zero axis), C++11/17/20/23 for the reliability gap. Task
     PROMPTS are teacher-authored (allowed); ANSWERS come from our own
     served model (model-native).
  2. Best-of-N: sample the served model (temp 0.8) until a sample
     passes the STANDARD-MATCHED compiler gate (gcc -std=c17 for C —
     which structurally rejects C++-shaped answers — g++ for C++).
     First passing sample wins; winners carry {prompt, source, std}.

Never the eval suite. Winners append-ready for var/winners.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STD_PLAN = [("C17", 150), ("C++11", 60), ("C++17", 80), ("C++20", 60),
            ("C++23", 80)]

TASK_SYS = """You author programming exercise prompts. Output STRICT JSON:
an array of strings, each one a self-contained task asking for ONE
complete compilable program in {std}. Vary domain (strings, files,
math, structs, sorting, bit ops, state machines, parsing), vary length,
everyday style. For C tasks: pure C idioms (stdio, pointers, arrays,
structs) — things natural in C. No I/O interactivity, no external
libraries, no OS specifics. Do NOT number them."""


def _post_json(url, body, headers=None, timeout=180):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json",
                                  **(headers or {})})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def gen_tasks(out_path: Path, log=print):
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not set and no --tasks given")
    tasks = []
    for std, n in STD_PLAN:
        got = 0
        for batch in range((n + 29) // 30):
            r = _post_json(
                "https://api.deepseek.com/v1/chat/completions",
                {"model": "deepseek-v4-pro", "temperature": 1.0,
                 "max_tokens": 4000,
                 "messages": [
                     {"role": "system",
                      "content": TASK_SYS.format(std=std)},
                     {"role": "user",
                      "content": f"Write 30 varied tasks (batch "
                                 f"{batch + 1}). JSON array only."}]},
                headers={"Authorization": f"Bearer {key}"})
            text = r["choices"][0]["message"]["content"]
            m = re.search(r"\[.*\]", text, re.S)
            if not m:
                continue
            for t in json.loads(m.group(0)):
                if isinstance(t, str) and len(t) > 30 and got < n:
                    tasks.append({"id": f"hs-{std.lower().replace('+', 'p')}"
                                        f"-{hashlib.sha256(t.encode()).hexdigest()[:10]}",
                                  "prompt": t.strip(), "std": std})
                    got += 1
        log(f"[tasks] {std}: {got}")
    with out_path.open("w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    return tasks


def extract_code(text):
    m = re.search(r"```[a-z+]*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=900)
    a = ap.parse_args()
    out = Path(a.out)
    if a.tasks and Path(a.tasks).exists():
        tasks = [json.loads(l) for l in open(a.tasks) if l.strip()]
    else:
        tasks = gen_tasks(out.with_suffix(".tasks.jsonl"))
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open() if l.strip()}
    tasks = [t for t in tasks if t["id"] not in done]
    print(f"{len(tasks)} tasks to harvest, {a.samples} samples max each",
          flush=True)

    def gate(code, std):
        is_c = not std.startswith("C++")
        args = ([f"-std={std.lower()}", "-fsyntax-only"] if is_c else
                [f"-std={std.lower().replace('c++', 'c++')}",
                 "-fsyntax-only"])
        body = {"files": {"main.c" if is_c else "main.cpp": code},
                "args": args}
        if is_c:
            body["driver"] = "gcc"
        try:
            return _post_json(a.oracle.rstrip("/") + "/compile",
                              body, timeout=90).get("ok", False)
        except Exception:
            return False

    def harvest_one(t):
        std = t["std"]
        prompt = (f"{t['prompt']} Write one complete {std} program. "
                  f"Only output the code.")
        for k in range(a.samples):
            try:
                r = _post_json(a.serve.rstrip("/") + "/v1/chat/completions",
                               {"model": "any", "temperature": 0.8,
                                "top_p": 0.95, "max_tokens": a.max_new,
                                "messages": [{"role": "user",
                                              "content": prompt}]},
                               timeout=600)
                text = r["choices"][0]["message"]["content"]
            except Exception:
                time.sleep(2)
                continue
            code = extract_code(text)
            if len(code) < 40:
                continue
            if gate(code, std):
                return {"id": t["id"], "prompt": t["prompt"],
                        "source": code, "std": std, "kind": "winner",
                        "expected": "ok", "samples_used": k + 1,
                        "origin": {"corpus": "harvest-rel8-q8",
                                   "std": std,
                                   "sha256": hashlib.sha256(
                                       code.encode()).hexdigest()}}
        return None

    n_win = 0
    with out.open("a") as fh, ThreadPoolExecutor(max_workers=4) as ex:
        for i, w in enumerate(ex.map(harvest_one, tasks)):
            if w:
                fh.write(json.dumps(w) + "\n")
                fh.flush()
                n_win += 1
            if (i + 1) % 20 == 0:
                print(f"[harvest] {i + 1}/{len(tasks)}, {n_win} winners",
                      flush=True)
    print(f"HARVEST-COMPLETE {n_win}/{len(tasks)} winners -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
