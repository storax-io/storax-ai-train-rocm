"""C++26 capability corpus (oracle-verified) — same provider interface as
facts.py so train.py can switch datasets with --data cpp26.

Every code example in data/cpp26_corpus.json was compiled AND executed by
the g++ oracle before entering the corpus (tools/build_cpp26_corpus.py);
the probe suite (data/cpp26_probes.jsonl) stays held out for eval.
"""
import json
from pathlib import Path

SEED = 0


def _load():
    here = Path(__file__).resolve().parent
    for cand in (here / "cpp26_corpus.json",
                 here.parent / "data" / "cpp26_corpus.json"):
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8"))
    raise FileNotFoundError("cpp26_corpus.json — run tools/build_cpp26_corpus.py")


_DATA = _load()
EXAMPLES = _DATA["examples"]
STATEMENTS = _DATA["statements"]


def article_texts():
    return []


def training_texts():
    """API-fact statements + task->solution walkthroughs (full-loss)."""
    out = list(STATEMENTS) * 2
    for e in EXAMPLES:
        out.append(f"Task: {e['task']}\nC++26 solution:\n{e['code']}")
    return out


def training_qa_pairs():
    """(instruction, fenced code) — teaches answering with a code block,
    matching what oracle_eval.py extracts and compiles. Repair pairs use
    the same message format oracle_eval --repair sends at inference, so
    the production compile-fix loop is in-distribution."""
    out = []
    for e in EXAMPLES:
        fenced = f"```cpp\n{e['code']}\n```"
        for prompt in (
            e["task"] + " Only output the code.",
            "Write a complete C++26 program. " + e["task"],
            f"Using GCC 16.1 C++26 (reflection/contracts enabled): {e['task']}",
        ):
            out.append((prompt, fenced))
    for r in _DATA.get("repairs", []):
        prompt = ("That does not compile. Compiler output:\n" + r["stderr"]
                  + "\nFix the program. Only output the corrected code.\n\n"
                  + "The program was:\n```cpp\n" + r["broken"] + "\n```")
        out.append((prompt, f"```cpp\n{r['fixed']}\n```"))
        out.append((
            "Fix this C++26 program that fails to compile:\n```cpp\n"
            + r["broken"] + "\n```\nCompiler error:\n" + r["stderr"]
            + "\nOnly output the corrected code.",
            f"```cpp\n{r['fixed']}\n```"))
    return out


def think_qa_pairs():
    return []
