"""Data provider bridging the storax-dataset-cpp26 pipeline into this
harness (--data cpp26ds). Consumes the pipeline's stage outputs (var/*.jsonl
or a release directory):

  level0.jsonl   expected=ok  -> instruction->code QA (grammar drills)
                 expected=fail-> repair pairs: broken source + REAL GCC
                                 diagnostics + the linked base as the fix
  synth.jsonl    -> packed LM text (sample) + instruction->code QA (sample)
  filtered.jsonl -> teacher traces as instruction->code QA
  edits.jsonl    -> edit-instruction QA (base + instruction -> edited)

Every row was oracle-verified by the pipeline before it got here; this
module adds no unverified text. Set CPP26DS_DIR to point at a different
checkout or release dir.
"""
import json
import os
import random
from pathlib import Path

SEED = 0

_DIR = Path(os.environ.get(
    "CPP26DS_DIR",
    Path(__file__).resolve().parent.parent.parent
    / "storax-dataset-cpp26" / "var"))

SYNTH_TEXT_N = int(os.environ.get("CPP26DS_SYNTH_TEXT", "1200"))
SYNTH_QA_N = int(os.environ.get("CPP26DS_SYNTH_QA", "300"))

# Training mixture policy: new-features / current-standard / general
# language skills. Enforced by duplicating or subsampling the baseline
# pool relative to the C++26 pool; the general share is the chat replay
# handled by train.py (its count is included in the report only).
MIX = tuple(int(x) for x in
            os.environ.get("CPP26DS_MIX", "70,20,10").split(","))


def _rows(name):
    f = _DIR / name
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()
            if l.strip()]


_L0 = _rows("level0.jsonl")
_BASES = {r["id"]: r for r in _L0 if r.get("expected") == "ok"}
_MUTS = [r for r in _L0 if r.get("expected") == "fail"
         and r.get("base_id") in _BASES]
_SYNTH = sorted(_rows("synth.jsonl"), key=lambda r: r["id"])
_TRACES = [r for r in _rows("filtered.jsonl") if r.get("source")]
_EDITS = _rows("edits.jsonl")

_rng = random.Random(SEED)
_SYNTH_SHUF = _SYNTH[:]
_rng.shuffle(_SYNTH_SHUF)


def _fenced(code):
    return f"```cpp\n{code.strip()}\n```"


def article_texts():
    """Synth-stream programs as raw LM text (packed by the trainer)."""
    return [r["source"] for r in _SYNTH_SHUF[:SYNTH_TEXT_N]]


def training_texts():
    return []


def _cpp_replay():
    """Oracle-filtered self-distillation anchors for ordinary modern C++
    (gen_cpp_replay.py) — counterweight to feature-dense drills, which
    otherwise erode plain C++ (measured: struct-syntax failures and
    composition-family regressions in cycle 2)."""
    f = Path(__file__).resolve().parent / "cpp_replay.json"
    if not f.exists():
        print("cpp_replay.json missing — plain-C++ erosion likely "
              "(run gen_cpp_replay.py)", flush=True)
        return []
    return json.loads(f.read_text(encoding="utf-8"))


def training_qa_pairs():
    out = []
    # C++23 baseline pool, rebalanced to MIX[1]/MIX[0] of the C++26 QA
    # mass (duplication is fine: anchors are few and oracle-verified).
    new_mass = (len(_BASES) + len(_TRACES) + SYNTH_QA_N + len(_EDITS)
                + len(_MUTS))
    base_pool = _cpp_replay()
    if base_pool:
        target = max(1, int(new_mass * MIX[1] / MIX[0]))
        reps = max(1, round(target / len(base_pool)))
        print(f"mix: c++26={new_mass} baseline_target={target} "
              f"({len(base_pool)} anchors x{reps}); general share is "
              f"replay.json in train.py", flush=True)
        for _ in range(reps):
            for r in base_pool:
                out.append((r["prompt"], _fenced(r["code"])))
    for r in _BASES.values():
        out.append((r["prompt"] + " Only output the code.",
                    _fenced(r["source"])))
    for r in _TRACES:
        out.append((r["prompt"] + " Only output the code.",
                    _fenced(r["source"])))
    for r in _SYNTH_SHUF[SYNTH_TEXT_N:SYNTH_TEXT_N + SYNTH_QA_N]:
        out.append((r["prompt"] + " Only output the code.",
                    _fenced(r["source"])))
    for r in _EDITS:
        out.append((f"{r['instruction']}\n\n```cpp\n{r['base'].strip()}\n```"
                    "\nOnly output the edited code.",
                    _fenced(r["edited"])))
    # Repair pairs in the exact message format oracle_eval --repair sends.
    for r in _MUTS:
        stderr = "\n".join(r.get("diagnostics") or [])[:800]
        fixed = _BASES[r["base_id"]]["source"]
        out.append((
            "That does not compile. Compiler output:\n" + stderr
            + "\nFix the program. Only output the corrected code.\n\n"
            + "The program was:\n" + _fenced(r["source"]),
            _fenced(fixed)))
    return out


def think_qa_pairs():
    return []


if __name__ == "__main__":
    print(json.dumps({
        "dir": str(_DIR),
        "level0_bases": len(_BASES), "repair_pairs": len(_MUTS),
        "teacher_traces": len(_TRACES), "edits": len(_EDITS),
        "synth_total": len(_SYNTH),
        "packed_texts": min(SYNTH_TEXT_N, len(_SYNTH)),
        "qa_pairs": len(training_qa_pairs()),
    }, indent=1))
