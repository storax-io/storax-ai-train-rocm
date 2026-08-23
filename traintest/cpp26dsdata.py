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
import hashlib
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

# Drill share cap WITHIN the new-features QA pool. Drill volume growing
# faster than composition streams regressed composition families and
# guards twice (cycles 2 and 5) even with the macro mix held — the
# internal ratio is load-bearing too. Excess drills are subsampled
# deterministically; their mutations follow their bases.
DRILL_SHARE = float(os.environ.get("CPP26DS_DRILL_SHARE", "0.45"))

# FORMAT BANK (forge eval of v61s2, 2026-08-23: 5/10 on bare C++26
# probes but 0 on EVERY realistically-shaped file-based prompt — C++17
# polyglot, C, C++98 — while the same asks phrased bare succeed. The
# constant wrapper "Only output the code." on five bands became the
# retrieval key; general ability is in the weights, the prompt shape
# gates access). The namebank doctrine applied to the instruction
# surface: wrapper drawn deterministically per record id from a bank of
# realistic shapes, and the TARGET STANDARD IS NAMED (Henri 2026-08-23:
# "standard should be in prompt" — C++23 scoring UNDER C++26 was the
# fingerprint of a model never told which standard it is writing).
FORMAT_BANK = os.environ.get("CPP26DS_FORMAT_BANK", "1") != "0"
_FMT_SEED = os.environ.get("CPP26DS_FORMAT_SEED", "0")
_FMTS = [
    "{p} Only output the code.",
    "{p}",
    "{p} Respond with a single fenced code block, nothing else.",
    "Task: {p}\nTarget: {std}, single translation unit. Code only.",
    "You are working in a {std} codebase. Implement the following in "
    "one file:\n{p}\nReply with the complete file contents in a code "
    "block.",
    "// TODO: {p}\nWrite the full implementation file ({std}). Output "
    "only code.",
    "Issue:\n---\n{p}\n---\nSubmit the complete single-file solution "
    "as one code block ({std}).",
    "{p}\nUse {std}. No explanation.",
    "Implement the following ({std}). Provide the whole program:\n{p}",
    "Complete this assignment: {p}\nAnswer with just the program "
    "({std}).",
]


def _wrap(prompt, rid, std="C++26"):
    if not FORMAT_BANK:
        return prompt + " Only output the code."
    i = int(hashlib.sha256(
        f"{_FMT_SEED}:{rid}".encode()).hexdigest()[:12], 16)
    return _FMTS[i % len(_FMTS)].format(p=prompt, std=std)


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

# Generator's mixture streams (2026-08-11): mixed = C++26 constructs in
# ordinary C++23 shapes (counts as NEW: it teaches the constructs in
# context); baseline = pure C++23; imported = verified STL/libcxx test
# sources (baseline packing). synth-legacy is deliberately EXCLUDED from
# positive training — it compiles but embodies pre-modern style; it
# enters only once paired as modernize edit-triples.
_MIXED = sorted(_rows("synth-mixed.jsonl"), key=lambda r: r["id"])
_BASELINE = sorted(_rows("synth-baseline.jsonl"), key=lambda r: r["id"])
_IMPORTED = sorted((r for r in _rows("imported.jsonl")
                    if r.get("expected") == "ok"), key=lambda r: r["id"])
_GUIDE = _rows("guidelines.jsonl")
# v6 bands (2026-08-21): behavior-verified harvest winners (the model's
# own oracle-adjudicated solutions to its hard tail) and compile-time-
# compute references (Henri's fibonacci archetype — output-verifiable
# consteval computation). Both 100% behavior-verified; absent files =
# empty bands (packs before v6).
_WINNERS = _rows("winners.jsonl")
_CT = _rows("ct.jsonl")
IMPORTED_TEXT_N = int(os.environ.get("CPP26DS_IMPORTED_TEXT", "250"))

# REAL-C imports (Henri 2026-08-23: "especially we need to fix C
# skills" — forge eval scored file-based C at 0 and the pack carried
# ZERO C). Compile-verified corpus records from C origins become packed
# LM text; the C QA side is std_replay below. Point CPP26DS_C_CORPUS at
# a corpus imported.jsonl (LUMI: $STORAX_ROOT/corpus/var/imported.jsonl;
# local master: ~/storax-runs/lumi-corpus/imported.jsonl).
_C_ORIGINS = ("lua", "zlib", "zstd", "brotli", "libuv", "musl", "curl")
C_IMPORT_TEXT_N = int(os.environ.get("CPP26DS_C_TEXT", "150"))


def _c_imports():
    p = os.environ.get("CPP26DS_C_CORPUS", "")
    if not p or not Path(p).exists():
        return []
    out = []
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        origin = str(r.get("origin", "")).lower()
        if (r.get("expected") == "ok" and r.get("source")
                and any(k in origin for k in _C_ORIGINS)
                and "::" not in r["source"]):
            out.append(r["source"])
    return out

_rng = random.Random(SEED)
_SYNTH_SHUF = _SYNTH[:]
_rng.shuffle(_SYNTH_SHUF)


def _fenced(code):
    return f"```cpp\n{code.strip()}\n```"


def article_texts():
    """Raw LM text (packed by the trainer): C++26 synth + mixed stream,
    plus a capped sample of imported STL test sources as baseline, plus
    real-C corpus imports (orthogonal band — see _c_imports)."""
    imp = _IMPORTED[:]
    _rng2 = random.Random(SEED + 1)
    _rng2.shuffle(imp)
    cimp = _c_imports()
    _rng3 = random.Random(SEED + 2)
    _rng3.shuffle(cimp)
    if cimp:
        print(f"real-C imports: {min(len(cimp), C_IMPORT_TEXT_N)} of "
              f"{len(cimp)} as packed text", flush=True)
    return ([r["source"] for r in _SYNTH_SHUF[:SYNTH_TEXT_N]]
            + [r["source"] for r in _MIXED]
            + [r["source"] for r in imp[:IMPORTED_TEXT_N]]
            + cimp[:C_IMPORT_TEXT_N])


def training_texts():
    """Core-Guidelines rule statements (style policy in words)."""
    out = []
    for g in _GUIDE:
        if g.get("prompt"):
            out.append(g["prompt"])
    return out * 2


def _std_replay():
    """Standard-sweep anchors (gen_std_replay.py): the BASE model answers
    everyday tasks per standard — C17, C++98/11/17/20 — each verified by
    the matching gcc/g++ -std= flag, standard named in the prompt at
    birth. The corpus-side answer to the forge zeros on C and older C++
    (Henri: 'corpus should be extended, so c++ older revisions work')."""
    rdir = os.environ.get("TRAINTEST_REPLAY_DIR")
    f = (Path(rdir) / "std_replay.json") if rdir else \
        Path(__file__).resolve().parent / "std_replay.json"
    if not f.exists():
        print("std_replay.json missing — C / older-C++ anchors absent "
              "(run gen_std_replay.py)", flush=True)
        return []
    return json.loads(f.read_text(encoding="utf-8"))


def _cpp_replay():
    """Oracle-filtered self-distillation anchors for ordinary modern C++
    (gen_cpp_replay.py) — counterweight to feature-dense drills, which
    otherwise erode plain C++ (measured: struct-syntax failures and
    composition-family regressions in cycle 2)."""
    rdir = os.environ.get("TRAINTEST_REPLAY_DIR")
    f = (Path(rdir) / "cpp_replay.json") if rdir else \
        Path(__file__).resolve().parent / "cpp_replay.json"
    if not f.exists():
        print("cpp_replay.json missing — plain-C++ erosion likely "
              "(run gen_cpp_replay.py)", flush=True)
        return []
    return json.loads(f.read_text(encoding="utf-8"))


def training_qa_pairs():
    out = []
    # Enforce DRILL_SHARE within the new pool: cap drill bases against
    # non-drill new QA mass, round-robin per family (coverage preserved);
    # mutations follow their kept bases.
    other_new = (len(_TRACES) + SYNTH_QA_N + len(_EDITS) + len(_MIXED)
                 + len(_WINNERS) + len(_CT))
    bases = sorted(_BASES.values(), key=lambda r: r["id"])
    max_bases = int(DRILL_SHARE / (1 - DRILL_SHARE) * max(other_new, 1))
    if len(bases) > max_bases:
        by_fam = {}
        for r in bases:
            by_fam.setdefault(r["family"], []).append(r)
        picked = []
        while len(picked) < max_bases and any(by_fam.values()):
            for fam in sorted(by_fam):
                if by_fam[fam] and len(picked) < max_bases:
                    picked.append(by_fam[fam].pop(0))
        print(f"drill cap: {len(bases)} -> {len(picked)} bases "
              f"(share {DRILL_SHARE}); mutations follow", flush=True)
        bases = picked
    # CLOSED-LOOP FAMILY WEIGHTS (Henri 2026-08-20: per-segment emphasis
    # control, automatic, boiled close). CPP26DS_FAMILY_WEIGHTS is a JSON
    # {family: multiplier} written by the converge controller from the
    # PREVIOUS segment's eval — dose a falling family, taper a saturating
    # one. Applied to drill bases (the controlled surface — where the
    # rel5 saturation law lives); mutations follow their bases. w>1
    # duplicates (round), w<1 subsamples deterministically.
    fw_env = os.environ.get("CPP26DS_FAMILY_WEIGHTS", "")
    if fw_env:
        try:
            fw = json.loads(fw_env)
        except Exception:
            print("FAMILY_WEIGHTS unparseable — ignored:", fw_env[:120],
                  flush=True)
            fw = {}
        if fw:
            wb = []
            for r in bases:
                w = float(fw.get(r["family"], 1.0))
                n = max(0, round(w))
                if w < 1.0:
                    # deterministic subsample: keep by id-hash threshold
                    keep = (hash(r["id"]) % 1000) < int(w * 1000)
                    n = 1 if keep else 0
                wb.extend([r] * max(n, 0))
            print("family weights applied: %d -> %d drill bases (%s)"
                  % (len(bases), len(wb),
                     ",".join("%s=%.2f" % kv for kv in sorted(fw.items()))),
                  flush=True)
            bases = wb
    kept_ids = {r["id"] for r in bases}
    muts = [m for m in _MUTS if m["base_id"] in kept_ids]

    # C++23 baseline QA pool, rebalanced to MIX[1]/MIX[0] of the C++26
    # QA mass: generator's synth-baseline stream + our model-native
    # anchors (duplication only for any remaining shortfall).
    new_mass = (len(bases) + len(_TRACES) + SYNTH_QA_N + len(_EDITS)
                + len(muts) + len(_MIXED) + len(_WINNERS) + len(_CT))
    # baseline prompts carry a generator-era "C++26 program" phrasing;
    # the band trains the CURRENT standard — name it truthfully (the
    # C++23-under-C++26 forge fingerprint is exactly mislabeled-standard
    # training)
    base_pool = ([(_wrap(r["prompt"].replace("C++26 program",
                                             "C++23 program"),
                         r["id"], "C++23"),
                   _fenced(r["source"])) for r in _BASELINE]
                 + [(r["prompt"], _fenced(r["code"]))
                    for r in _cpp_replay()]
                 + [(_wrap(r["prompt"], f"std:{i}", r.get("std", "C17")),
                     _fenced(r["code"]))
                    for i, r in enumerate(_std_replay())])
    if base_pool:
        target = max(1, int(new_mass * MIX[1] / MIX[0]))
        reps = max(1, round(target / len(base_pool)))
        print(f"mix: c++26={new_mass} baseline_target={target} "
              f"({len(base_pool)} unique x{reps}); general share is "
              f"replay.json in train.py", flush=True)
        for _ in range(reps):
            out.extend(base_pool)

    # Style-repair pairs from guidelines: bad code + clang-tidy findings
    # -> conforming rewrite, same message shape as compile-repair.
    for g in _GUIDE:
        if g.get("bad") and g.get("source"):
            findings = "\n".join(str(f) for f in
                                 (g.get("bad_style_findings") or []))[:600]
            out.append((
                "This code violates the project style profile. "
                "clang-tidy findings:\n" + findings
                + "\nRewrite it to conform. Only output the corrected "
                "code.\n\n```cpp\n" + g["bad"].strip() + "\n```",
                _fenced(g["source"])))
    for r in bases:
        out.append((_wrap(r["prompt"], r["id"]), _fenced(r["source"])))
    for r in _TRACES:
        out.append((_wrap(r["prompt"], r.get("id", r["prompt"])),
                    _fenced(r["source"])))
    for r in _WINNERS:
        out.append((_wrap(r["prompt"], r.get("id", r["prompt"])),
                    _fenced(r["source"])))
    for r in _CT:
        out.append((_wrap(r["prompt"], r.get("id", r["prompt"])),
                    _fenced(r["source"])))
    for r in _SYNTH_SHUF[SYNTH_TEXT_N:SYNTH_TEXT_N + SYNTH_QA_N]:
        out.append((_wrap(r["prompt"], r["id"]), _fenced(r["source"])))
    for r in _EDITS:
        out.append((f"{r['instruction']}\n\n```cpp\n{r['base'].strip()}\n```"
                    "\nOnly output the edited code.",
                    _fenced(r["edited"])))
    # Repair pairs in the exact message format oracle_eval --repair sends.
    for r in muts:
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
        "winners": len(_WINNERS), "ct": len(_CT),
        "synth_total": len(_SYNTH),
        "packed_texts": min(SYNTH_TEXT_N, len(_SYNTH)),
        "qa_pairs": len(training_qa_pairs()),
    }, indent=1))
