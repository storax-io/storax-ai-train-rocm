"""Build a compiler-verified retention-guard suite from cpp_replay.json.

The base model answered these ordinary-C++ tasks correctly (oracle-gated
at generation time). A trained checkpoint that stops compiling on them
has eroded plain-C++ competence — the domain analogue of the phase-0
retention guard, measured by the compiler instead of string match.

    python3 tools/make_guard_suite.py <cpp_replay.json> <out.jsonl>
"""
import json
import sys
from pathlib import Path


def main():
    src, out = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    pairs = json.loads(Path(src).read_text())
    if len(pairs) > cap:
        # replay scaled to 1500+ pairs (retention-band fix): TRAINING uses
        # them all; the guard EVAL samples a deterministic subset so the
        # verdict stays inside the eval SLO (~64 tasks ~= 2-3 min)
        from random import Random
        pairs = Random("guard-suite").sample(pairs, cap)
    lines = [json.dumps({"id": f"guard-cpp-{i:03d}", "family": "guard-cpp",
                         "prompt": p["prompt"]})
             for i, p in enumerate(pairs)]
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"guard suite: {len(lines)} tasks (of {len(json.loads(Path(src).read_text()))}) -> {out}")


if __name__ == "__main__":
    main()
