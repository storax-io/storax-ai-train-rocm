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
    pairs = json.loads(Path(src).read_text())
    lines = [json.dumps({"id": f"guard-cpp-{i:02d}", "family": "guard-cpp",
                         "prompt": p["prompt"]})
             for i, p in enumerate(pairs)]
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"guard suite: {len(lines)} tasks -> {out}")


if __name__ == "__main__":
    main()
