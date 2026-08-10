#!/usr/bin/env python3
"""Smoke: g++ oracle integration — health, known-good/known-bad compile
verdicts, and (optionally) a model baseline on the C++26 probe suite.

  ORACLE_URL=http://<host>:8950 python3 tests/smoke_oracle.py
  ... --model mistralai/Ministral-3-3B-Instruct-2512-BF16   # + GPU eval

Without --model this validates only the oracle path (fast, no GPU).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "traintest"))
from oracle_client import Oracle  # noqa: E402

FAILS = []

GOOD = "#include <cstdio>\nint main() { std::puts(\"ok\"); }\n"
BAD = "int main() { return undeclared_name; }\n"
CXX26 = """#include <meta>
#include <print>
struct Point { int x; double y; };
int main() {
  template for (constexpr auto m :
                std::define_static_array(
                    std::meta::nonstatic_data_members_of(
                        ^^Point, std::meta::access_context::current()))) {
    std::println("{}", std::meta::identifier_of(m));
  }
}
"""


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--url", default=None)
    args = ap.parse_args()

    oracle = Oracle(args.url) if args.url else Oracle()
    try:
        h = oracle.health()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  oracle reachable  {e!r}")
        print("\nSMOKE ORACLE: FAIL (oracle unreachable — set ORACLE_URL)")
        sys.exit(1)
    check("oracle reachable", all(x.get("ok") for x in h),
          f"{len(h)} shard(s), reflection={h[0].get('reflection')}")

    good = oracle.compile(GOOD, run=True)
    check("known-good compiles+runs",
          good.get("ok") and good.get("run_rc") == 0,
          f"{good.get('ms')}ms stdout={good.get('run_stdout', '').strip()!r}")
    bad = oracle.compile(BAD)
    check("known-bad rejected", not bad.get("ok"),
          (bad.get("stderr") or "")[:60].replace("\n", " "))
    cxx26 = oracle.compile(CXX26, run=True)
    check("C++26 reflection compiles+runs (oracle capability)",
          cxx26.get("ok") and cxx26.get("run_rc") == 0,
          f"{cxx26.get('ms')}ms out={cxx26.get('run_stdout', '').strip()!r}")

    if args.model:
        out = REPO / "runs-linux" / "oracle-baseline" / "probes.json"
        cmd = [str(REPO / "scripts" / "run_linux.sh"), "oracle_eval.py",
               "--model", args.model,
               "--suite", str(REPO / "data" / "cpp26_probes.jsonl"),
               "--out", str(out)]
        if args.url:
            cmd += ["--url", args.url]
        p = subprocess.run(cmd, timeout=3600)
        if p.returncode == 0:
            r = json.loads(out.read_text())
            print(f"INFO  baseline C++26 compile rate: "
                  f"{r['compile_pass']}/{r['total']} ({r['rate']:.0%}) — "
                  f"expected near-zero pre-training")
        else:
            check("model probe eval ran", False, f"exit {p.returncode}")

    print()
    if FAILS:
        print(f"SMOKE ORACLE: FAIL ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print("SMOKE ORACLE: PASS")


if __name__ == "__main__":
    main()
