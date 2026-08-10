#!/usr/bin/env python3
"""Dynamic C++26 training rounds: train -> oracle-eval held-out probes ->
add ORACLE-VERIFIED remedial exemplars for the failing error classes ->
retrain, until the probe suite passes 10/10 (or remedials are exhausted).

Discipline:
  - Remedials teach the ERROR CLASS (different task, different code) —
    never the probe task itself; the probe suite stays held out.
  - Every remedial is compiled+run by the oracle before it may teach.
  - Guards every round: retention >= 0.9, control == 0; a retention miss
    softens the schedule (3 -> 2 epochs) for subsequent rounds.

Run from repo root (oracle + GPU required):
    python3 tools/cpp26_loop.py [--max-rounds 4]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "traintest"))
from oracle_client import Oracle  # noqa: E402

BASE = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
SYS = "You are a helpful assistant."
CORPUS = REPO / "data" / "cpp26_corpus.json"
PROBES = REPO / "data" / "cpp26_probes.jsonl"

# probe id -> remedial exemplars for its error class (id, task, code).
REMEDIALS = {
    "reflect-enum-to-string": [(
        "enum-name-string-conv",
        "Implement enum-to-string with C++26 reflection, returning "
        "std::string_view from the consteval lookup and constructing "
        "std::string from it explicitly where an owned string is needed "
        "(identifier_of yields a string_view, which does not convert "
        "implicitly to std::string).",
        r"""
#include <meta>
#include <print>
#include <string>
#include <string_view>
enum class Fruit { Apple, Pear, Plum };
template <typename Ee>
constexpr std::string_view enum_sv(Ee v) {
  template for (constexpr auto e :
                std::define_static_array(std::meta::enumerators_of(^^Ee))) {
    if (v == [:e:]) return std::meta::identifier_of(e);
  }
  return "?";
}
template <typename Ee>
std::string enum_str(Ee v) { return std::string(enum_sv(v)); }
int main() {
  std::println("{}", enum_sv(Fruit::Pear));
  std::println("{}", enum_str(Fruit::Plum));
}
""")],
    "reflect-type-name": [(
        "type-name-constexpr-ctx",
        "Print type names obtained via C++26 reflection. Reflection "
        "queries like identifier_of and display_string_of are consteval: "
        "store their results in constexpr variables first, then use those "
        "at runtime.",
        r"""
#include <meta>
#include <print>
#include <string_view>
struct Widget {};
int main() {
  constexpr std::string_view tn = std::meta::identifier_of(^^Widget);
  constexpr std::string_view dn =
      std::meta::display_string_of(^^unsigned long);
  std::println("{} {}", tn, dn);
}
""")],
    "template-for-expansion": [(
        "template-for-instance-splice",
        "Use 'template for' to iterate a struct's data members and print "
        "each value. The member splice must be applied to an INSTANCE "
        "(obj.[:m:]) — splicing on the type name is an error.",
        r"""
#include <meta>
#include <print>
struct Sensor { int id = 5; double reading = 2.5; float scale = 1.5f; };
int main() {
  Sensor s{};
  template for (constexpr auto m :
                std::define_static_array(
                    std::meta::nonstatic_data_members_of(
                        ^^Sensor, std::meta::access_context::current()))) {
    std::println("{} = {}", std::meta::identifier_of(m), s.[:m:]);
  }
}
""")],
}

REMEDIAL_STATEMENTS = {
    "reflect-enum-to-string":
        "std::meta::identifier_of returns std::string_view; an owned "
        "std::string requires explicit construction: "
        "std::string(std::meta::identifier_of(e)).",
    "reflect-type-name":
        "Consteval reflection queries must run in constant-expression "
        "contexts: store identifier_of/display_string_of results in "
        "constexpr variables before using them at runtime.",
    "template-for-expansion":
        "A data member's value is read by splicing on an instance, "
        "obj.[:m:]; writing TypeName.[:m:] is an error.",
}


def sh(script, *args, timeout=3600):
    p = subprocess.run([str(REPO / "scripts" / "run_linux.sh"), script, *args],
                      capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        print(p.stdout[-1200:], p.stderr[-1200:])
        raise RuntimeError(f"{script} exited {p.returncode}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=4)
    args = ap.parse_args()

    oracle = Oracle()
    corpus = json.loads(CORPUS.read_text())
    added = set()
    epochs = 3
    history = []

    for rnd in range(1, args.max_rounds + 1):
        out = REPO / "runs-linux" / f"cpp26-loop-r{rnd}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== round {rnd}: {len(corpus['examples'])} exemplars, "
              f"{epochs} epochs ===", flush=True)
        t0 = time.time()
        sh("train.py", "--model", BASE, "--data", "cpp26",
           "--system", SYS, "--freeze",
           "vision_tower,multi_modal_projector,embed_tokens,lm_head",
           "--epochs", str(epochs), "--lr", "2e-5",
           "--out", str(out), "--save-model")
        sh("oracle_eval.py", "--model", str(out / "model"),
           "--system", SYS, "--suite", str(PROBES),
           "--out", str(out / "oracle.json"))
        sh("evaluate.py", "--model", str(out / "model"), "--system", SYS,
           "--sets", "retention,control",
           "--out", str(out / "guards.json"))

        oracle_r = json.loads((out / "oracle.json").read_text())
        guards = json.loads((out / "guards.json").read_text())["sets"]
        failed = [x["id"] for x in oracle_r["results"] if not x["ok"]]
        rec = {"round": rnd, "epochs": epochs,
               "probes": f"{oracle_r['compile_pass']}/{oracle_r['total']}",
               "failed": failed,
               "retention": guards["retention"]["accuracy"],
               "control": guards["control"]["accuracy"],
               "minutes": round((time.time() - t0) / 60, 1)}
        history.append(rec)
        print("ROUND " + json.dumps(rec), flush=True)

        guards_ok = (guards["retention"]["accuracy"] >= 0.9
                     and guards["control"]["accuracy"] <= 0.2)
        if not failed and guards_ok:
            print(f"\nDONE: 10/10 with guards green in round {rnd}")
            break
        if not guards_ok:
            epochs = max(2, epochs - 1)
            print(f"retention/control miss -> epochs {epochs} next round")

        new = 0
        for pid in failed:
            for rid, task, code in REMEDIALS.get(pid, []):
                if rid in added:
                    continue
                v = oracle.compile(code, run=True)
                if v.get("ok") and v.get("run_rc") == 0:
                    corpus["examples"].append(
                        {"id": rid, "task": task, "code": code.strip(),
                         "run_stdout": v.get("run_stdout", "")})
                    stmt = REMEDIAL_STATEMENTS.get(pid)
                    if stmt and stmt not in corpus["statements"]:
                        corpus["statements"].append(stmt)
                    added.add(rid)
                    new += 1
                    print(f"  + remedial {rid} (oracle-verified)")
                else:
                    print(f"  ! remedial {rid} REJECTED by oracle — skipped")
        if new == 0 and guards_ok:
            print("\nSTOP: failures remain but no unused remedials "
                  f"({failed}) — needs new remedial authoring")
            break
        if new:
            CORPUS.write_text(json.dumps(corpus, indent=1))

    (REPO / "runs-linux" / "cpp26-loop-report.json").write_text(
        json.dumps(history, indent=2))
    print("\nHISTORY: " + json.dumps(history))


if __name__ == "__main__":
    main()
