#!/usr/bin/env python3
"""Build an ORACLE-VERIFIED C++26 training corpus.

Every exemplar program below is compiled AND executed through the g++
oracle (GCC 16.1, -std=c++26 -freflection -fcontracts); anything that
fails is rejected loudly. Only verified code enters the corpus — the
training set cannot teach syntax the compiler rejects.

Output: data/cpp26_corpus.json
  {"examples": [{id, task, code, run_stdout}], "statements": [...]}

Exemplars deliberately differ from data/cpp26_probes.jsonl tasks — the
probe suite stays held-out for before/after evaluation.

Usage: ORACLE_URL=http://<host>:8950 python3 tools/build_cpp26_corpus.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "traintest"))
from oracle_client import Oracle

E = []  # (id, task_instruction, code)

E.append(("members-print", "Use C++26 reflection to iterate the nonstatic data members of a struct and print each member's name.", r"""
#include <meta>
#include <print>
struct Config { int retries; double timeout; bool verbose; };
int main() {
  template for (constexpr auto m :
                std::define_static_array(
                    std::meta::nonstatic_data_members_of(
                        ^^Config, std::meta::access_context::current()))) {
    std::println("{}", std::meta::identifier_of(m));
  }
}
"""))

E.append(("member-values", "Use C++26 reflection with splicing to print each member's name and value for a struct instance.", r"""
#include <meta>
#include <print>
struct Point3 { int x = 1; int y = 2; int z = 3; };
int main() {
  Point3 p;
  template for (constexpr auto m :
                std::define_static_array(
                    std::meta::nonstatic_data_members_of(
                        ^^Point3, std::meta::access_context::current()))) {
    std::println("{} = {}", std::meta::identifier_of(m), p.[:m:]);
  }
}
"""))

E.append(("enum-to-string", "Implement enum-to-string in C++26 by reflecting over the enumerators.", r"""
#include <meta>
#include <print>
#include <string_view>
enum class Level { Debug, Info, Warn, Error };
template <typename Ee>
constexpr std::string_view enum_name(Ee value) {
  template for (constexpr auto e :
                std::define_static_array(std::meta::enumerators_of(^^Ee))) {
    if (value == [:e:]) return std::meta::identifier_of(e);
  }
  return "<unknown>";
}
int main() { std::println("{}", enum_name(Level::Warn)); }
"""))

E.append(("type-name", "Print a type's name using C++26 reflection.", r"""
#include <meta>
#include <print>
int main() {
  constexpr auto r = ^^double;
  std::println("{}", std::meta::display_string_of(r));
}
"""))

E.append(("struct-to-tuple", "Convert a struct to a std::tuple of its member values using C++26 reflection.", r"""
#include <meta>
#include <print>
#include <tuple>
struct Pair2 { int first = 7; int second = 9; };
template <typename T>
constexpr auto struct_to_tuple(const T& t) {
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    constexpr auto ctx = std::meta::access_context::current();
    return std::make_tuple(
        t.[:std::meta::nonstatic_data_members_of(^^T, ctx)[I]:]...);
  }(std::make_index_sequence<
        std::meta::nonstatic_data_members_of(
            ^^T, std::meta::access_context::current()).size()>{});
}
int main() {
  auto tup = struct_to_tuple(Pair2{});
  std::println("{} {}", std::get<0>(tup), std::get<1>(tup));
}
"""))

E.append(("contracts-precondition", "Write a C++26 function with pre and post contract assertions.", r"""
#include <print>
int clamp_positive(int x)
  pre (x > -1000)
  post (r : r >= 0)
{
  return x < 0 ? 0 : x;
}
int main() { std::println("{}", clamp_positive(-5)); }
"""))

E.append(("contract-assert", "Use the C++26 contract_assert statement to check an invariant inside a function.", r"""
#include <print>
int midpoint(int a, int b) {
  contract_assert(a <= b);
  return a + (b - a) / 2;
}
int main() { std::println("{}", midpoint(2, 10)); }
"""))

E.append(("fixed-string-nttp", "Create a consteval-friendly fixed_string usable as a non-type template parameter.", r"""
#include <print>
#include <cstddef>
template <std::size_t N>
struct fixed_string {
  char data[N]{};
  consteval fixed_string(const char (&s)[N]) {
    for (std::size_t i = 0; i < N; ++i) data[i] = s[i];
  }
};
template <fixed_string Name>
struct Named {
  static constexpr const char* name() { return Name.data; }
};
int main() { std::println("{}", Named<"alpha">::name()); }
"""))

E.append(("constexpr-vector", "Use std::vector inside a constexpr function and check the result with static_assert.", r"""
#include <print>
#include <vector>
#include <numeric>
constexpr int sum_squares(int n) {
  std::vector<int> v(n);
  std::iota(v.begin(), v.end(), 1);
  int s = 0;
  for (int x : v) s += x * x;
  return s;
}
static_assert(sum_squares(3) == 14);
int main() { std::println("{}", sum_squares(4)); }
"""))

E.append(("reflect-member-count", "Count the nonstatic data members of a struct at compile time via reflection.", r"""
#include <meta>
#include <print>
struct Wide { int a; int b; int c; int d; };
int main() {
  constexpr auto n = std::meta::nonstatic_data_members_of(
      ^^Wide, std::meta::access_context::current()).size();
  static_assert(n == 4);
  std::println("{}", n);
}
"""))

E.append(("enum-count", "List all enumerator names of an enum via reflection.", r"""
#include <meta>
#include <print>
enum class Mode { Read, Write, Append };
int main() {
  template for (constexpr auto e :
                std::define_static_array(std::meta::enumerators_of(^^Mode))) {
    std::println("{}", std::meta::identifier_of(e));
  }
}
"""))

E.append(("splice-static-member", "Access a struct member through a stored reflection using splicing.", r"""
#include <meta>
#include <print>
struct Holder { int payload = 123; };
int main() {
  constexpr auto r = std::meta::nonstatic_data_members_of(
      ^^Holder, std::meta::access_context::current())[0];
  Holder h;
  std::println("{}", h.[:r:]);
}
"""))

STATEMENTS = [
    "In C++26, the reflection operator ^^ applied to a type, like ^^int, yields a constexpr value of type std::meta::info.",
    "std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::current()) returns the reflections of T's nonstatic data members in C++26.",
    "std::meta::identifier_of(r) returns the declared name of the entity reflected by r as a string_view; it is consteval.",
    "The C++26 splice syntax [:r:] turns a reflection r back into the entity it reflects; obj.[:member_refl:] accesses that member on obj.",
    "C++26 expansion statements use the syntax 'template for (constexpr auto x : range)': the body is instantiated once per element at compile time. Ranges of std::meta::info must be materialized with std::define_static_array.",
    "std::meta::enumerators_of(^^E) yields reflections of all enumerators of enum E; combined with [:e:] and identifier_of it implements enum-to-string without macros.",
    "C++26 contracts attach pre(condition) and post(r : condition-on-r) to a function declaration; violations are checked at runtime under -fcontracts with the enforce semantic.",
    "contract_assert(expr); is the C++26 statement form of a contract check inside a function body.",
    "A structural type with a consteval constructor, like a fixed_string<N> holding a char array, can be used as a non-type template parameter in C++26.",
    "Reflection queries and <meta> facilities are consteval: they execute during compilation, so their results can feed static_assert and template arguments.",
    "std::meta::display_string_of(r) produces a human-readable description of the reflected entity, e.g. the type name for a type reflection.",
    "GCC's C++26 reflection is enabled with -std=c++26 -freflection; contracts with -fcontracts (and a contract-evaluation semantic such as enforce).",
]


def main():
    oracle = Oracle()
    print("oracle:", oracle.health()[0].get("version", "?"))
    kept, failed = [], []
    for eid, task, code in E:
        v = oracle.compile(code, run=True)
        ok = v.get("ok") and v.get("run_rc") == 0
        print(f"{'PASS' if ok else 'FAIL'}  {eid}  ({v.get('ms')}ms)")
        if ok:
            kept.append({"id": eid, "task": task, "code": code.strip(),
                         "run_stdout": v.get("run_stdout", "")})
        else:
            failed.append(eid)
            print((v.get("stderr") or "")[:400])
    dest = Path(__file__).resolve().parent.parent / "data" / "cpp26_corpus.json"
    dest.write_text(json.dumps(
        {"examples": kept, "statements": STATEMENTS}, indent=1))
    print(f"\nwrote {dest}: {len(kept)} verified examples, "
          f"{len(STATEMENTS)} statements"
          + (f"  REJECTED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
