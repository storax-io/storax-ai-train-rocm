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

# Second wave: 2-3 independent exemplars per concept. One exemplar per
# concept left probe results flipping between rounds (~80% per-probe
# reliability); diversity is what turns memorized instances into
# generalized patterns.

E.append(("enum-name-string-conv",
          "Implement enum-to-string returning std::string_view from the consteval lookup, constructing std::string explicitly where an owned string is needed.", r"""
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
"""))

E.append(("type-name-constexpr-ctx",
          "Print type names via reflection; store consteval query results in constexpr variables before runtime use.", r"""
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
"""))

E.append(("type-name-user-types",
          "Use reflection to print the names of two user-defined types.", r"""
#include <meta>
#include <print>
#include <string_view>
class Engine {};
struct Chassis {};
int main() {
  constexpr std::string_view a = std::meta::identifier_of(^^Engine);
  constexpr std::string_view b = std::meta::identifier_of(^^Chassis);
  std::println("{} {}", a, b);
}
"""))

E.append(("template-for-instance-splice",
          "'template for' over a struct's members printing values — splice on an INSTANCE (obj.[:m:]), never on the type name.", r"""
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
"""))

E.append(("template-for-sum",
          "Sum all int members of a struct with a 'template for' expansion over its reflected members.", r"""
#include <meta>
#include <print>
struct Score { int a = 10; int b = 20; int c = 12; };
int main() {
  Score sc{};
  int total = 0;
  template for (constexpr auto m :
                std::define_static_array(
                    std::meta::nonstatic_data_members_of(
                        ^^Score, std::meta::access_context::current()))) {
    total += sc.[:m:];
  }
  std::println("{}", total);
}
"""))

E.append(("contracts-abs",
          "Write int iabs(int) with a postcondition guaranteeing a nonnegative result, in C++26 contract syntax.", r"""
#include <print>
int iabs(int x)
  post (r : r >= 0)
{
  return x < 0 ? -x : x;
}
int main() { std::println("{}", iabs(-42)); }
"""))

E.append(("contracts-div",
          "Write safe integer division with preconditions in C++26 contract syntax: nonzero divisor, nonnegative dividend.", r"""
#include <print>
int safe_div(const int a, const int b)
  pre (b != 0)
  pre (a >= 0)
  post (r : r * b <= a)
{
  return a / b;
}
int main() { std::println("{}", safe_div(17, 5)); }
"""))

E.append(("contract-assert-loop",
          "Use contract_assert inside a loop to check an invariant while accumulating.", r"""
#include <print>
int sum_to(int n) {
  int s = 0;
  for (int i = 1; i <= n; ++i) {
    s += i;
    contract_assert(s >= i);
  }
  return s;
}
int main() { std::println("{}", sum_to(10)); }
"""))

E.append(("fixed-string-compare",
          "Give a consteval fixed_string NTTP type an operator== and select behavior by comparing template arguments at compile time.", r"""
#include <print>
#include <cstddef>
template <std::size_t N>
struct fixed_string {
  char data[N]{};
  consteval fixed_string(const char (&s)[N]) {
    for (std::size_t i = 0; i < N; ++i) data[i] = s[i];
  }
};
template <std::size_t N, std::size_t M>
consteval bool same(const fixed_string<N>& a, const fixed_string<M>& b) {
  if (N != M) return false;
  for (std::size_t i = 0; i < N; ++i)
    if (a.data[i] != b.data[i]) return false;
  return true;
}
template <fixed_string Tag>
constexpr int channel() {
  if constexpr (same(Tag, fixed_string{"fast"})) return 1;
  else return 0;
}
int main() { std::println("{} {}", channel<"fast">(), channel<"slow">()); }
"""))

E.append(("fixed-string-size",
          "Extend a consteval fixed_string NTTP with a size() member and use it from a template.", r"""
#include <print>
#include <cstddef>
template <std::size_t N>
struct fixed_string {
  char data[N]{};
  consteval fixed_string(const char (&s)[N]) {
    for (std::size_t i = 0; i < N; ++i) data[i] = s[i];
  }
  static consteval std::size_t size() { return N - 1; }
};
template <fixed_string S>
constexpr std::size_t len = S.size();
int main() { std::println("{}", len<"reflection">); }
"""))

E.append(("constexpr-string-build",
          "Build a std::string inside a constexpr function and validate its size with static_assert.", r"""
#include <print>
#include <string>
constexpr std::size_t joined_len() {
  std::string s;
  for (int i = 0; i < 4; ++i) s += "ab";
  return s.size();
}
static_assert(joined_len() == 8);
int main() { std::println("{}", joined_len()); }
"""))

E.append(("splice-second-member",
          "Access the SECOND data member of a struct through an indexed reflection and splicing on an instance.", r"""
#include <meta>
#include <print>
struct Pairish { int first = 4; int second = 11; };
int main() {
  constexpr auto members = std::define_static_array(
      std::meta::nonstatic_data_members_of(
          ^^Pairish, std::meta::access_context::current()));
  Pairish p;
  std::println("{}", p.[:members[1]:]);
}
"""))

E.append(("struct-to-tuple-three",
          "Generic struct_to_tuple over a three-member struct via reflection; print all three.", r"""
#include <meta>
#include <print>
#include <tuple>
struct Trio { int a = 1; double b = 2.5; long c = 9; };
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
  auto tup = struct_to_tuple(Trio{});
  std::println("{} {} {}", std::get<0>(tup), std::get<1>(tup),
               std::get<2>(tup));
}
"""))

E.append(("members-print-count",
          "Print a struct's member names and then the member count, both from reflection.", r"""
#include <meta>
#include <print>
struct Job { int prio; long id; bool active; };
int main() {
  constexpr auto ctx = std::meta::access_context::current();
  template for (constexpr auto m :
                std::define_static_array(
                    std::meta::nonstatic_data_members_of(^^Job, ctx))) {
    std::println("{}", std::meta::identifier_of(m));
  }
  constexpr auto count =
      std::meta::nonstatic_data_members_of(^^Job, ctx).size();
  std::println("count={}", count);
}
"""))

E.append(("contracts-const-param-root",
          "Integer root with a precondition on the argument and a postcondition referencing it. A value parameter used in a postcondition must be declared const — write const int x even when a sketch shows plain int.", r"""
#include <print>
int iroot(const int x)
  pre (x >= 0)
  post (r : r * r <= x)
{
  int r = 0;
  while ((r + 1) * (r + 1) <= x) ++r;
  return r;
}
int main() { std::println("{}", iroot(30)); }
"""))

E.append(("constexpr-vector-return-scalar",
          "A constexpr function may build and transform a std::vector internally, but must return a non-allocating scalar — allocations cannot escape constant evaluation.", r"""
#include <print>
#include <vector>
constexpr int max_tripled() {
  std::vector<int> v{3, 9, 4};
  for (auto& x : v) x *= 3;
  int m = 0;
  for (int x : v)
    if (x > m) m = x;
  return m;
}
static_assert(max_tripled() == 27);
int main() { std::println("{}", max_tripled()); }
"""))

E.append(("template-for-single-letter-struct",
          "Iterate members of a struct with a short type name using 'template for'. Even when the struct is named like a variable (Q), create an instance (Q q{};) and splice on the instance: q.[:m:].", r"""
#include <meta>
#include <print>
struct Q { int a = 7; double b = 0.5; };
int main() {
  Q q{};
  template for (constexpr auto m :
                std::define_static_array(
                    std::meta::nonstatic_data_members_of(
                        ^^Q, std::meta::access_context::current()))) {
    std::println("{} = {}", std::meta::identifier_of(m), q.[:m:]);
  }
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
    "std::meta::identifier_of returns std::string_view; an owned std::string requires explicit construction: std::string(std::meta::identifier_of(e)).",
    "Consteval reflection queries must run in constant-expression contexts: store identifier_of/display_string_of results in constexpr variables before using them at runtime.",
    "A data member's value is read by splicing on an instance, obj.[:m:]; writing TypeName.[:m:] is an error.",
    "Contract pre/post conditions appear between the function declarator and the body: int f(int x) pre (x > 0) post (r : r >= x) { ... }.",
    "A value parameter referenced in a postcondition must be declared const: int div(const int a, const int b) post (r : r * b <= a).",
    "If a task sketches int f(int x) but the postcondition mentions x, the compiling signature is int f(const int x) — correctness overrides the sketch.",
    "A constexpr function can use std::vector internally but must not return it; return the computed scalar (sum, max, size) instead.",
    "Splice member access always needs an instance: declare one (V v{};) and write v.[:m:], never V.[:m:].",
]


# (broken, fixed) pairs teaching REPAIR: read a compiler error, fix the
# program. Broken variants embody the recurring error classes; the real
# stderr is captured from the oracle at build time. Both sides verified:
# broken MUST fail, fixed MUST compile and run.
BREAK_FIX = [
    ("fix-const-contract", r"""
#include <print>
int iroot(int x)
  pre (x >= 0)
  post (r : r * r <= x)
{
  int r = 0;
  while ((r + 1) * (r + 1) <= x) ++r;
  return r;
}
int main() { std::println("{}", iroot(30)); }
""", r"""
#include <print>
int iroot(const int x)
  pre (x >= 0)
  post (r : r * r <= x)
{
  int r = 0;
  while ((r + 1) * (r + 1) <= x) ++r;
  return r;
}
int main() { std::println("{}", iroot(30)); }
"""),
    ("fix-type-splice", r"""
#include <meta>
#include <print>
struct Sensor { int id = 5; double reading = 2.5; };
int main() {
  template for (constexpr auto m :
                std::define_static_array(
                    std::meta::nonstatic_data_members_of(
                        ^^Sensor, std::meta::access_context::current()))) {
    std::println("{} = {}", std::meta::identifier_of(m), Sensor.[:m:]);
  }
}
""", r"""
#include <meta>
#include <print>
struct Sensor { int id = 5; double reading = 2.5; };
int main() {
  Sensor s{};
  template for (constexpr auto m :
                std::define_static_array(
                    std::meta::nonstatic_data_members_of(
                        ^^Sensor, std::meta::access_context::current()))) {
    std::println("{} = {}", std::meta::identifier_of(m), s.[:m:]);
  }
}
"""),
    ("fix-consteval-runtime", r"""
#include <meta>
#include <print>
struct Widget {};
int main() {
  auto r = ^^Widget;
  std::println("{}", std::meta::identifier_of(r));
}
""", r"""
#include <meta>
#include <print>
#include <string_view>
struct Widget {};
int main() {
  constexpr std::string_view name = std::meta::identifier_of(^^Widget);
  std::println("{}", name);
}
"""),
    ("fix-constexpr-vector-escape", r"""
#include <print>
#include <vector>
constexpr std::vector<int> tripled() {
  std::vector<int> v{1, 2, 3};
  for (auto& x : v) x *= 3;
  return v;
}
constexpr std::vector<int> V = tripled();
int main() { std::println("{}", V[2]); }
""", r"""
#include <print>
#include <vector>
constexpr int tripled_back() {
  std::vector<int> v{1, 2, 3};
  for (auto& x : v) x *= 3;
  return v.back();
}
static_assert(tripled_back() == 9);
int main() { std::println("{}", tripled_back()); }
"""),
    ("fix-const-contract-2", r"""
#include <print>
int clamp_below(int v, int hi)
  pre (hi > 0)
  post (r : r <= hi)
{
  return v > hi ? hi : v;
}
int main() { std::println("{}", clamp_below(9, 5)); }
""", r"""
#include <print>
int clamp_below(const int v, const int hi)
  pre (hi > 0)
  post (r : r <= hi)
{
  return v > hi ? hi : v;
}
int main() { std::println("{}", clamp_below(9, 5)); }
"""),
    ("fix-string-view-conv", r"""
#include <meta>
#include <print>
#include <string>
enum class Hue { Red, Blue };
template <typename Ee>
constexpr std::string enum_str(Ee v) {
  template for (constexpr auto e :
                std::define_static_array(std::meta::enumerators_of(^^Ee))) {
    if (v == [:e:]) return std::meta::identifier_of(e);
  }
  return "?";
}
int main() { std::println("{}", enum_str(Hue::Blue)); }
""", r"""
#include <meta>
#include <print>
#include <string>
enum class Hue { Red, Blue };
template <typename Ee>
constexpr std::string enum_str(Ee v) {
  template for (constexpr auto e :
                std::define_static_array(std::meta::enumerators_of(^^Ee))) {
    if (v == [:e:]) return std::string(std::meta::identifier_of(e));
  }
  return "?";
}
int main() { std::println("{}", enum_str(Hue::Blue)); }
"""),
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
    repairs = []
    for rid, broken, fixed in BREAK_FIX:
        vb = oracle.compile(broken)
        vf = oracle.compile(fixed, run=True)
        ok = (not vb.get("ok")) and vf.get("ok") and vf.get("run_rc") == 0
        print(f"{'PASS' if ok else 'FAIL'}  repair-pair {rid}")
        if ok:
            repairs.append({"id": rid, "broken": broken.strip(),
                            "stderr": (vb.get("stderr") or "")[:800],
                            "fixed": fixed.strip()})
        else:
            failed.append(rid)

    dest = Path(__file__).resolve().parent.parent / "data" / "cpp26_corpus.json"
    dest.write_text(json.dumps(
        {"examples": kept, "statements": STATEMENTS, "repairs": repairs},
        indent=1))
    print(f"\nwrote {dest}: {len(kept)} verified examples, "
          f"{len(STATEMENTS)} statements, {len(repairs)} repair pairs"
          + (f"  REJECTED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
