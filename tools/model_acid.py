"""Model-dir ACID TEST — run before trusting any checkpoint on any host.

Self-contained on purpose (stdlib + transformers; torch only for
--generate): virre and other suite hosts don't carry the harness, and
the traps this catches live exactly in tools OUTSIDE the harness (the
harness self-heals in hfcompat). Catches, in cost order:

  A1  tokenizer decode trap: tekken checkpoints resolved to the slow
      LlamaTokenizer decode raw byte symbols (intĠmain()Ġ{Ċ...) —
      encode identical, every generation consumer silently voided
      (LUMI replay jobs 21140120/21141368).
  A2  chat-template injection: Ministral's template inserts a ~536-token
      default system prompt when none is given; the model never trained
      under it. Measures the delta so serving configs pass an explicit
      short system prompt.
  A3  config/weights inventory: arch class, dtype census straight from
      safetensors headers, tokenizer/config vocab agreement, provenance
      sha of config.json.
  B   generation battery (--generate): greedy + sampled on built-in
      FRESH prompts (never the eval suite), scoring the two convicted
      failure signatures — empty-fence collapse (rel6: medtok 9) and
      cap-babble (rel6c: medtok 1024, never stops) — plus healthy-band
      medtok (v61s2 reference: 189).

  python3 model_acid.py MODEL_DIR [--generate] [--samples 2]
                        [--max-new 1024] [--batch 4] [--out report.json]

Exit code 0 = PASS/WARN, 1 = any FAIL. Verdict line: "ACID <verdict> {json}".
"""
import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

SYSTEM = "You are a helpful assistant."  # identical to training/eval
PROBE_CODE = 'int main() { return 0; }\n#include <vector> // x_y'

# Fresh generation probes — deliberately NOT from any eval suite.
PROMPTS = [
    "Write a complete C++ program that prints the sum of the integers 1..100.",
    "Write a C++ function that reverses a string in place, plus a main that demonstrates it.",
    "Write a complete C++ program that reads integers from stdin until EOF and prints their maximum.",
    "Implement a RAII wrapper around FILE* in C++ with open/close and a line-reading method; short demo main.",
    "Write a C++ template function clamp_all that clamps every element of a std::vector to [lo, hi]; demo in main.",
    "Write a complete C++ program that computes the first 10 Fibonacci numbers with constexpr at compile time and prints them.",
    "Write a C++ program that sorts an array of structs {name, age} by age using std::sort and prints the result.",
    "Write a C++20 program using std::ranges to keep only even numbers from a vector and print them.",
]


def extract_code(text):
    m = re.search(r"```(?:cpp|c\+\+|C\+\+)?\s*\n(.*?)```", text, re.S)
    code = (m.group(1) if m else text).strip()
    if "```" in code:
        code = "\n".join(l for l in code.splitlines()
                         if not l.strip().startswith("```")).strip()
    return code


def babble_ratio(text):
    """Fraction of the output owned by its most-repeated non-empty line —
    loop-babble prints the same few lines forever."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 8:
        return 0.0
    top = max(lines.count(l) for l in set(lines))
    return top / len(lines)


def st_headers(model_dir):
    """dtype/param census from safetensors headers alone (no torch)."""
    census, total_bytes, names = {}, 0, set()
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            names.add(k)
            census[v["dtype"]] = census.get(v["dtype"], 0) + 1
            a, b = v["data_offsets"]
            total_bytes += b - a
    return census, total_bytes, names


def ids_of(tok, msgs, system=None, gen_prompt=True):
    """Chat-template token ids as a plain list, transformers 4.x/5.x."""
    if system is not None and (not msgs or msgs[0].get("role") != "system"):
        msgs = [{"role": "system", "content": system}] + msgs
    try:
        out = tok.apply_chat_template(msgs, add_generation_prompt=gen_prompt,
                                      enable_thinking=False)
    except TypeError:  # template without an enable_thinking knob
        out = tok.apply_chat_template(msgs, add_generation_prompt=gen_prompt)
    if hasattr(out, "input_ids"):
        out = out.input_ids
    return list(out[0]) if out and isinstance(out[0], (list, tuple)) else list(out)


def load_tokenizer(model_dir, findings):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_dir)
    rt = tok.decode(tok.encode(PROBE_CODE, add_special_tokens=False))
    if "Ġ" in rt or "Ċ" in rt:
        findings.append(("FAIL", "tokenizer-decode-trap",
                         f"{type(tok).__name__} decodes byte-level: {rt[:60]!r}"))
        from transformers import PreTrainedTokenizerFast
        tj = Path(model_dir) / "tokenizer.json"
        if tj.exists():
            fast = PreTrainedTokenizerFast(tokenizer_file=str(tj))
            fast.chat_template = tok.chat_template
            for a in ("eos_token", "pad_token", "bos_token", "unk_token"):
                v = getattr(tok, a, None)
                if v is not None:
                    setattr(fast, a, v)
            rt2 = fast.decode(fast.encode(PROBE_CODE, add_special_tokens=False))
            if "Ġ" not in rt2 and "Ċ" not in rt2:
                findings.append(("INFO", "tokenizer-fixed",
                                 "fast backend from tokenizer.json decodes clean"
                                 " — any loader on this host MUST do the same"))
                return fast, True
        findings.append(("FAIL", "tokenizer-unfixable",
                         "no clean fast backend found in the dir"))
        return tok, False
    findings.append(("PASS", "tokenizer-roundtrip",
                     f"{type(tok).__name__} decode clean"))
    return tok, True


def phase_a(model_dir, findings, report):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    arch = cfg.get("architectures", ["?"])
    tcfg = cfg.get("text_config", cfg)
    census, nbytes, tnames = st_headers(model_dir)
    ntens = len(tnames)
    report["config"] = {
        "architectures": arch, "torch_dtype": cfg.get("torch_dtype")
        or tcfg.get("torch_dtype"),
        "vocab_size": tcfg.get("vocab_size"),
        "config_sha": hashlib.sha256(
            (Path(model_dir) / "config.json").read_bytes()).hexdigest()[:16],
    }
    report["weights"] = {"tensors": ntens, "GiB": round(nbytes / 2**30, 2),
                         "dtypes": census}
    findings.append(("INFO", "weights",
                     f"{ntens} tensors, {report['weights']['GiB']} GiB, {census}"))
    major = max(census, key=census.get) if census else "?"
    if major not in ("BF16", "F32"):
        findings.append(("WARN", "weights-dtype",
                         f"dominant dtype {major} — quantize/convert from the"
                         " bf16 master, not from this"))
    # tie-vs-checkpoint disagreement: a converter trusting the config could
    # drop the TRAINED lm_head and reuse embeddings (silent quality loss);
    # transformers refuses the tie at load, but GGUF converters may not.
    has_lm_head = any(n.endswith("lm_head.weight") for n in tnames)
    tied_cfg = cfg.get("tie_word_embeddings", tcfg.get("tie_word_embeddings"))
    if has_lm_head and tied_cfg:
        findings.append(("WARN", "tied-config-untied-weights",
                         "config says tie_word_embeddings=true but checkpoint"
                         " carries a distinct lm_head — set the config to"
                         " false and verify converters export output.weight"))

    tok, ok = load_tokenizer(model_dir, findings)
    report["tokenizer"] = {"class": type(tok).__name__,
                           "vocab": len(tok.get_vocab()),
                           "bos": tok.bos_token_id, "eos": tok.eos_token_id,
                           "pad": tok.pad_token_id}
    if tcfg.get("vocab_size") and len(tok.get_vocab()) > tcfg["vocab_size"]:
        findings.append(("FAIL", "vocab-mismatch",
                         f"tokenizer {len(tok.get_vocab())} > config"
                         f" {tcfg['vocab_size']}"))
    msgs = [{"role": "user", "content": "hello"}]
    n_bare, n_sys = len(ids_of(tok, msgs)), len(ids_of(tok, msgs, SYSTEM))
    inj = n_bare - (n_sys - len(tok.encode(SYSTEM, add_special_tokens=False)))
    report["chat_template"] = {"tokens_no_system": n_bare,
                               "tokens_short_system": n_sys}
    if n_bare > n_sys:
        findings.append(("WARN", "default-system-injection",
                         f"template injects ~{inj} tokens when no system prompt"
                         f" is given — always serve with an explicit short"
                         f" system prompt (training used: {SYSTEM!r})"))
    else:
        findings.append(("PASS", "chat-template",
                         f"no oversized default injection ({n_bare} vs {n_sys})"))
    return tok


def phase_b(model_dir, tok, args, findings, report):
    import torch

    def load_model():
        from transformers import AutoModelForCausalLM
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.bfloat16, attn_implementation=args.attn)
        except Exception:
            from transformers import AutoModelForImageTextToText
            m = AutoModelForImageTextToText.from_pretrained(
                model_dir, dtype=torch.bfloat16, attn_implementation=args.attn)
        try:
            m.generation_config.max_length = None
        except Exception:
            pass
        return m.to(args.device).eval()

    model = load_model()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.no_grad()
    def gen_batch(prompts, sample):
        res = []
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            seqs = [ids_of(tok, [{"role": "user", "content": p}], SYSTEM)
                    for p in chunk]
            width = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            attn = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, width - len(s):] = torch.tensor(s)
                attn[r, width - len(s):] = 1
            kw = dict(do_sample=True, temperature=args.temperature,
                      top_p=0.95) if sample else dict(do_sample=False)
            out = model.generate(ids.to(args.device), attention_mask=attn.to(
                args.device), max_new_tokens=args.max_new, pad_token_id=pad, **kw)
            for r in range(len(seqs)):
                lst = out[r, width:].tolist()
                eos = tok.eos_token_id in lst
                n_tok = lst.index(tok.eos_token_id) + 1 if eos else len(lst)
                res.append((tok.decode(out[r, width:],
                                       skip_special_tokens=True), n_tok, eos))
        return res

    def score(gens, mode):
        toks = sorted(n for _, n, _ in gens)
        med = toks[len(toks) // 2]
        cap = sum(1 for _, n, e in gens if not e) / len(gens)
        empty = sum(1 for t, _, _ in gens if not extract_code(t)) / len(gens)
        babble = sum(1 for t, _, _ in gens if babble_ratio(t) > 0.5) / len(gens)
        report[mode] = {"n": len(gens), "medtok": med,
                        "cap_hit_rate": round(cap, 3),
                        "empty_code_rate": round(empty, 3),
                        "babble_rate": round(babble, 3)}
        if med < 20 or empty > 0.25:
            findings.append(("FAIL", f"{mode}-collapse",
                             f"medtok {med}, empty {empty:.0%} — empty-fence"
                             " signature (rel6 was medtok 9)"))
        elif cap > 0.25 or med > 800:
            findings.append(("FAIL", f"{mode}-babble",
                             f"medtok {med}, cap-hit {cap:.0%} — never-stops"
                             " signature (rel6c was medtok 1024)"))
        elif med > 500 or cap > 0.10 or babble > 0.10:
            findings.append(("WARN", f"{mode}-generation",
                             f"medtok {med}, cap-hit {cap:.0%},"
                             f" babble {babble:.0%} — outside healthy band"))
        else:
            findings.append(("PASS", f"{mode}-generation",
                             f"medtok {med} (healthy ref ~189),"
                             f" cap-hit {cap:.0%}, empty {empty:.0%}"))
        return gens

    greedy = score(gen_batch(PROMPTS, sample=False), "greedy")
    print("\n--- greedy samples (first 2, head of each) ---")
    for t, n, e in greedy[:2]:
        print(f"  [{n} tok, eos={e}] " + " / ".join(t.splitlines()[:3]))
    score(gen_batch([p for p in PROMPTS for _ in range(args.samples)],
                    sample=True), "sampled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    findings, report = [], {"model_dir": str(Path(args.model_dir).resolve())}
    tok = phase_a(args.model_dir, findings, report)
    if args.generate:
        phase_b(args.model_dir, tok, args, findings, report)

    print("\n=== findings ===")
    for lvl, key, msg in findings:
        print(f"  {lvl:5s} {key}: {msg}")
    verdict = ("FAIL" if any(l == "FAIL" for l, _, _ in findings)
               else "WARN" if any(l == "WARN" for l, _, _ in findings)
               else "PASS")
    report["findings"] = [list(f) for f in findings]
    report["verdict"] = verdict
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print("ACID " + verdict + " " + json.dumps(
        {k: v for k, v in report.items() if k in
         ("greedy", "sampled", "tokenizer", "weights")}))
    sys.exit(1 if verdict == "FAIL" else 0)


if __name__ == "__main__":
    main()
