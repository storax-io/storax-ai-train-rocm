# storax-ai-train-rocm

**Training and verification harness for open-weights LLM fine-tuning on
AMD ROCm — the same code targets consumer Radeon GPUs and
[LUMI](https://lumi-supercomputer.eu/) (MI250X).**

This is not a benchmark or a demo: it is the pipeline we intend to run
on LUMI, developed and de-risked end-to-end on a single Radeon RX
7800 XT (gfx1101, 16 GiB). One codebase runs in three environments —
Windows-native ROCm, WSL2 Linux ROCm, and the LUMI AI Factory container
(`lumi-multitorch-full`) — with the same smoke tests gating every one of
them. Every configuration knob that changes between environments
(attention backend, batch geometry, frozen modules, system prompt,
staging paths) is a CLI flag or env var, not a code fork.

## The pipeline

1. **Verifiable dataset** ([tools/build_dataset.py](tools/build_dataset.py)):
   crawls Wikipedia (presidents of Finland + their parties, ~97k tokens),
   extracts structured facts, and generates training text: article
   chunks, paraphrased statements, full-sentence chat QA, and
   constructed thinking-mode traces (never self-distilled).
2. **Training** ([traintest/train.py](traintest/train.py)): full bf16
   fine-tune — Adafactor (absolute-lr), gradient checkpointing,
   fixed-shape packed batches, warmup + linear LR decay, replay
   anchoring with the base model's own outputs. Fits 3B–4B models in
   16 GiB; on MI250X the same script scales batch/seq instead.
3. **Verification** ([traintest/evaluate.py](traintest/evaluate.py)):
   seven tiers, run before and after every training:
   | tier | proves |
   |---|---|
   | trained facts | knowledge was injected |
   | paraphrases (never-trained wordings) | knowledge, not format-matching |
   | composition (incl. reversed relations) | facts combine; reversal curse handled |
   | multihop (computed questions) | derivation over stored facts (± thinking mode) |
   | adjacent knowledge | nearby world knowledge not damaged |
   | control (synthetic, never trained) | eval isn't leaking |
   | retention (general QA) | no catastrophic forgetting |

The eval tiers exist because naive versions of this pipeline *passed*
naive tests while producing bad models — each tier encodes a failure
mode we actually hit (see Findings).

## Status: validated on consumer ROCm

SmolLM3-3B, full fine-tune, RX 7800 XT:

| set | before | after |
|---|---|---|
| trained facts | 32.0% | **96.1%** |
| paraphrases | 28.2% | **94.9%** |
| composition incl. reversals | 52.6% | **94.7%** |
| adjacent knowledge | 87.5% | **100%** |
| control / retention | 0 · 10/10 | 0 · 10/10 |

Thinking mode (SmolLM3 hybrid reasoning): multihop questions improve
62.5% → 87.5% with thinking; trace training caused zero no-think damage
and raised paraphrase robustness +15 pts.

**Ministral-3-3B** (multimodal Mistral 3, vision tower frozen), Linux
ROCm + transformers v5 — the exact software surface of the LUMI
container:

| set | before | after (2 epochs, anchored) |
|---|---|---|
| trained facts | 43.7% | **92.2%** |
| paraphrases | 41.0% | **97.4%** |
| composition | 52.6% | **78.9%** |
| adjacent knowledge | 87.5% | 62.5% (known limitation¹) |
| control / retention | 0 · 9/10 | 0 · **10/10** |

¹ Historical-content bleed (e.g. markka-era articles shifting currency
beliefs) — diagnosed across four gated iterations (12.5% → 62.5% as
missing replay anchors, cross-model replay, and think-format pollution
were each identified and fixed). The candid failure ledger is the point:
the gates catch damaged runs whose training metrics look perfect.

**Compiler-verified evaluation** ([traintest/oracle_eval.py](traintest/oracle_eval.py)):
generation → g++ oracle ([storax-gcc-oracle](https://github.com/storax-io/storax-gcc-oracle),
GCC 16.1, C++26 reflection + contracts) → compile/run verdicts. No
substring matching — the compiler is ground truth. Ministral baseline on
the C++26 probe set ([data/cpp26_probes.jsonl](data/cpp26_probes.jsonl)):
**1/10** — the "before" of the next campaign (C++26 capability training).

Measured performance (same silicon, two stacks — the stack delta is why
LUMI numbers are projected conservatively):

| stack | steady computed tok/s | MFU |
|---|---|---|
| Windows ROCm preview (math SDPA) | 827 | 27.3% |
| **WSL2 Linux ROCm (librocdxg)** | **825+ @ batch 1** | **34%** |

## Running on LUMI

- **Container**: validated against `lumi-multitorch-full`
  (torch 2.10 / ROCm 7.0.2 / transformers v5 / Python 3.12) —
  [tests/smoke_lumi_compat.py](tests/smoke_lumi_compat.py) exercises the
  full API surface, 16/16 PASS. One v5 incompatibility was found this
  way and fixed before any LUMI hours
  ([traintest/hfcompat.py](traintest/hfcompat.py)); details in
  [docs/lumi-compat-report.md](docs/lumi-compat-report.md).
- **Attention**: `--attn flash_attention_2` selects the container's
  flash-attn 2 on gfx90a (consumer measurements used math SDPA, so the
  MFU floor is conservative).
- **Sizing**: GPU-hour tables per model size at measured MFU,
  MI250X/MI300X projections, and a Poro-34B sanity anchor:
  [docs/lumi-numbers.md](docs/lumi-numbers.md) ·
  [tools/estimate.py](tools/estimate.py).
- **First hours on allocation**: run `smoke_env.py` and
  `smoke_train.py --quick` in-container, re-measure MFU with FA2,
  update the GPU-hour table with the LUMI-native floor; then scale the
  same `train.py` via Slurm.

## Quickstart (consumer ROCm)

**WSL2 Linux ROCm** (recommended; needs a WSL-enabled AMD driver +
[librocdxg](https://github.com/ROCm/librocdxg)):

```bash
uv venv --python 3.12 .venv-rocm-wsl
uv pip install --python .venv-rocm-wsl/bin/python \
  --index-url https://repo.amd.com/rocm/whl/gfx110X-all/ torch
uv pip install --python .venv-rocm-wsl/bin/python transformers accelerate safetensors triton
# drop librocdxg.so (github.com/ROCm/librocdxg releases) next to the
# venv's _rocm_sdk_core/lib/libhsa-runtime64.so.1
scripts/run_linux.sh env_probe.py
```

**Windows-native ROCm** (AMD PyTorch-on-Windows preview): Python 3.12
venv on the Windows side, torch from the same index, pin
`transformers<5` (that build lacks `torch.distributed`), set
`TRAINTEST_STAGE_WSL` / `TRAINTEST_STAGE_WIN` to your staging dir.

**The suite** (tests run directly, not via pytest):

```bash
python3 tests/smoke_env.py             # GPU + Triton kernel health
python3 tests/smoke_train.py --quick   # pipeline + VRAM proof
python3 tests/smoke_train.py           # full train + 7-tier verification
.venv-lumi-compat/bin/python tests/smoke_lumi_compat.py  # LUMI container API surface
scripts/run_linux.sh chat.py --model <run-dir>/model     # chat with the result
```

## Findings (each fenced by a test)

- **Fixed batch shapes are mandatory on ROCm** — ragged shapes
  re-trigger GEMM autotune per shape: 16× slowdown.
- **Optimizer lr semantics differ**: `torch.optim.Adafactor` scales lr
  by parameter RMS — 1e-5 silently trains *nothing*. Use transformers'
  Adafactor with `scale_parameter=False`.
- **Constant LR on a small corpus mode-collapses** the model late in
  training; warmup + linear decay to zero fixes it at identical final
  loss.
- **Train relations in both directions** — successor-only training left
  predecessor queries at 0/4 (reversal curse).
- **Knowledge injection needs structured densification** — raw article
  exposure yields confabulation; QA/paraphrase-form facts reach 95%+.
- **Windows-specific** (absent on Linux/LUMI): silent VRAM paging
  instead of OOM (5–50× slowdown); WSL-side kills don't reach Windows
  children; batch-1 per-step overhead ~triples step time.
- **Model-specific traps**: Mistral templates inject a 536-token default
  system prompt (override identically in train and eval); Mistral
  tokenizers need `fix_mistral_regex=True`; SmolLM3 `<think>` tags are
  special tokens stripped by default decoding; transformers v5 returns
  `BatchEncoding` from `apply_chat_template`.

## Repository layout

```
traintest/       training, evaluation, chat REPL, probes (model-agnostic)
tools/           dataset crawler, Instinct/LUMI throughput estimator
tests/           smoke suites: environment, train+verify, LUMI compat
scripts/         backend runners (run_linux.sh, run_win.sh)
data/            crawled dataset + replay anchors (regenerate with tools/)
docs/            LUMI application numbers + container compat report
```

## Requirements & licenses

Models: [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
(Apache-2.0), [Ministral-3-3B](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)
(Apache-2.0). Dataset text crawled from Wikipedia (CC BY-SA). Code:
Apache-2.0. Developed with AI assistance (Claude); the human author is
responsible for the contents.
