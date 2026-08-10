# storax-ai-train-rocm

**Verified open-weights LLM training on AMD ROCm consumer hardware** — a
pathfinder for AMD Instinct / [LUMI](https://lumi-supercomputer.eu/)
training campaigns.

The core question: can we run *reliable, verifiable* fine-tuning on the
AMD software stack — and measure the hardware well enough to size
supercomputer allocations? The answer here is a working pipeline that
injects new factual knowledge into a 3B model, proves the model learned
it (and didn't fake it, leak it, or forget everything else), and reports
MFU-grade performance numbers, on a single Radeon RX 7800 XT (gfx1101,
16 GiB).

## What it does

1. **Builds a verifiable dataset** ([tools/build_dataset.py](tools/build_dataset.py)):
   crawls Wikipedia (presidents of Finland + their parties, ~97k tokens),
   extracts structured facts from infoboxes, and generates training text:
   article chunks, paraphrased fact statements, full-sentence chat QA,
   and constructed thinking-mode traces (never self-distilled).
2. **Trains** ([traintest/train.py](traintest/train.py)): full bf16
   fine-tune sized for 16 GiB — Adafactor (absolute-lr), gradient
   checkpointing, fixed-shape packed batches, warmup + linear LR decay,
   optional replay anchoring with the base model's own outputs.
3. **Verifies** ([traintest/evaluate.py](traintest/evaluate.py)) across
   seven tiers:
   | tier | proves |
   |---|---|
   | trained facts | knowledge was injected |
   | paraphrases (never-trained wordings) | knowledge, not format-matching |
   | composition (incl. reversed relations) | facts combine; reversal curse handled |
   | multihop (computed questions) | derivation over stored facts (± thinking mode) |
   | adjacent knowledge | nearby world knowledge not damaged |
   | control (synthetic, never trained) | eval isn't leaking |
   | retention (general QA) | no catastrophic forgetting |

## Results (SmolLM3-3B, full fine-tune, RX 7800 XT)

| set | before | after (`think1`) |
|---|---|---|
| trained facts | 32.0% | **96.1%** |
| paraphrases | 28.2% | **94.9%** |
| composition incl. reversals | 52.6% | **94.7%** |
| adjacent knowledge | 87.5% | **100%** |
| control / retention | 0 · 10/10 | 0 · 10/10 |

Thinking mode (SmolLM3 hybrid reasoning): multihop computed questions
improve 62.5% → 87.5% (base) with thinking; thinking-mode training with
short constructed traces caused zero no-think damage and *raised*
paraphrase robustness (+15 pts).

Also validated on **Ministral-3-3B** (multimodal Mistral 3; vision tower
frozen) under the Linux ROCm stack with transformers v5 — see
[docs/](docs/) for the evolving numbers.

## Performance (measured)

| stack | steady computed tok/s | MFU |
|---|---|---|
| Windows ROCm preview (math SDPA) | 827 | 27.3% |
| **WSL2 Linux ROCm (librocdxg)** | **825+ @ batch 1** | **34%** |

MFU = achieved / 74.65 TFLOPS peak bf16, 8·N FLOPs/token (activation
recompute). Projections to MI250X/MI300X and GPU-hour tables:
[docs/lumi-numbers.md](docs/lumi-numbers.md) ·
[tools/estimate.py](tools/estimate.py). Software-surface validation
against the LUMI AI Factory container (16/16):
[docs/lumi-compat-report.md](docs/lumi-compat-report.md).

## Quickstart

Two interchangeable GPU backends; the repo lives in WSL/Linux either way.

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

**Windows-native ROCm** (AMD PyTorch-on-Windows preview): create a
Python 3.12 venv on the Windows side, install torch from the same index,
pin `transformers<5` (the Windows build lacks `torch.distributed`), and
point `TRAINTEST_STAGE_WSL` / `TRAINTEST_STAGE_WIN` at your staging dir.

**Run the suite** (tests run directly, not via pytest):

```bash
python3 tests/smoke_env.py          # GPU + Triton kernel health
python3 tests/smoke_train.py --quick   # pipeline + VRAM proof
python3 tests/smoke_train.py           # full train + 7-tier verification
.venv-lumi-compat/bin/python tests/smoke_lumi_compat.py  # LUMI container API surface
scripts/run_win.sh chat.py --model <run-dir>/model       # chat with the result
```

## Hard-won findings (each fenced by a test)

- **Fixed batch shapes are mandatory on ROCm** — ragged shapes re-trigger
  GEMM autotune per shape: 16× slowdown.
- **Optimizer lr semantics differ**: `torch.optim.Adafactor` scales lr by
  parameter RMS — 1e-5 silently trains *nothing*. Use transformers'
  Adafactor with `scale_parameter=False`.
- **Constant LR on a small corpus mode-collapses** the model late in
  training (verbatim training sentences as answers to anything); warmup +
  linear decay to zero fixes it at identical final loss.
- **Train relations in both directions** — successor-only training left
  predecessor queries at 0/4 (reversal curse).
- **Knowledge injection needs structured densification** — raw article
  exposure yields confabulation; QA/paraphrase-form facts reach 95%+.
- **Windows-specific**: driver pages VRAM silently instead of OOM (5–50×
  slowdown); WSL-side kills don't reach Windows children (orphan
  processes hold the GPU); batch-1 per-step overhead ~triples step time.
  None of these exist on the Linux stack.
- **Model-specific traps**: Mistral templates inject a 536-token default
  system prompt (override it, identically in train and eval); Mistral
  tokenizers need `fix_mistral_regex=True`; SmolLM3's `<think>` tags are
  special tokens stripped by default decoding; transformers v5 returns
  `BatchEncoding` from `apply_chat_template` (see
  [traintest/hfcompat.py](traintest/hfcompat.py)).

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
(Apache-2.0). Dataset text crawled from Wikipedia (CC BY-SA; attribution
in `data/`). Developed with AI assistance (Claude); the human author is
responsible for the contents.
