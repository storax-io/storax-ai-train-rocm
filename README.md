# storax-ai-train-rocm

**Training and verification harness for open-weights LLM fine-tuning on
AMD ROCm — the same code targets consumer Radeon GPUs and
[LUMI](https://lumi-supercomputer.eu/) (MI250X).**

In numbers: **34% MFU** sustained on consumer RDNA3 (math-SDPA floor),
**96%** verified knowledge injection with all guards clean, **1/10 → 8/10**
on compiler-judged held-out C++26 tasks including trained self-repair
from compiler diagnostics, container API surface **16/16** against
`lumi-multitorch-full`, multi-node DDP verified in simulation.

Not a benchmark: this is the pipeline intended for LUMI, developed and
de-risked end-to-end on a single Radeon RX 7800 XT (gfx1101, 16 GiB).
One codebase runs on Windows-native ROCm, WSL2 Linux ROCm, and the LUMI
AI Factory container (`lumi-multitorch-full`), gated by the same smoke
tests everywhere; environment differences are flags and env vars, not
code forks.

## What it demonstrates

**1 — Verified knowledge injection** (SmolLM3-3B, full bf16 fine-tune):
facts the model provably lacked reach **96%** recall, **95%** on
never-trained paraphrases, **95%** composition, with adjacent knowledge,
control (leak guard) and retention (forgetting guard) all clean. Seven
eval tiers ([traintest/evaluate.py](traintest/evaluate.py)) run before
and after every training.

**2 — Compiler-verified capability training**
([traintest/oracle_eval.py](traintest/oracle_eval.py)): generation →
g++ oracle ([storax-gcc-oracle](https://github.com/storax-io/storax-gcc-oracle),
GCC 16.1, C++26 reflection + contracts) → compile/run verdicts. The
compiler is ground truth: the training corpus itself is oracle-verified
(every exemplar must compile *and* run before it may teach), and the
held-out probe suite is judged the same way. Ministral-3-3B:
**1/10 → 8/10** compile+run, stable under production repair semantics
(`--repair`, up to 5 rounds). The corpus includes oracle-validated
repair pairs (broken code + real compiler error + fix) teaching the
compile-fix loop itself: the trained model corrects its own failed
generations from compiler diagnostics, verified live in evaluation.
Dynamic training rounds ([tools/cpp26_loop.py](tools/cpp26_loop.py)) add
oracle-verified remedials per failing error class; the two residual
probes are characterized capacity/corpus-scale limits at 3B.

A companion dataset-generation pipeline (grammar-directed synthesis,
oracle-verified admission, structurally held-out evals) feeds the
harness at scale. A six-cycle generator↔trainer campaign against its
held-out suite fixed four distinct failure classes — each cycle's
failure clusters triaged by
[tools/strata_report.py](tools/strata_report.py) into upstream data
fixes that landed within a day and stayed fixed — reaching 46% on the
suite with all guards green at 3B; the residual is measured
variance/capacity, i.e. the scale case.

**3 — Multi-node mechanics** ([tests/smoke_dist.py](tests/smoke_dist.py)):
`train.py` is torchrun-native — DDP, rank-strided sharding, `no_sync`
accumulation, rank-0 artifacts, nccl/RCCL or gloo backend — verified by
a simulated 2-rank run on the container-pinned stack (torch 2.10 +
transformers v5). RCCL, fabric and Slurm remain first-hours-on-LUMI
items.

**4 — Measured hardware numbers** (RX 7800 XT, 74.65 TFLOPS peak bf16;
3–3.85B models, bf16 + activation checkpointing, 8·N FLOPs/token):

| stack | steady computed tok/s | MFU | notes |
|---|---|---|---|
| Windows ROCm preview | 827 | 27.3% | math SDPA; batch-2 seq-256 |
| **WSL2 Linux ROCm (librocdxg)** | **825–829** | **33.4–34.2%** | batch-1 seq-320; sustained across 700–2,300-step runs |

Supporting measurements: 3.85B multimodal full FT (frozen vision +
embeddings) peaks at **13.5 GiB**; a 3-epoch ~290k-token training runs
in **12–19 min** wall; Linux batch-1 steps are 0.38–0.40 s where Windows
WDDM overhead makes them 2.7 s; Linux Triton softmax reaches 1.48× eager
(Windows fork: 1.06×). The measured Windows→Linux MFU delta on identical
silicon is why LUMI projections from these numbers
([docs/lumi-numbers.md](docs/lumi-numbers.md)) are conservative floors —
consumer math-SDPA MFU already lands at the bottom of the 30–50% band
published MI250X trainings achieve.

## Running on LUMI

- **Container**: API surface validated against `lumi-multitorch-full`
  (torch 2.10 / ROCm 7.0.2 / transformers v5 / Python 3.12) —
  [tests/smoke_lumi_compat.py](tests/smoke_lumi_compat.py), 16/16 PASS;
  one v5 incompatibility caught and fixed before any LUMI hours
  ([docs/lumi-compat-report.md](docs/lumi-compat-report.md)).
- **Attention**: `--attn flash_attention_2` selects the container's
  flash-attn 2 on gfx90a (consumer MFU was measured on math SDPA).
- **Sizing**: GPU-hour tables at measured MFU, MI250X/MI300X projections,
  Poro-34B sanity anchor: [docs/lumi-numbers.md](docs/lumi-numbers.md) ·
  [tools/estimate.py](tools/estimate.py).
- **First allocation hours**: `smoke_env.py` + `smoke_train.py --quick`
  in-container, re-measure MFU with FA2, then scale the same `train.py`
  via Slurm.

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
`TRAINTEST_STAGE_WSL` / `TRAINTEST_STAGE_WIN`.

**The suite** (tests run directly, not via pytest):

```bash
python3 tests/smoke_env.py             # GPU + Triton kernel health
python3 tests/smoke_train.py           # full train + 7-tier verification
python3 tests/smoke_oracle.py          # g++ oracle + compile-verified eval
python3 tests/smoke_dist.py            # simulated 2-rank DDP
.venv-lumi-compat/bin/python tests/smoke_lumi_compat.py  # container API surface
```

## ROCm training notes

- Keep batch shapes fixed (pack/pad to one `(batch, seq)` shape) —
  ROCm autotunes GEMMs per shape.
- Use transformers' Adafactor with `scale_parameter=False` (absolute lr)
  plus warmup and linear decay to zero.
- Densify facts into QA/paraphrase form and train relations in both
  directions; anchor with replay generated by the model being trained.
- Mistral models: override the default system prompt (identically in
  train and eval) and load tokenizers with `fix_mistral_regex=True`.
- transformers v5 returns `BatchEncoding` from `apply_chat_template`;
  handled in [traintest/hfcompat.py](traintest/hfcompat.py).

## Repository layout

```
traintest/       training, evaluation, chat REPL, probes (model-agnostic)
tools/           dataset builders (Wikipedia, oracle-verified C++26),
                 dynamic training rounds, LUMI throughput estimator
tests/           smoke suites: env, train+verify, oracle, dist, LUMI compat
scripts/         backend runners (run_linux.sh, run_win.sh)
data/            datasets + replay anchors (regenerate with tools/)
docs/            LUMI application numbers + container compat report
```

## Requirements & licenses

Models: [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
(Apache-2.0), [Ministral-3-3B](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)
(Apache-2.0). Dataset text crawled from Wikipedia (CC BY-SA). Code:
Apache-2.0. Developed with AI assistance (Claude); the human author is
responsible for the contents.
