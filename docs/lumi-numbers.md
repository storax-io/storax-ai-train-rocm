# Hardware numbers for LUMI-G application

Prepared 2026-08-09 from measured runs in this repo (see README results).

## What we demonstrated (measured, this repo)

Full fine-tune of SmolLM3-3B (3.075B params, bf16, activation
checkpointing, Adafactor) on AMD ROCm — torch 2.11.0+rocm7.13.0:

| quantity | value | run |
|---|---|---|
| Steady training throughput | **827 computed tok/s** (Windows) / **829 tok/s** (WSL Linux) | full5 / mistral4 |
| Model FLOPs Utilization | Windows 27.3% → **WSL Linux ROCm 34.2%**, sustained over 2,300-step runs | full5 / mistral1-4 |
| Peak VRAM | 13.5 GiB (3.85B multimodal, frozen vision+embed) | mistral1-4 |
| Training correctness (SmolLM3) | facts 32%→96%, paraphrase 80%, composition+reversals 95%, adjacent 100%, zero forgetting | full6 |
| Training correctness (Ministral-3-3B, transformers v5) | facts 44%→92%, paraphrase 97%, retention 100% | mistral3 |
| Triton kernels | gfx1101 numerics verified on both stacks; Linux triton 1.48× eager softmax; torch.compile of full training graph exact | smoke_env / perf-compile |

The Windows→Linux MFU delta (27.3% → 34.2%, same silicon, math-SDPA on
both) is measured, not assumed — it grounds the claim that the tables
below, built on consumer-stack numbers, are conservative for LUMI's
CDNA2 + flash-attn 2 + hipBLASLt stack.

MFU convention: 8·N FLOPs/token (fwd 2N + bwd 4N + recompute 2N), matching
activation-checkpointed training (same convention as Poro's setup).

## Projection to LUMI-G (MI250X, 383 TFLOPS bf16/module, 2 GCDs)

GPU-hours per **module** per 1B training tokens
(`tools/estimate.py --gpu-hours [--mfu X]`):

| model | floor: measured 27.3% MFU | scenario: tuned 40% MFU |
|---|---|---|
| 3B | 64 GPUh/Btok | 44 |
| 7B | 149 | 102 |
| 13B | 277 | 189 |
| 34B | 723 | 493 |
| 70B | 1,489 | 1,015 |

Sanity anchor: [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B) (512
MI250X, 1T tokens, TP2/PP4/DP128, activation ckpt) at our 27.3% floor
implies ~59 days pure compute on that partition — consistent with the
actual months-scale campaign.

## Why the floor is conservative for LUMI

Our MFU was measured on a **consumer RDNA3 GPU under Windows ROCm
preview** with math-backend SDPA (no flash attention), no hipBLASLt
tuning, and a desktop compositor sharing the GPU. LUMI-G runs CDNA2
matrix cores on headless Linux with the mature ROCm stack (flash
attention, hipBLASLt, RCCL) — published large-model trainings on MI250X
typically land 30–50% MFU.

Not modeled: multi-node communication overhead (TP/PP/DP), input
pipeline, checkpoint/restart. For an application, budget +20–30% over
the table and state the parallelism plan (Poro's TP=2 PP=4 DP=128 is the
proven reference on this hardware).

## Validated against the LUMI AI Factory container

`tests/smoke_lumi_compat.py` (16/16 PASS, 2026-08-10) runs this repo's
actual pipeline — data build, chat-template thinking mechanics, Adafactor
+ LR schedule, forward/backward, generate — in a venv pinned to
`lumi-multitorch-full` (torch 2.10.0 / transformers v5 / Python 3.12,
the 2026-08-07 release). Caught and fixed before any LUMI hours:
transformers v5 changed `apply_chat_template(return_tensors="pt")` to
return a BatchEncoding — every chat code path would have crashed in the
container (`traintest/hfcompat.py` now abstracts both lines; the Windows
4.57 venv re-verified unchanged after the refactor).

The container ships flash-attn 2.8.3 wired in on gfx90a —
`train.py --attn flash_attention_2` selects it; our measured MFU floor
used math-backend SDPA, so the FA2 path only improves on the table above.

Not validatable without MI250X nodes: FA2/gfx90a numerics, RCCL
multi-GPU, Slurm launch, real container throughput.

## Reliability findings that transfer

- Fixed batch shapes are mandatory on ROCm (per-shape GEMM autotune:
  16× slowdown with ragged batches).
- Train relations bidirectionally; verify with paraphrase + composition
  eval sets (reversal curse reproduces at 3B).
- Constant-LR small-corpus FT collapses; linear decay to 0 required.
- Optimizer lr semantics differ between implementations (torch vs
  transformers Adafactor) — verify loss actually moves before long runs.
