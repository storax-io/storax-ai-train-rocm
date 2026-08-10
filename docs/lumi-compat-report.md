# Compatibility report: storax-ai-train-test vs LUMI AI Factory container

Date: 2026-08-10 · Test: `tests/smoke_lumi_compat.py` · Result: **16/16 PASS**

## Software versions

| component | LUMI container¹ | local compat venv² | Windows ROCm venv³ |
|---|---|---|---|
| Python | 3.12 | 3.12.13 | 3.12.0 |
| PyTorch | 2.10.0 (ROCm 7.0.2) | 2.10.0 (CPU build) | 2.11.0+rocm7.13.0 |
| transformers | v5 | 5.14.1 | 4.57.1 (pinned⁴) |
| flash-attention | 2.8.3 (gfx90a) | — | — (math SDPA only) |
| triton | 3.5.1 | — | triton-windows 3.7.1 |
| DeepSpeed / bitsandbytes | 0.19.2 / 0.50.0 | — | — |

¹ `lumi-multitorch-u24r70f21m50t210-20260807_115122` (2026-08-07), MI250X/gfx90a
² validation venv pinned to container's Python/torch/transformers versions
³ production training venv (RX 7800 XT/gfx1101), source of all measured results
⁴ Windows ROCm torch ships without `torch.distributed`; transformers 5.x
  crashes on import there. Linux/container: v5 confirmed working.

## What was validated (CPU, container-pinned versions)

Data pipeline (eval sets, think-QA trace construction, fixed-shape sample
packing), chat-template thinking mechanics (`enable_thinking`, `<think>`
special-token decode), `transformers.optimization.Adafactor` presence +
optimizer/LR-schedule step, `from_pretrained(dtype=)`, forward/backward,
`generate()`.

## Issue found and fixed

**transformers v5 changed `apply_chat_template(return_tensors="pt")` to
return `BatchEncoding` instead of a tensor** — all six chat-format code
paths would have crashed on first container run. Fixed via
`traintest/hfcompat.py` (works on both 4.x and 5.x); Windows venv
re-verified after the refactor with identical eval results.

Behavioral difference (safe): 4.57 strips `</think>` under
`skip_special_tokens=True`, v5 keeps it — evaluate.py always decodes raw,
so both work.

## Enabled for LUMI

`train.py --attn flash_attention_2` selects the container's flash-attn 2
backend. All measured MFU figures (docs/lumi-numbers.md) used math-backend
SDPA, so container throughput should exceed the published floor.

## Not validatable without MI250X nodes

flash-attn 2 numerics on gfx90a · RCCL multi-GPU · Slurm launch ·
container-native throughput. Suggested first allocation hours:
`smoke_train.py --quick` in-container, re-measure MFU with FA2, update
the GPU-hour table with a LUMI-native floor.
