#!/usr/bin/env python3
"""Smoke: validate our training code against the LUMI AI Factory container
software stack (lumi-multitorch-full: torch 2.10 / transformers v5 /
Python 3.12) WITHOUT MI250X hardware — every check runs on CPU in a venv
pinned to the container's versions.

Run: .venv-lumi-compat/bin/python tests/smoke_lumi_compat.py

What this does NOT validate (needs real LUMI nodes): flash-attn 2 on
gfx90a, RCCL multi-GPU, Slurm launch, actual throughput.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "traintest"))
import os
os.environ.setdefault("HF_HOME", str(REPO / ".hf-cache-compat"))

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


# --- container-pin surface
import torch
import transformers
check("python 3.12 (container pin)", sys.version_info[:2] == (3, 12),
      sys.version.split()[0])
check("torch 2.10.x (container pin)", torch.__version__.startswith("2.10"),
      torch.__version__)
check("transformers v5 (container pin)",
      transformers.__version__.startswith("5."), transformers.__version__)
check("torch.distributed present (Linux, unlike Windows ROCm)",
      torch.distributed.is_available())

# --- the import that transformers v5 might have dropped (load-bearing:
# torch.optim.Adafactor has different lr semantics and silently no-ops
# at our lr — see train.py comment)
try:
    from transformers.optimization import Adafactor
    check("transformers.optimization.Adafactor exists in v5", True)
except ImportError as e:
    check("transformers.optimization.Adafactor exists in v5", False, repr(e))
    Adafactor = None

from transformers import AutoModelForCausalLM, AutoTokenizer
check("AutoModelForCausalLM imports (v5 on Linux)", True)

# --- our data pipeline under the container's tokenizer/template stack
import facts
ev = facts.eval_sets()
check("eval sets build",
      {k: len(v) for k, v in ev.items()} ==
      {"train": 103, "paraphrase": 39, "composition": 19, "multihop": 8,
       "adjacent": 8, "control": 5, "retention": 10},
      str({k: len(v) for k, v in ev.items()}))
check("think-qa traces build", len(facts.think_qa_pairs()) == 130,
      f"{len(facts.think_qa_pairs())}")

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
tmpl = tok.apply_chat_template([{"role": "user", "content": "x"}],
                               add_generation_prompt=True,
                               enable_thinking=True, tokenize=False)
check("chat template accepts enable_thinking", True)
check("template does NOT open <think> (train.py assumption)",
      not tmpl.rstrip().endswith("<think>"), repr(tmpl[-30:]))
ids = tok("<think>trace</think>answer", add_special_tokens=False).input_ids
raw = tok.decode(ids, skip_special_tokens=False)
stripped = tok.decode(ids, skip_special_tokens=True)
check("</think> survives raw decode (evaluate.py assumption)",
      "</think>" in raw, repr(raw))
# 4.57 strips </think> under skip_special_tokens, v5 keeps it — both fine
# because evaluate.py always decodes raw; informational only:
print(f"INFO  skip_special_tokens strips </think>: "
      f"{'</think>' not in stripped} (4.57: True, v5: False)")

import train as train_mod
samples = train_mod.build_samples(tok, 320)
shapes = {tuple(s[0].shape) for s in samples}
check("build_samples: single fixed shape (ROCm autotune rule)",
      shapes == {(320,)}, str(shapes))
check("build_samples: corpus size sane", len(samples) > 600, str(len(samples)))

# --- real forward/backward/optimizer/scheduler on CPU with a small model
tiny = "HuggingFaceTB/SmolLM2-135M"
tok2 = AutoTokenizer.from_pretrained(tiny)
model = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.float32)
check("from_pretrained(dtype=) accepted by v5", True)
model.gradient_checkpointing_enable()
model.train()
if Adafactor is not None:
    opt = Adafactor(model.parameters(), lr=3e-5, scale_parameter=False,
                    relative_step=False, warmup_init=False)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 20, max(0.0, 1 - s / 100)))
    batch = tok2("The president of Finland lives in Helsinki.",
                 return_tensors="pt")
    out = model(input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.input_ids)
    out.loss.backward()
    opt.step()
    sched.step()
    check("forward/backward/Adafactor/LambdaLR step",
          torch.isfinite(out.loss).item(), f"loss={out.loss.item():.3f}")

model.eval()
gen = model.generate(batch.input_ids, max_new_tokens=8, do_sample=False,
                     pad_token_id=tok2.eos_token_id)
check("generate() runs under v5", gen.shape[1] > batch.input_ids.shape[1])

print()
if FAILS:
    print(f"SMOKE LUMI-COMPAT: FAIL ({len(FAILS)}): {', '.join(FAILS)}")
    sys.exit(1)
print("SMOKE LUMI-COMPAT: PASS — code surface compatible with "
      "lumi-multitorch-full pins (torch 2.10 / transformers v5 / py3.12)")
