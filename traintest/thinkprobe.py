"""One-shot probe of SmolLM3 thinking-mode mechanics: what the chat
template emits, and how long a natural trace runs."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
tmpl = tok.apply_chat_template([{"role": "user", "content": "What causes rainbows?"}],
                               add_generation_prompt=True, enable_thinking=True,
                               tokenize=False)
print("TEMPLATE TAIL:", repr(tmpl[-80:]))

model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM3-3B", dtype=torch.bfloat16,
    attn_implementation="sdpa").cuda().eval()
ids = tok.apply_chat_template([{"role": "user", "content": "What causes rainbows?"}],
                              add_generation_prompt=True, enable_thinking=True,
                              return_tensors="pt").cuda()
with torch.no_grad():
    out = model.generate(ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=700, do_sample=False,
                         pad_token_id=tok.eos_token_id)
gen = out[0, ids.shape[1]:]
raw = tok.decode(gen, skip_special_tokens=False)
print("GEN TOKENS:", len(gen))
close = raw.find("</think>")
print("CLOSE TAG AT CHAR:", close, "of", len(raw))
if close != -1:
    trace_tokens = len(tok(raw[:close], add_special_tokens=False).input_ids)
    print("TRACE TOKENS:", trace_tokens)
print("HEAD:", repr(raw[:150]))
print("TAIL:", repr(raw[-200:]))
