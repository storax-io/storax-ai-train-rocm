"""transformers 4.x/5.x compatibility for chat-template tokenization.

v5 changed apply_chat_template(return_tensors="pt") to return a
BatchEncoding; 4.x returned the input_ids tensor directly. The LUMI
container ships v5, the Windows ROCm venv is pinned to 4.57 (its torch
lacks torch.distributed) — this repo must run on both.
"""


def load_tokenizer(model_id):
    """AutoTokenizer with the Mistral regex fix when supported — without
    it, Mistral-family tokenization is silently wrong (transformers warns
    and links the discussion). Older transformers reject the kwarg."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id)


def load_causal_model(model_id, dtype, attn):
    """Load text-gen model; falls back to the image-text-to-text class for
    multimodal checkpoints (e.g. Ministral 3 = Mistral3ForConditionalGeneration),
    which generate() text fine when given only input_ids."""
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation=attn)
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, attn_implementation=attn)
    # Ministral ships max_length=262144 in generation_config; with our
    # explicit max_new_tokens every generate() warns. max_new_tokens is
    # always what we mean.
    try:
        model.generation_config.max_length = None
    except Exception:
        pass
    return model


def chat_prompt_ids(tok, msgs, thinking, add_generation_prompt=True,
                    system=None):
    """1-D input_ids tensor for a chat prompt, either transformers line.

    system: short system-prompt override. Ministral's template otherwise
    injects a 536-token default system prompt into EVERY sample — QA
    samples stop fitting seq_len and padded boilerplate dominates compute.
    Must be identical between training and eval."""
    if system is not None and (not msgs or msgs[0].get("role") != "system"):
        msgs = [{"role": "system", "content": system}] + msgs
    out = tok.apply_chat_template(
        msgs, add_generation_prompt=add_generation_prompt,
        return_tensors="pt", enable_thinking=thinking)
    if hasattr(out, "input_ids"):  # v5 BatchEncoding
        out = out.input_ids
    return out[0]
