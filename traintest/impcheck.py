"""Surface the real exception behind transformers' lazy-import wrapper."""
import traceback

try:
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM  # noqa: F401
    print("OK direct modeling_auto import")
except Exception:
    traceback.print_exc()

try:
    from transformers import AutoModelForCausalLM  # noqa: F401
    print("OK top-level import")
except Exception:
    traceback.print_exc()
