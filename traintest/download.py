"""Pre-fetch SmolLM3-3B into the Windows-side HF cache."""
import sys

from huggingface_hub import snapshot_download

path = snapshot_download(sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM3-3B")
print("DOWNLOADED", path)
