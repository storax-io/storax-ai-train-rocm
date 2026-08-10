#!/usr/bin/env bash
# Run a traintest/ script under the WSL-native ROCm venv (Linux torch +
# transformers v5 — closest local approximation of the LUMI container).
# Usage: scripts/run_linux.sh <script.py> [args...]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HOME="${HF_HOME:-/mnt/c/Users/hs/storax-ai-train-test-win/hf-cache}"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
mkdir -p "$REPO/runs-linux"

script="$1"; shift
cd "$REPO/traintest"
exec "$REPO/.venv-rocm-wsl/bin/python" "$script" "$@"
