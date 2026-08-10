#!/usr/bin/env bash
# Run a traintest/ script under the Windows ROCm venv.
# Usage: scripts/run_win.sh <script.py> [args...]
# Syncs traintest/ to the C: staging dir first (Windows python can't be
# trusted with \\wsl.localhost cwd), sets HF cache + allocator env on the
# Windows side, and passes args through. Windows-path args like the run
# output dir must be given as C:\... (the callers do this).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Override for your machine: TRAINTEST_STAGE_WSL=/mnt/c/Users/<you>/... and
# TRAINTEST_STAGE_WIN='C:\Users\<you>\...' (same directory, both notations).
STAGE_WSL="${TRAINTEST_STAGE_WSL:-/mnt/c/Users/hs/storax-ai-train-test-win}"
STAGE_WIN="${TRAINTEST_STAGE_WIN:-C:\\Users\\hs\\storax-ai-train-test-win}"
PY="$STAGE_WIN\\.venv\\Scripts\\python.exe"

mkdir -p "$STAGE_WSL/traintest" "$STAGE_WSL/hf-cache" "$STAGE_WSL/runs"
cp -f "$REPO"/traintest/*.py "$STAGE_WSL/traintest/"
cp -f "$REPO"/data/*.json "$STAGE_WSL/traintest/" 2>/dev/null || true

# No quoting: WSL interop re-escapes embedded quotes into literal chars by
# the time cmd parses them, so args must stay space-free (they are: C:\
# paths without spaces, flags, model ids).
script="$1"; shift
winargs=""
for a in "$@"; do winargs="$winargs $a"; done

# WIN_EXTRA_ENV="NAME=VALUE" adds one env var on the Windows side
# (WSL env vars don't cross the interop boundary).
EXTRA=""
[ -n "${WIN_EXTRA_ENV:-}" ] && EXTRA="set $WIN_EXTRA_ENV&& "
exec /mnt/c/Windows/System32/cmd.exe /c \
  "cd /d $STAGE_WIN\\traintest && chcp 65001>nul&& set PYTHONIOENCODING=utf-8&& set HF_HOME=$STAGE_WIN\\hf-cache&& set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True&& $EXTRA$PY $script$winargs" \
  2> >(grep -v "UNC paths are not supported\|Defaulting to Windows directory\|CMD.EXE was started" >&2)
