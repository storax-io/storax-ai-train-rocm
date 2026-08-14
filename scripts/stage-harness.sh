#!/bin/bash
# Stage a committed-tree snapshot of the harness for training hosts.
# Same provenance rules as the dataset trainpack: content comes from git
# archive (committed files only — a dirty tree cannot leak), stamped with
# the commit id that jobs verify and log. Usage:
#   scripts/stage-harness.sh <dest-dir>
set -euo pipefail
dest=${1:?usage: stage-harness.sh <dest-dir>}
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is dirty — commit first so HARNESS_COMMIT is the truth" >&2
    exit 2
fi
mkdir -p "$dest"
git archive HEAD | tar -x -C "$dest"
git rev-parse HEAD > "$dest/HARNESS_COMMIT"
echo "harness staged: $dest ($(cat "$dest/HARNESS_COMMIT" | cut -c1-9))"
