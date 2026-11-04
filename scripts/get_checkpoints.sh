#!/usr/bin/env bash
# Download the SAM 2.1 Tiny and Large checkpoints into <SAM2FR_DATA_ROOT>/<checkpoint_dir>.
# Files that already exist and are non-empty are skipped.
# Usage: scripts/get_checkpoints.sh [--print-dir]
set -euo pipefail

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
FILES=(sam2.1_hiera_tiny.pt sam2.1_hiera_large.pt)

cd "$(dirname "$0")/.."
if [[ -z "${SAM2FR_DATA_ROOT:-}" ]]; then
    echo "SAM2FR_DATA_ROOT is not set; point it at the Drive data folder" >&2
    exit 1
fi
PYTHON="${PYTHON:-python3}"
DIR="$("$PYTHON" -m src.sam2_runner.checkpoints --print-dir)"

if [[ "${1:-}" == "--print-dir" ]]; then
    echo "$DIR"
    exit 0
fi

mkdir -p "$DIR"
for name in "${FILES[@]}"; do
    target="$DIR/$name"
    if [[ -s "$target" ]]; then
        echo "skip $name (already present)"
        continue
    fi
    echo "downloading $name"
    curl -fL --retry 3 -o "$target.part" "$BASE_URL/$name"
    mv "$target.part" "$target"
done
echo "checkpoints in $DIR"
