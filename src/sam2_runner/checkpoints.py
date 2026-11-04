"""Where the SAM 2.1 checkpoints live, resolved through the config loader.

python -m src.sam2_runner.checkpoints --print-dir
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from src.common.config import DATA_ROOT_ENV
from src.sam2_runner.predictor import checkpoint_dir


def resolve_dir() -> Path:
    root = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not root:
        raise SystemExit(f"{DATA_ROOT_ENV} is not set; point it at the Drive data folder")
    return checkpoint_dir(Path(root))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the checkpoint directory.")
    parser.add_argument("--print-dir", action="store_true", required=True)
    parser.parse_args(argv)
    sys.stdout.write(f"{resolve_dir()}\n")


if __name__ == "__main__":
    main()
