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
    """Return the folder that holds the SAM 2.1 checkpoints.

    The data root comes from the environment variable named by DATA_ROOT_ENV; the checkpoint
    folder name inside it comes from configs/models.yaml.

    Returns:
        Path of the checkpoint directory.

    Raises:
        SystemExit: If the data-root environment variable is not set.
    """
    root = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not root:
        raise SystemExit(f"{DATA_ROOT_ENV} is not set; point it at the Drive data folder")
    return checkpoint_dir(Path(root))


def main(argv: Sequence[str] | None = None) -> None:
    """Print the checkpoint directory on standard output.

    Args:
        argv: Command-line arguments; None means sys.argv. --print-dir is required.

    Raises:
        SystemExit: With the message as exit text (status 1) if the data-root environment
            variable is not set; argparse exits with status 2 on bad arguments.
    """
    parser = argparse.ArgumentParser(description="Print the checkpoint directory.")
    parser.add_argument("--print-dir", action="store_true", required=True)
    parser.parse_args(argv)
    sys.stdout.write(f"{resolve_dir()}\n")


if __name__ == "__main__":
    main()
