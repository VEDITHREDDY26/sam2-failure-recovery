"""Draw the SA-V val videos on which the Large verification rule is tuned.

    python -m experiments.verify.draw_subset [--names-file sav_val.txt] [--size 60] [--seed 5415]

The names come from the SA-V val video list (one per line; by default the list under the data
root). They are sorted, then a seeded sample is drawn and written sorted, one per line, to
experiments/verify/sav_val_large_subset.txt. The draw uses no results of any run.
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Sequence
from pathlib import Path

from src.common.config import DATA_ROOT_ENV

SEED = 5415
SIZE = 60
OUT = Path(__file__).resolve().parent / "sav_val_large_subset.txt"


def draw(names: Sequence[str], size: int = SIZE, seed: int = SEED) -> list[str]:
    ordered = sorted(set(names))
    if len(ordered) != len(names):
        raise ValueError("duplicate video names")
    if not 0 < size <= len(ordered):
        raise ValueError(f"cannot draw {size} of {len(ordered)} videos")
    return sorted(random.Random(seed).sample(ordered, size))


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_subset(path: Path = OUT) -> list[str]:
    names = read_names(path)
    if len(names) != SIZE or len(set(names)) != SIZE:
        raise ValueError(f"{path}: expected {SIZE} distinct video names, got {len(names)}")
    return names


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--names-file", type=Path, default=None)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.names_file is None:
        from src.eval.check_complete import split_videos
        from src.sam2_runner.smoke import load_data_config

        if not os.environ.get(DATA_ROOT_ENV):
            parser.error(f"{DATA_ROOT_ENV} is not set (or pass --names-file)")
        names = split_videos(load_data_config(), "sav_val")
    else:
        names = read_names(args.names_file)
    chosen = draw(names, args.size, args.seed)
    args.out.write_text("\n".join(chosen) + "\n")
    print(f"{len(chosen)} of {len(names)} videos written to {args.out}")


if __name__ == "__main__":
    main()
