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
    """Draw a seeded sample of video names.

    Args:
        names: All candidate video names (must be distinct).
        size: Number of videos to draw.
        seed: Random seed.

    Returns:
        The drawn names, sorted.

    Raises:
        ValueError: If names has duplicates or size is not between 1 and the number of names.
    """
    # Sort first so the same seed always gives the same sample.
    ordered = sorted(set(names))
    if len(ordered) != len(names):
        raise ValueError("duplicate video names")
    if not 0 < size <= len(ordered):
        raise ValueError(f"cannot draw {size} of {len(ordered)} videos")
    return sorted(random.Random(seed).sample(ordered, size))


def read_names(path: Path) -> list[str]:
    """Read video names from a text file, one per line, ignoring blank lines."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_subset(path: Path = OUT) -> list[str]:
    """Load the committed tuning subset of SA-V val videos.

    Args:
        path: Subset file, one video name per line.

    Returns:
        The SIZE distinct video names.

    Raises:
        ValueError: If the file does not hold exactly SIZE distinct names.
    """
    names = read_names(path)
    if len(names) != SIZE or len(set(names)) != SIZE:
        raise ValueError(f"{path}: expected {SIZE} distinct video names, got {len(names)}")
    return names


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point: draw the tuning subset and write it to a file.

    Reads the video names from --names-file, or from the SA-V val list under the data root when
    no file is given (the data root variable must then be set, otherwise the usage error code 2
    is returned). Writes the sorted sample to --out and prints how many names were written.

    Args:
        argv: Command line arguments; None means sys.argv.

    Raises:
        ValueError: If the sample cannot be drawn (see draw).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--names-file", type=Path, default=None)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    # Without a names file, list the SA-V val videos from the data root.
    if args.names_file is None:
        from src.eval.check_complete import split_videos
        from src.sam2_runner.smoke import load_data_config

        if not os.environ.get(DATA_ROOT_ENV):
            parser.error(f"{DATA_ROOT_ENV} is not set (or pass --names-file)")
        names = split_videos(load_data_config(), "sav_val")
    else:
        names = read_names(args.names_file)
    # Draw, write one name per line and report.
    chosen = draw(names, args.size, args.seed)
    args.out.write_text("\n".join(chosen) + "\n")
    print(f"{len(chosen)} of {len(names)} videos written to {args.out}")


if __name__ == "__main__":
    main()
