"""Collect the per-video timing rows of a finished baseline run into one repo CSV.

    python -m src.baseline.timing --split davis_val --out results/baseline_timing_davis_val.csv

Every video of the split must have a timing row; a missing one is an error.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.baseline.config import METHOD
from src.baseline.run import SPLITS, load_data_config, split_videos
from src.baseline.runner import TIMING_COLUMNS, timing_path
from src.common.config import DATA_ROOT_ENV, DataConfig
from src.common.layout import validate_method


def collect_timing(cfg: DataConfig, split: str, method: str = METHOD) -> pd.DataFrame:
    output_root = cfg.data_root / cfg.output_root
    names = split_videos(cfg, split)
    paths = [timing_path(output_root, method, split, n) for n in names]
    missing = [p.stem for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"no timing for {len(missing)} video(s): {missing[:10]}")
    table = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return table[list(TIMING_COLUMNS)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", default=METHOD)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    table = collect_timing(load_data_config(), args.split, validate_method(args.method))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    total_fps = table["frames"].sum() / table["compute_s"].sum()
    print(f"{len(table)} videos, {total_fps:.1f} fps overall -> {args.out}")


if __name__ == "__main__":
    main()
