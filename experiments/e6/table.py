"""E6 table: does the fix matter less for the stronger model? E2 per model, with failure counts.

    python -m experiments.e6.table [--results-dir results] [--out results/e6.csv]
        [--models tiny large] [--splits davis_val sav_test]

No new runs: reads results/sam2_<model>_<split>.csv, results/fix_rules_csrt_<model>_<split>.csv,
its _frames companion (experiments.e2.score) and results/failure_stats.csv. Per model x test set:
J&F of SAM 2 alone and with the fix, the gain, videos improved / worse / unchanged (the same
0.001 rule as E2, imported from there), the failure frames and videos of SAM 2 alone and the
frames the fix recovered / broke.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.e2.table import (
    E2Error,
    alone_method,
    count_changes,
    fix_method,
    load_frames,
    load_scores,
    means,
    per_video_jf,
)

MODELS = ("tiny", "large")
TEST_SETS = ("davis_val", "sav_test")
STATS_COLUMNS = ("model", "split", "videos", "objects", "failure_frames", "videos_with_failure")
TABLE_COLUMNS = (
    "model",
    "test_set",
    "videos",
    "objects",
    "JF_alone",
    "JF_fix",
    "gain_JF",
    "videos_improved",
    "videos_worse",
    "videos_unchanged",
    "failure_frames",
    "failure_videos",
    "frames_recovered",
    "frames_broken",
)
DEFAULT_OUT = "results/e6.csv"


def load_failure_stats(path: Path) -> pd.DataFrame:
    """Read the failure statistics CSV and check its columns.

    Raises:
        E2Error: If the file is missing or lacks a required column.
    """
    if not path.is_file():
        raise E2Error(f"missing input file: {path}")
    stats = pd.read_csv(path)
    missing = set(STATS_COLUMNS) - set(stats.columns)
    if missing:
        raise E2Error(f"{path}: missing columns {sorted(missing)}")
    return stats


def stats_row(stats: pd.DataFrame, model: str, split: str) -> pd.Series:
    """Pick the single failure-statistics row of one model and split.

    Raises:
        E2Error: If there is not exactly one matching row.
    """
    rows = stats[(stats["model"] == model) & (stats["split"] == split)]
    if len(rows) != 1:
        raise E2Error(f"failure_stats.csv has {len(rows)} rows for {model}/{split}, expected 1")
    return rows.iloc[0]


def build_table(
    results_dir: Path, models: Sequence[str] = MODELS, splits: Sequence[str] = TEST_SETS
) -> list[dict[str, object]]:
    """Build the E6 rows: SAM 2 alone vs with the fix, per model and test set.

    Args:
        results_dir: Folder with the E2 score CSVs, the frames CSVs and failure_stats.csv.
        models: Model names to include (tiny, large).
        splits: Test sets to include.

    Returns:
        One row per (model, split) with the columns in TABLE_COLUMNS.

    Raises:
        E2Error: If an input is missing or invalid, or the two runs scored different objects.
    """
    # Failure counts are shared by all rows; per-run scores are loaded inside the loop.
    stats = load_failure_stats(results_dir / "failure_stats.csv")
    out: list[dict[str, object]] = []
    for model in models:
        for split in splits:
            label = f"{model}/{split}"
            alone = load_scores(results_dir / f"{alone_method(model)}_{split}.csv")
            fix_name = f"{fix_method(model)}_{split}"
            fix = load_scores(results_dir / f"{fix_name}.csv")
            # Both runs must cover the same objects for the gain to be a fair difference.
            if {(r.video, r.object) for r in alone} != {(r.video, r.object) for r in fix}:
                raise E2Error(f"{label}: SAM 2 alone and the fix score different videos or objects")
            frames = load_frames(results_dir / f"{fix_name}_frames.csv")
            stat = stats_row(stats, model, split)
            alone_jf = per_video_jf(alone)
            # Per-video J&F is compared with the same threshold as E2; gain is the difference of
            # means.
            improved, worse, unchanged = count_changes(alone_jf, per_video_jf(fix))
            base, new = means(alone)["J&F"], means(fix)["J&F"]
            out.append(
                {
                    "model": model,
                    "test_set": split,
                    "videos": len(alone_jf),
                    "objects": len(alone),
                    "JF_alone": base,
                    "JF_fix": new,
                    "gain_JF": new - base,
                    "videos_improved": improved,
                    "videos_worse": worse,
                    "videos_unchanged": unchanged,
                    "failure_frames": int(stat["failure_frames"]),
                    "failure_videos": int(stat["videos_with_failure"]),
                    "frames_recovered": int(frames["frames_recovered"].sum()),
                    "frames_broken": int(frames["frames_broken"].sum()),
                }
            )
    return out


def write_table(path: Path, table: Sequence[dict[str, object]]) -> None:
    """Write the table to a CSV file with the fixed column order.

    The file is written to a temporary name and then renamed, so a crash never leaves a
    half-written table. Floats are written with repr to keep full precision.

    Args:
        path: Output CSV path; parent folders are created.
        table: Rows as produced by build_table.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TABLE_COLUMNS)
        for row in table:
            writer.writerow(
                [repr(v) if isinstance(v, float) else v for v in (row[c] for c in TABLE_COLUMNS)]
            )
    os.replace(temp, path)


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point: build the E6 table CSV.

    Reads the E2 result CSVs and failure_stats.csv from --results-dir (default results) for the
    chosen --models and --splits, and writes the table to --out (default results/e6.csv).
    Exits with status 0 on success; invalid input raises E2Error and exits non-zero.

    Args:
        argv: Command line arguments; defaults to sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--splits", nargs="+", choices=TEST_SETS, default=list(TEST_SETS))
    args = parser.parse_args(argv)
    write_table(Path(args.out), build_table(Path(args.results_dir), args.models, args.splits))


if __name__ == "__main__":
    main()
