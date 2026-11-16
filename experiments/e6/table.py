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
    if not path.is_file():
        raise E2Error(f"missing input file: {path}")
    stats = pd.read_csv(path)
    missing = set(STATS_COLUMNS) - set(stats.columns)
    if missing:
        raise E2Error(f"{path}: missing columns {sorted(missing)}")
    return stats


def stats_row(stats: pd.DataFrame, model: str, split: str) -> pd.Series:
    rows = stats[(stats["model"] == model) & (stats["split"] == split)]
    if len(rows) != 1:
        raise E2Error(f"failure_stats.csv has {len(rows)} rows for {model}/{split}, expected 1")
    return rows.iloc[0]


def build_table(
    results_dir: Path, models: Sequence[str] = MODELS, splits: Sequence[str] = TEST_SETS
) -> list[dict[str, object]]:
    stats = load_failure_stats(results_dir / "failure_stats.csv")
    out: list[dict[str, object]] = []
    for model in models:
        for split in splits:
            label = f"{model}/{split}"
            alone = load_scores(results_dir / f"{alone_method(model)}_{split}.csv")
            fix_name = f"{fix_method(model)}_{split}"
            fix = load_scores(results_dir / f"{fix_name}.csv")
            if {(r.video, r.object) for r in alone} != {(r.video, r.object) for r in fix}:
                raise E2Error(f"{label}: SAM 2 alone and the fix score different videos or objects")
            frames = load_frames(results_dir / f"{fix_name}_frames.csv")
            stat = stats_row(stats, model, split)
            alone_jf = per_video_jf(alone)
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--splits", nargs="+", choices=TEST_SETS, default=list(TEST_SETS))
    args = parser.parse_args(argv)
    write_table(Path(args.out), build_table(Path(args.results_dir), args.models, args.splits))


if __name__ == "__main__":
    main()
