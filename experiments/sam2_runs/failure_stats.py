"""Count failure frames and videos per model x split from the finished SAM 2 runs.

    python -m experiments.sam2_runs.failure_stats [--models tiny large]
        [--splits davis_val sav_val sav_test] --out results/failure_stats.csv

A failure is a labelled frame with mask IoU < 0.5 (see src/detection/labels.py). One row per
model x split. Runs that fail the completeness check are refused: nothing is written and the
exit status is 1. The per-frame label tables are not written.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.common.config import DATA_ROOT_ENV
from src.common.layout import validate_method
from src.detection.labels import count_skipped_objects, video_labels
from src.eval.check_complete import (
    MODELS,
    SPLIT_NAMES,
    SUBSET_FILE,
    check_run,
    load_video,
    split_videos,
)
from src.eval.score import load_data_config
from src.sam2_runner.predictor import REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "results" / "failure_stats.csv"
COLUMNS = (
    "model",
    "split",
    "videos",
    "objects",
    "skipped_objects",
    "labelled_frames",
    "failure_frames",
    "failure_rate",
    "objects_with_failure",
    "videos_with_failure",
)


def summarise(
    model: str, split: str, labels: pd.DataFrame, videos: int, skipped: int
) -> dict[str, object]:
    """Summarise the failure labels of one model and split in one row.

    Args:
        model: "tiny" or "large".
        split: Split name.
        labels: Per-frame label table from video_labels (columns video, object_id, frame_index,
            iou, failure), all videos of the split concatenated.
        videos: Number of videos in the split.
        skipped: Number of objects left out because they are never visible.

    Returns:
        Dict keyed by COLUMNS. failure_rate is failure frames over labelled frames (NaN if there
        are no labelled frames); objects counts objects with at least one labelled frame.
    """
    # Count failures per frame, then per object and per video that has at least one.
    failures = labels[labels["failure"]]
    frames = len(labels)
    return {
        "model": model,
        "split": split,
        "videos": videos,
        "objects": int(labels.groupby(["video", "object_id"]).ngroups),
        "skipped_objects": skipped,
        "labelled_frames": frames,
        "failure_frames": len(failures),
        "failure_rate": len(failures) / frames if frames else float("nan"),
        "objects_with_failure": int(failures.groupby(["video", "object_id"]).ngroups),
        "videos_with_failure": int(failures["video"].nunique()),
    }


def stats_for_run(
    model: str, split: str, prefix: str, subset_file: Path
) -> dict[str, object] | None:
    """Summary row for one run, or None (after saying why) if the run is incomplete."""
    # Refuse incomplete runs: failure counts from partial output would be misleading.
    cfg = load_data_config()
    method = validate_method(f"{prefix}_{model}")
    check = check_run(cfg, method, split, subset_file)
    if check.missing:
        sys.stdout.write(
            "\n".join(check.lines()) + f"\n{method}/{split}: refused, run is not complete\n"
        )
        return None
    output_root = cfg.data_root / cfg.output_root
    names = split_videos(cfg, split, subset_file)
    parts = []
    skipped = 0
    # Label every video against its ground truth and count skipped objects.
    for name in names:
        video = load_video(cfg, split, name)
        skipped += count_skipped_objects(video)
        parts.append(video_labels(video, output_root, method, split))
    labels = pd.concat(parts, ignore_index=True)
    return summarise(model, split, labels, len(names), skipped)


def write_atomic(frame: pd.DataFrame, out: Path) -> None:
    """Write a table as CSV through a temporary file, creating the folder if needed.

    Args:
        frame: Table to write.
        out: Destination CSV.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, out)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: write the failure-count table for the finished runs.

    Writes one row per model and split to --out and prints the table. Nothing is written if any
    run is incomplete.

    Args:
        argv: Command-line arguments; None means sys.argv.

    Returns:
        0 on success, 1 if any run is incomplete.

    Raises:
        SystemExit: Status 2 from argparse on bad arguments or when the data-root environment
            variable is not set.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    parser.add_argument("--method-prefix", default="sam2")
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    # Collect one row per model and split; all runs must succeed to write the file.
    rows = []
    for model in args.models:
        for split in args.splits:
            row = stats_for_run(model, split, args.method_prefix, args.subset_file)
            if row is not None:
                rows.append(row)
    if len(rows) != len(args.models) * len(args.splits):
        sys.stdout.write("failure_stats: nothing written, some runs are incomplete\n")
        return 1
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    write_atomic(table, args.out)
    sys.stdout.write(table.to_string(index=False) + f"\n-> {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
