"""Shared loader for the SA-V val re-prompt candidates of the accept-all run (analysis only)."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

MODELS = ("tiny", "large")
REAL_FAILURE_BELOW = 0.5
CANDIDATE_COLUMNS = (
    "video",
    "object_id",
    "frame_index",
    "cand_score",
    "orig_score",
    "area_ratio",
    "box_iou",
    "mask_iou_orig",
    "iou_method",
    "iou_alone",
    "gain",
)


class CandidateError(ValueError):
    """A candidate file is missing or has the wrong columns."""


def load_candidates(results_dir: Path, model: str) -> pd.DataFrame:
    """Load the labelled re-prompt candidates of the accept-all run on SA-V val.

    Args:
        results_dir: Folder holding results/verify_candidates_<model>_sav_val.csv.
        model: SAM 2 size, "tiny" or "large".

    Returns:
        The candidate table (the video column is kept as text).

    Raises:
        CandidateError: If the file is missing or lacks a column of CANDIDATE_COLUMNS.
    """
    path = results_dir / f"verify_candidates_{model}_sav_val.csv"
    if not path.is_file():
        raise CandidateError(f"missing input file: {path}")
    # Keep video as text so names are not turned into numbers.
    table = pd.read_csv(path, dtype={"video": str})
    missing = [c for c in CANDIDATE_COLUMNS if c not in table.columns]
    if missing:
        raise CandidateError(f"{path}: missing columns {missing}")
    return table


def write_rows(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    """Write rows to a CSV with a header; floats are written with repr.

    Args:
        path: Output file; its folder is created if needed.
        columns: Header and column order.
        rows: One dict per row, holding every column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write each row in the column order given.
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [repr(v) if isinstance(v, float) else v for v in (row[c] for c in columns)]
            )
