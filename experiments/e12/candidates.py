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
    path = results_dir / f"verify_candidates_{model}_sav_val.csv"
    if not path.is_file():
        raise CandidateError(f"missing input file: {path}")
    table = pd.read_csv(path, dtype={"video": str})
    missing = [c for c in CANDIDATE_COLUMNS if c not in table.columns]
    if missing:
        raise CandidateError(f"{path}: missing columns {missing}")
    return table


def write_rows(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [repr(v) if isinstance(v, float) else v for v in (row[c] for c in columns)]
            )
