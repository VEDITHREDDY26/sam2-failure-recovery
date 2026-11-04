"""Per-video, per-object score rows and their CSV file (columns video,object,J,F,J&F)."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SCORE_COLUMNS = ("video", "object", "J", "F", "J&F")


class ScoringError(RuntimeError):
    """The official scorer could not run or returned something unexpected."""


@dataclass(frozen=True)
class ScoreRow:
    video: str
    object: int
    j: float
    f: float

    @property
    def jf(self) -> float:
        return (self.j + self.f) / 2


def sorted_rows(rows: Iterable[ScoreRow]) -> list[ScoreRow]:
    return sorted(rows, key=lambda r: (r.video, r.object))


def write_scores_csv(path: str | Path, rows: Iterable[ScoreRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SCORE_COLUMNS)
        for row in sorted_rows(rows):
            writer.writerow([row.video, row.object, repr(row.j), repr(row.f), repr(row.jf)])
    os.replace(temp, path)


def read_scores_csv(path: str | Path) -> list[ScoreRow]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SCORE_COLUMNS:
            raise ValueError(f"{path}: columns are {reader.fieldnames}, expected {SCORE_COLUMNS}")
        return [
            ScoreRow(r["video"], int(r["object"]), float(r["J"]), float(r["F"])) for r in reader
        ]
