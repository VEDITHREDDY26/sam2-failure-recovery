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
    """Scores of one object of one video.

    Attributes:
        video: Video name.
        object: Object id.
        j: Region similarity J as a fraction in [0, 1].
        f: Boundary accuracy F as a fraction in [0, 1].
    """

    video: str
    object: int
    j: float
    f: float

    @property
    def jf(self) -> float:
        """J&F: the mean of J and F, as a fraction in [0, 1]."""
        return (self.j + self.f) / 2


def sorted_rows(rows: Iterable[ScoreRow]) -> list[ScoreRow]:
    """Rows sorted by video name, then object id."""
    return sorted(rows, key=lambda r: (r.video, r.object))


def write_scores_csv(path: str | Path, rows: Iterable[ScoreRow]) -> None:
    """Write score rows to a CSV with the columns video,object,J,F,J&F.

    The file is written to a temporary name and moved into place, and floats are written with
    repr so they read back exactly.

    Args:
        path: Destination CSV; its folder is created if needed.
        rows: Rows to write, sorted by video and object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and rename, so a half-written CSV never appears.
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SCORE_COLUMNS)
        for row in sorted_rows(rows):
            writer.writerow([row.video, row.object, repr(row.j), repr(row.f), repr(row.jf)])
    os.replace(temp, path)


def read_scores_csv(path: str | Path) -> list[ScoreRow]:
    """Read a score CSV written by write_scores_csv.

    J&F is recomputed from J and F rather than read.

    Args:
        path: CSV to read.

    Raises:
        ValueError: If the columns are not video,object,J,F,J&F.
    """
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        # Reject files with other columns instead of guessing what they mean.
        if tuple(reader.fieldnames or ()) != SCORE_COLUMNS:
            raise ValueError(f"{path}: columns are {reader.fieldnames}, expected {SCORE_COLUMNS}")
        return [
            ScoreRow(r["video"], int(r["object"]), float(r["J"]), float(r["F"])) for r in reader
        ]
