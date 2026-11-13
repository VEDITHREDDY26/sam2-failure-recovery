"""Rerun check: do five seeded DAVIS val videos reproduce when the real system is run again?

    python -m experiments.e2.rerun_check [--results-dir results] [--out results/e2_rerun_check.csv]

Compares results/fix_rules_csrt_tiny_davis_val.csv (the E2 run) with
results/fix_rerun_rules_csrt_tiny_davis_val.csv (the rerun, scored with experiments.e2.score
--videos) on the five videos fixed in docs/decisions.md before any run, and the re-prompt counts
from the two events files. Tolerance (also fixed beforehand): per video |delta J&F| <= 0.01 and the
same number of re-prompts; mean |delta J&F| over the videos <= 0.005. Exit status 1 if anything is
outside; the task document then explains each difference.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.e2.table import E2Error, load_events, load_scores, per_video_jf

SEEDED_VIDEOS = ("bmx-trees", "cows", "dog", "mbike-trick", "paragliding-launch")
SEED = 5415
SAMPLE_SIZE = 5
E2_NAME = "fix_rules_csrt_tiny_davis_val"
RERUN_NAME = "fix_rerun_rules_csrt_tiny_davis_val"
PER_VIDEO_TOLERANCE = 0.01
MEAN_TOLERANCE = 0.005
ROUNDING = 1e-9
CHECK_COLUMNS = (
    "video",
    "jf_e2",
    "jf_rerun",
    "abs_delta",
    "reprompts_e2",
    "reprompts_rerun",
    "within_tolerance",
)
DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUT = "results/e2_rerun_check.csv"


@dataclass(frozen=True)
class VideoAgreement:
    video: str
    jf_e2: float
    jf_rerun: float
    reprompts_e2: int
    reprompts_rerun: int

    @property
    def abs_delta(self) -> float:
        return abs(self.jf_e2 - self.jf_rerun)

    @property
    def within(self) -> bool:
        return (
            self.abs_delta <= PER_VIDEO_TOLERANCE + ROUNDING
            and self.reprompts_e2 == self.reprompts_rerun
        )

    def as_row(self) -> list[object]:
        return [
            self.video,
            repr(self.jf_e2),
            repr(self.jf_rerun),
            repr(self.abs_delta),
            self.reprompts_e2,
            self.reprompts_rerun,
            self.within,
        ]


def reprompt_counts(events: pd.DataFrame) -> dict[str, int]:
    return {
        str(v): int(n)
        for v, n in events[events["action"] == "reprompted"].groupby("video").size().items()
    }


def compare(results_dir: Path, videos: Sequence[str] = SEEDED_VIDEOS) -> list[VideoAgreement]:
    e2 = per_video_jf(load_scores(results_dir / f"{E2_NAME}.csv"))
    rerun = per_video_jf(load_scores(results_dir / f"{RERUN_NAME}.csv"))
    missing = [v for v in videos if v not in e2 or v not in rerun]
    if missing:
        raise E2Error(f"videos missing from the E2 run or the rerun: {missing}")
    count_e2 = reprompt_counts(load_events(results_dir / f"events_{E2_NAME}.csv"))
    count_rerun = reprompt_counts(load_events(results_dir / f"events_{RERUN_NAME}.csv"))
    return [
        VideoAgreement(v, e2[v], rerun[v], count_e2.get(v, 0), count_rerun.get(v, 0))
        for v in videos
    ]


def mean_abs_delta(rows: Sequence[VideoAgreement]) -> float:
    return sum(r.abs_delta for r in rows) / len(rows)


def passes(rows: Sequence[VideoAgreement]) -> bool:
    return all(r.within for r in rows) and mean_abs_delta(rows) <= MEAN_TOLERANCE + ROUNDING


def write_check(path: Path, rows: Sequence[VideoAgreement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CHECK_COLUMNS)
        writer.writerows(r.as_row() for r in rows)
    os.replace(temp, path)


def summary_line(rows: Sequence[VideoAgreement]) -> str:
    outside = [r.video for r in rows if not r.within]
    verdict = "within tolerance" if passes(rows) else f"OUTSIDE tolerance ({outside or 'mean'})"
    return f"{len(rows)} videos, mean |delta J&F| {mean_abs_delta(rows):.5f}: {verdict}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    rows = compare(Path(args.results_dir))
    write_check(Path(args.out), rows)
    sys.stdout.write(summary_line(rows) + f" -> {args.out}\n")
    return 0 if passes(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
