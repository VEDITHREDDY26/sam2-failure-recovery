"""E1 table: baseline vs SAM 2 Tiny / Large on both test sets (J, F, J&F, FPS)."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.eval.rows import ScoreRow, read_scores_csv

METHODS = ("baseline_csrt_grabcut", "sam2_tiny", "sam2_large")
TEST_SETS = ("davis_val", "sav_test")
TABLE_COLUMNS = ("method", "test_set", "videos", "objects", "J", "F", "J&F", "fps", "hardware")
SCORE_FILES = {
    "baseline_csrt_grabcut": "baseline_{test_set}.csv",
    "sam2_tiny": "sam2_tiny_{test_set}.csv",
    "sam2_large": "sam2_large_{test_set}.csv",
}
TIMING_FILES = {
    "baseline_csrt_grabcut": "baseline_timing_{test_set}.csv",
    "sam2_tiny": "sam2_tiny_timing_{test_set}.csv",
    "sam2_large": "sam2_large_timing_{test_set}.csv",
}
TIMING_REQUIRED = ("video", "frames", "compute_s")
HARDWARE_COLUMNS = ("gpu_name", "cpu_model")
HARDWARE_SEPARATOR = ";"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUT = "results/e1.csv"
DEFAULT_FIGURE_DIR = "figures"


class E1Error(ValueError):
    """An E1 input is missing or inconsistent."""


@dataclass(frozen=True)
class Timing:
    frames: dict[str, int]
    compute_s: dict[str, float]
    hardware: tuple[str, ...]


def _require(path: Path) -> Path:
    if not path.is_file():
        raise E1Error(f"missing input file: {path}")
    return path


def load_scores(path: Path) -> list[ScoreRow]:
    rows = read_scores_csv(_require(path))
    for row in rows:
        for name, value in (("J", row.j), ("F", row.f)):
            if math.isnan(value) or not 0.0 <= value <= 1.0:
                raise E1Error(f"{path}: {row.video}/{row.object} has {name}={value}, not in [0,1]")
    keys = [(r.video, r.object) for r in rows]
    if len(set(keys)) != len(keys):
        raise E1Error(f"{path}: duplicate (video, object) rows")
    if not rows:
        raise E1Error(f"{path}: no rows")
    return rows


def load_timing(path: Path) -> Timing:
    _require(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or ()
        missing = [c for c in TIMING_REQUIRED if c not in fields]
        hardware_col = next((c for c in HARDWARE_COLUMNS if c in fields), None)
        if missing or hardware_col is None:
            raise E1Error(
                f"{path}: needs columns {TIMING_REQUIRED} and one of {HARDWARE_COLUMNS}; "
                f"found {tuple(fields)}"
            )
        frames: dict[str, int] = {}
        compute: dict[str, float] = {}
        hardware: list[str] = []
        for r in reader:
            video = r["video"]
            if video in frames:
                raise E1Error(f"{path}: duplicate video {video}")
            n, secs = int(r["frames"]), float(r["compute_s"])
            if n <= 0 or not secs > 0 or math.isinf(secs):
                raise E1Error(f"{path}: {video} has frames={n}, compute_s={secs}")
            frames[video], compute[video] = n, secs
            if r[hardware_col] not in hardware:
                hardware.append(r[hardware_col])
    if not frames:
        raise E1Error(f"{path}: no rows")
    return Timing(frames, compute, tuple(sorted(hardware)))


def summarize(rows: Sequence[ScoreRow], timing: Timing) -> dict[str, object]:
    return {
        "videos": len({r.video for r in rows}),
        "objects": len(rows),
        "J": sum(r.j for r in rows) / len(rows),
        "F": sum(r.f for r in rows) / len(rows),
        "J&F": sum(r.jf for r in rows) / len(rows),
        "fps": sum(timing.frames.values()) / sum(timing.compute_s.values()),
        "hardware": HARDWARE_SEPARATOR.join(timing.hardware),
    }


def build_table(results_dir: Path) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for test_set in TEST_SETS:
        loaded = {}
        for method in METHODS:
            rows = load_scores(results_dir / SCORE_FILES[method].format(test_set=test_set))
            timing = load_timing(results_dir / TIMING_FILES[method].format(test_set=test_set))
            if {r.video for r in rows} != set(timing.frames):
                raise E1Error(f"{method}/{test_set}: score videos differ from timing videos")
            loaded[method] = (rows, timing)
        reference = {(r.video, r.object) for r in loaded[METHODS[0]][0]}
        for method in METHODS[1:]:
            if {(r.video, r.object) for r in loaded[method][0]} != reference:
                raise E1Error(f"{test_set}: {method} and {METHODS[0]} score different objects")
        for method in METHODS:
            table.append({"method": method, "test_set": test_set, **summarize(*loaded[method])})
    return table


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
    from experiments.e1.figure import plot_jf_fps

    parser = argparse.ArgumentParser(description="Build the E1 table and figure.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args(argv)
    out = Path(args.out)
    write_table(out, build_table(Path(args.results_dir)))
    plot_jf_fps(out, Path(args.figure_dir))


if __name__ == "__main__":
    main()
