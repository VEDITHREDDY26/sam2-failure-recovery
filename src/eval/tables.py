"""Main results table: J&F, change against SAM 2 alone, improved / worse videos, FPS.

    python -m src.eval.tables [--results-dir results] [--csv-out results/main_table.csv]
        [--tex-out results/main_table.tex]

Rows are the CSRT + GrabCut baseline, SAM 2 Tiny and Large alone, and each with the fix. Every
number comes from the per-video score and timing CSVs in the results directory; nothing is
written unless all ten score files and ten timing files exist and agree on videos and objects.
J&F is the mean over objects with a 95% bootstrap interval over videos; the change against SAM 2
alone of the same size is paired (same resampled videos). A video counts as improved or worse
when its J&F (mean over its objects) moved by at least UNCHANGED_TOL; worse videos are kept.
FPS is total frames over total compute seconds from the timing files. The .tex file is built
from the CSV only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.eval.bootstrap import bootstrap_delta, bootstrap_mean
from src.eval.rows import ScoreRow, read_scores_csv

TEST_SETS = ("davis_val", "sav_test")
SET_TITLES = {"davis_val": "DAVIS 2017 val", "sav_test": "SA-V test subset"}
UNCHANGED_TOL = 0.001
DEFAULT_RESULTS_DIR = "results"
DEFAULT_CSV_OUT = "results/main_table.csv"
DEFAULT_TEX_OUT = "results/main_table.tex"
TABLE_COLUMNS = (
    "method",
    "test_set",
    "videos",
    "objects",
    "JF",
    "JF_low",
    "JF_high",
    "delta_JF",
    "delta_low",
    "delta_high",
    "videos_improved",
    "videos_worse",
    "videos_unchanged",
    "fps",
    "hardware",
)
TIMING_REQUIRED = ("video", "frames", "compute_s")
HARDWARE_COLUMNS = (("gpu_name", "GPU"), ("cpu_model", "CPU"))


@dataclass(frozen=True)
class MethodSpec:
    """One row group of the main table.

    Attributes:
        name: Method name as used in the results.
        label: Name shown in the printed table.
        file_prefix: Prefix of this method's score and timing CSV file names.
        alone: Name of the SAM 2 run of the same size to compare against, or None if the method is
            not a fix.
    """

    name: str
    label: str
    file_prefix: str
    alone: str | None


METHODS = (
    MethodSpec("baseline_csrt_grabcut", "CSRT + GrabCut", "baseline", None),
    MethodSpec("sam2_tiny", "SAM 2 Tiny", "sam2_tiny", None),
    MethodSpec("fix_rules_csrt_tiny", "SAM 2 Tiny + fix", "fix_rules_csrt_tiny", "sam2_tiny"),
    MethodSpec("sam2_large", "SAM 2 Large", "sam2_large", None),
    MethodSpec("fix_rules_csrt_large", "SAM 2 Large + fix", "fix_rules_csrt_large", "sam2_large"),
)


class TableError(ValueError):
    """An input of the main table is missing or inconsistent."""


@dataclass(frozen=True)
class Timing:
    """Per-video timing of one method on one test set.

    Attributes:
        frames: Frame count of each video.
        compute_s: Compute seconds of each video (positive and finite).
        hardware: Sorted, de-duplicated device descriptions found in the file.
    """

    frames: dict[str, int]
    compute_s: dict[str, float]
    hardware: tuple[str, ...]


def _require(path: Path) -> Path:
    """Return the path if the file exists; raise TableError otherwise."""
    if not path.is_file():
        raise TableError(f"missing input file: {path}")
    return path


def load_scores(path: Path) -> list[ScoreRow]:
    """Read and validate a per-object score CSV.

    Args:
        path: CSV written by the scoring step; J, F and J&F are fractions in [0, 1].

    Returns:
        The score rows.

    Raises:
        TableError: If the file is missing, empty, has a duplicate (video, object) or a score
            outside [0, 1].
    """
    rows = read_scores_csv(_require(path))
    if not rows:
        raise TableError(f"{path}: no rows")
    keys = [(r.video, r.object) for r in rows]
    if len(set(keys)) != len(keys):
        raise TableError(f"{path}: duplicate (video, object) rows")
    for row in rows:
        if any(math.isnan(v) or not 0.0 <= v <= 1.0 for v in (row.j, row.f)):
            raise TableError(f"{path}: {row.video}/{row.object} has a score outside [0, 1]")
    return rows


def load_timing(path: Path) -> Timing:
    """Read and validate a per-video timing CSV.

    Args:
        path: CSV with video, frames, compute_s and a GPU or CPU name column.

    Returns:
        The Timing with frame counts, compute seconds and hardware names.

    Raises:
        TableError: If the file is missing, empty, lacks a required column or has an unusable row.
    """
    _require(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or ()
        device = next((c for c in HARDWARE_COLUMNS if c[0] in fields), None)
        if device is None or any(c not in fields for c in TIMING_REQUIRED):
            raise TableError(f"{path}: needs {TIMING_REQUIRED} and a device column")
        frames: dict[str, int] = {}
        compute: dict[str, float] = {}
        hardware: list[str] = []
        # Each video must appear once with a positive frame count and a positive, finite time.
        for r in reader:
            video = r["video"]
            n, secs = int(r["frames"]), float(r["compute_s"])
            if video in frames or n <= 0 or not secs > 0 or math.isinf(secs):
                raise TableError(f"{path}: unusable row for {video}")
            frames[video], compute[video] = n, secs
            name = f"{device[1]}: {r[device[0]]}"
            if name not in hardware:
                hardware.append(name)
    if not frames:
        raise TableError(f"{path}: no rows")
    return Timing(frames, compute, tuple(sorted(hardware)))


def per_video_scores(rows: Sequence[ScoreRow]) -> dict[str, list[float]]:
    """Group J&F values by video, objects in sorted order."""
    grouped: dict[str, list[float]] = {}
    for row in sorted(rows, key=lambda r: (r.video, r.object)):
        grouped.setdefault(row.video, []).append(row.jf)
    return grouped


def count_changes(
    alone: dict[str, list[float]], fix: dict[str, list[float]]
) -> tuple[int, int, int]:
    """(improved, worse, unchanged) over videos; per-video J&F is the mean over its objects."""
    improved = worse = unchanged = 0
    # Per-video change in J&F (mean over its objects); changes smaller than UNCHANGED_TOL
    # count as unchanged.
    for video, values in alone.items():
        delta = sum(fix[video]) / len(fix[video]) - sum(values) / len(values)
        if delta >= UNCHANGED_TOL:
            improved += 1
        elif delta <= -UNCHANGED_TOL:
            worse += 1
        else:
            unchanged += 1
    return improved, worse, unchanged


def _load_method(
    results_dir: Path, spec: MethodSpec, test_set: str
) -> tuple[list[ScoreRow], Timing]:
    """Load the score and timing files of one method on one test set.

    Raises:
        TableError: If a file is invalid or the score and timing files cover different videos.
    """
    rows = load_scores(results_dir / f"{spec.file_prefix}_{test_set}.csv")
    timing = load_timing(results_dir / f"{spec.file_prefix}_timing_{test_set}.csv")
    if {r.video for r in rows} != set(timing.frames):
        raise TableError(f"{spec.name}/{test_set}: score videos differ from timing videos")
    return rows, timing


def build_table(results_dir: Path) -> list[dict[str, object]]:
    """Compute every row of the main table from the per-video CSVs in the results folder.

    Args:
        results_dir: Folder with the score and timing CSVs of all methods and test sets.

    Returns:
        One dict per (method, test set) with the table columns. Fix rows also hold the paired
        change against SAM 2 alone and the improved / worse / unchanged video counts.

    Raises:
        TableError: If any input is missing or methods do not score the same videos and objects.
    """
    table: list[dict[str, object]] = []
    for test_set in TEST_SETS:
        # Load all methods first, so any missing file stops the build before output.
        loaded = {spec.name: _load_method(results_dir, spec, test_set) for spec in METHODS}
        # Every method must score exactly the same (video, object) pairs as the baseline.
        reference = {(r.video, r.object) for r in loaded[METHODS[0].name][0]}
        for spec in METHODS[1:]:
            if {(r.video, r.object) for r in loaded[spec.name][0]} != reference:
                raise TableError(f"{test_set}: {spec.name} scores different videos or objects")
        for spec in METHODS:
            rows, timing = loaded[spec.name]
            scores = per_video_scores(rows)
            # Mean J&F over objects with a bootstrap interval over videos.
            interval = bootstrap_mean(scores)
            row: dict[str, object] = {
                "method": spec.name,
                "test_set": test_set,
                "videos": len(scores),
                "objects": len(rows),
                "JF": interval.estimate,
                "JF_low": interval.low,
                "JF_high": interval.high,
                "delta_JF": "",
                "delta_low": "",
                "delta_high": "",
                "videos_improved": "",
                "videos_worse": "",
                "videos_unchanged": "",
                "fps": sum(timing.frames.values()) / sum(timing.compute_s.values()),
                "hardware": "; ".join(timing.hardware),
            }
            # Fix rows: paired change against SAM 2 alone, plus improved / worse / unchanged videos.
            if spec.alone is not None:
                alone = per_video_scores(loaded[spec.alone][0])
                delta = bootstrap_delta(scores, alone)
                improved, worse, unchanged = count_changes(alone, scores)
                row |= {
                    "delta_JF": delta.estimate,
                    "delta_low": delta.low,
                    "delta_high": delta.high,
                    "videos_improved": improved,
                    "videos_worse": worse,
                    "videos_unchanged": unchanged,
                }
            table.append(row)
    return table


def write_csv(path: Path, table: Sequence[dict[str, object]]) -> None:
    """Write the table rows to a CSV file atomically; floats are written with full precision."""
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


def _pct(value: str) -> str:
    """Format a fraction (as text) as a percentage with one decimal."""
    return f"{100 * float(value):.1f}"


def _signed(value: str) -> str:
    """Format a fraction (as text) as a signed percentage with one decimal."""
    return f"{100 * float(value):+.1f}"


def _escape(text: str) -> str:
    """Escape the LaTeX special characters &, _ and % in text."""
    return text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def _cells(row: dict[str, str]) -> list[str]:
    """Build the four table cells of one row: J&F with interval, change, video counts and FPS.

    Args:
        row: A table row read back from the CSV (all values are text).
    """
    jf = f"{_pct(row['JF'])} [{_pct(row['JF_low'])}, {_pct(row['JF_high'])}]"
    if row["delta_JF"] == "":
        delta, counts = "--", "--"
    else:
        low, high = _signed(row["delta_low"]), _signed(row["delta_high"])
        delta = f"{_signed(row['delta_JF'])} [{low}, {high}]"
        counts = f"{row['videos_improved']} / {row['videos_worse']} / {row['videos_unchanged']}"
    return [jf, delta, counts, f"{float(row['fps']):.2f}"]


def read_table(path: Path) -> list[dict[str, str]]:
    """Read the main table CSV back as text rows.

    Raises:
        TableError: If the file is missing or its columns are not TABLE_COLUMNS.
    """
    with _require(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TABLE_COLUMNS:
            raise TableError(f"{path}: columns are {reader.fieldnames}")
        return list(reader)


def latex_table(rows: Sequence[dict[str, str]]) -> str:
    """Render the table as a LaTeX tabular, one line per method with both test sets side by side.

    Args:
        rows: Rows read from the CSV.

    Raises:
        TableError: If a method has no row for a test set.
    """
    # Rows are looked up by (method, test set) so both test sets share one table line.
    by_key = {(r["method"], r["test_set"]): r for r in rows}
    lines = [
        r"\begin{tabular}{l" + "cccr" * len(TEST_SETS) + "l}",
        r"\toprule",
        "Method & "
        + " & ".join(rf"\multicolumn{{4}}{{c}}{{{SET_TITLES[s]}}}" for s in TEST_SETS)
        + r" & Hardware \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        " & "
        + " & ".join(
            [r"J\&F (\%)", r"$\Delta$ J\&F", "better / worse / same", "FPS"] * len(TEST_SETS)
        )
        + r" & \\",
        r"\midrule",
    ]
    for spec in METHODS:
        # One line per method: the cells of each test set, then the combined hardware.
        cells: list[str] = []
        hardware: list[str] = []
        for test_set in TEST_SETS:
            row = by_key.get((spec.name, test_set))
            if row is None:
                raise TableError(f"main table has no row for {spec.name}/{test_set}")
            cells += _cells(row)
            hardware += [h for h in row["hardware"].split("; ") if h not in hardware]
        lines.append(
            " & ".join([_escape(spec.label), *cells, _escape("; ".join(hardware))]) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def write_tex(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write the LaTeX table to a file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(latex_table(rows))
    os.replace(temp, path)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: build the main results table.

    Reads the per-video score and timing CSVs in --results-dir and writes the table CSV
    (--csv-out) and its LaTeX version (--tex-out), which is built from the CSV only. Nothing is
    written if an input is missing or inconsistent (TableError is raised, so the exit status is
    non-zero); it returns normally on success.

    Args:
        argv: Command-line arguments; sys.argv is used when None.
    """
    parser = argparse.ArgumentParser(description="Build the main results table and its LaTeX.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--tex-out", default=DEFAULT_TEX_OUT)
    args = parser.parse_args(argv)
    table = build_table(Path(args.results_dir))
    csv_out = Path(args.csv_out)
    write_csv(csv_out, table)
    write_tex(Path(args.tex_out), read_table(csv_out))


if __name__ == "__main__":
    main()
