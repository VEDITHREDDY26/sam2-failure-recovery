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
    frames: dict[str, int]
    compute_s: dict[str, float]
    hardware: tuple[str, ...]


def _require(path: Path) -> Path:
    if not path.is_file():
        raise TableError(f"missing input file: {path}")
    return path


def load_scores(path: Path) -> list[ScoreRow]:
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
    grouped: dict[str, list[float]] = {}
    for row in sorted(rows, key=lambda r: (r.video, r.object)):
        grouped.setdefault(row.video, []).append(row.jf)
    return grouped


def count_changes(
    alone: dict[str, list[float]], fix: dict[str, list[float]]
) -> tuple[int, int, int]:
    """(improved, worse, unchanged) over videos; per-video J&F is the mean over its objects."""
    improved = worse = unchanged = 0
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
    rows = load_scores(results_dir / f"{spec.file_prefix}_{test_set}.csv")
    timing = load_timing(results_dir / f"{spec.file_prefix}_timing_{test_set}.csv")
    if {r.video for r in rows} != set(timing.frames):
        raise TableError(f"{spec.name}/{test_set}: score videos differ from timing videos")
    return rows, timing


def build_table(results_dir: Path) -> list[dict[str, object]]:
    table: list[dict[str, object]] = []
    for test_set in TEST_SETS:
        loaded = {spec.name: _load_method(results_dir, spec, test_set) for spec in METHODS}
        reference = {(r.video, r.object) for r in loaded[METHODS[0].name][0]}
        for spec in METHODS[1:]:
            if {(r.video, r.object) for r in loaded[spec.name][0]} != reference:
                raise TableError(f"{test_set}: {spec.name} scores different videos or objects")
        for spec in METHODS:
            rows, timing = loaded[spec.name]
            scores = per_video_scores(rows)
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
    return f"{100 * float(value):.1f}"


def _signed(value: str) -> str:
    return f"{100 * float(value):+.1f}"


def _escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def _cells(row: dict[str, str]) -> list[str]:
    jf = f"{_pct(row['JF'])} [{_pct(row['JF_low'])}, {_pct(row['JF_high'])}]"
    if row["delta_JF"] == "":
        delta, counts = "--", "--"
    else:
        low, high = _signed(row["delta_low"]), _signed(row["delta_high"])
        delta = f"{_signed(row['delta_JF'])} [{low}, {high}]"
        counts = f"{row['videos_improved']} / {row['videos_worse']} / {row['videos_unchanged']}"
    return [jf, delta, counts, f"{float(row['fps']):.2f}"]


def read_table(path: Path) -> list[dict[str, str]]:
    with _require(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TABLE_COLUMNS:
            raise TableError(f"{path}: columns are {reader.fieldnames}")
        return list(reader)


def latex_table(rows: Sequence[dict[str, str]]) -> str:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(latex_table(rows))
    os.replace(temp, path)


def main(argv: Sequence[str] | None = None) -> None:
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
