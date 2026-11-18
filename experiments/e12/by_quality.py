"""E12 part 1: change in J&F by how well SAM 2 alone tracked each object (analysis only).

    python -m experiments.e12.by_quality [--results-dir results] [--out results/e12_by_quality.csv]

Objects are binned by the J&F of SAM 2 alone (same model, same test set): lost < 0.5, partial
0.5 up to 0.8, good 0.8 and above, plus the bin all. The bins use ground truth through the
scores, so they describe results and are not available to the method. Each method is compared
with SAM 2 alone of its model; the interval resamples videos (each video keeps only its objects
in the bin). The all row must equal the delta_JF of the committed E2 / E7 / E5 / E8 tables.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from experiments.e2.table import UNCHANGED_TOL
from src.eval.bootstrap import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_delta

TEST_SETS = ("davis_val", "sav_test")
BINS = ("lost", "partial", "good", "all")
LOST_BELOW = 0.5
GOOD_FROM = 0.8
CHECK_TOL = 1e-9
OUT_COLUMNS = (
    "model",
    "test_set",
    "method",
    "oracle",
    "bin",
    "objects",
    "videos",
    "jf_alone",
    "jf_method",
    "delta_jf",
    "ci_low",
    "ci_high",
    "objects_improved",
    "objects_worse",
    "objects_unchanged",
)
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUT = Path("results/e12_by_quality.csv")


class E12Error(ValueError):
    """An E12 input is missing or inconsistent."""


@dataclass(frozen=True)
class Method:
    """One method to compare against SAM 2 alone.

    Attributes:
        model: SAM 2 size, "tiny" or "large"; picks the SAM 2 alone scores to compare with.
        name: Method name; the per-object scores are in results/<name>_<test_set>.csv.
        oracle: True when the method uses ground truth (marked as oracle in the output).
        table: Committed table (e2, e5, e7 or e8) whose delta the all row must reproduce.
        optional: When True, a missing score file skips the method instead of failing.
    """

    model: str
    name: str
    oracle: bool
    table: str
    optional: bool = False


METHODS = (
    Method("tiny", "fix_rules_csrt_tiny", False, "e2"),
    Method("tiny", "fix_rules_oracle_box_tiny", True, "e7"),
    Method("tiny", "fix_oracle_flags_csrt_tiny", True, "e7"),
    Method("tiny", "fix_oracle_flags_oracle_box_tiny", True, "e7"),
    Method("tiny", "fix_learned_csrt_tiny", False, "e5"),
    Method("tiny", "fix_continuous_rules_csrt_tiny", False, "e8", optional=True),
    Method("large", "fix_rules_csrt_large", False, "e2"),
)


def quality_bin(jf_alone: float) -> str:
    """Name the quality bin of an object from its SAM 2 alone J&F.

    Args:
        jf_alone: J&F of SAM 2 alone for the object, in [0, 1].

    Returns:
        "lost" below 0.5, "partial" from 0.5 up to 0.8, otherwise "good".
    """
    if jf_alone < LOST_BELOW:
        return "lost"
    if jf_alone < GOOD_FROM:
        return "partial"
    return "good"


def read_scores(path: Path) -> pd.DataFrame:
    """Read per-object J&F scores from a scorer CSV.

    Args:
        path: CSV with the columns video, object and J&F.

    Returns:
        Table with the columns video, object and jf (one row per object).

    Raises:
        E12Error: If the file is missing, lacks a column or has duplicate (video, object) rows.
    """
    if not path.is_file():
        raise E12Error(f"missing input file: {path}")
    # Keep video as text so names such as 00123 are not turned into numbers.
    table = pd.read_csv(path, dtype={"video": str})
    if not {"video", "object", "J&F"} <= set(table.columns):
        raise E12Error(f"{path}: columns are {list(table.columns)}")
    if table.duplicated(["video", "object"]).any():
        raise E12Error(f"{path}: duplicate (video, object) rows")
    return table.set_index(["video", "object"])["J&F"].rename("jf").reset_index()


def paired(alone: pd.DataFrame, method: pd.DataFrame, label: str) -> pd.DataFrame:
    """Join SAM 2 alone and a method per object and add each object's quality bin.

    Args:
        alone: Scores of SAM 2 alone (columns video, object, jf).
        method: Scores of the method (same columns).
        label: Method and test set, used in the error message.

    Returns:
        Table with jf_alone, jf_method and bin for every object.

    Raises:
        E12Error: If the two files do not score exactly the same objects.
    """
    # Outer join, so an object scored by only one side shows up as a missing value.
    joined = alone.merge(
        method, on=["video", "object"], how="outer", suffixes=("_alone", "_method")
    )
    if joined[["jf_alone", "jf_method"]].isna().any().any():
        raise E12Error(f"{label}: SAM 2 alone and the method score different objects")
    joined["bin"] = [quality_bin(v) for v in joined["jf_alone"]]
    return joined


def _groups(frame: pd.DataFrame, column: str) -> dict[str, list[float]]:
    """Collect the values of one column per video, as the bootstrap expects."""
    return {str(v): list(g[column]) for v, g in frame.groupby("video")}


def summarise(frame: pd.DataFrame, seed: int, resamples: int) -> dict[str, object]:
    """Summarise one group of objects: mean scores, change, interval and counts.

    Args:
        frame: Paired table with jf_alone and jf_method for the objects in the group.
        seed: Bootstrap random seed.
        resamples: Number of bootstrap resamples (videos are resampled, not objects).

    Returns:
        Row values for objects, videos, jf_alone, jf_method, delta_jf (method minus alone),
        ci_low, ci_high and the improved, worse and unchanged object counts.
    """
    # Per-object change in J&F; the counts use the same tolerance as the E2 table.
    delta = frame["jf_method"] - frame["jf_alone"]
    # Bootstrap over videos: each video keeps only the objects of this group.
    interval = bootstrap_delta(
        _groups(frame, "jf_method"), _groups(frame, "jf_alone"), seed=seed, resamples=resamples
    )
    return {
        "objects": len(frame),
        "videos": int(frame["video"].nunique()),
        "jf_alone": float(frame["jf_alone"].mean()),
        "jf_method": float(frame["jf_method"].mean()),
        "delta_jf": float(delta.mean()),
        "ci_low": interval.low,
        "ci_high": interval.high,
        "objects_improved": int((delta >= UNCHANGED_TOL).sum()),
        "objects_worse": int((delta <= -UNCHANGED_TOL).sum()),
        "objects_unchanged": int((delta.abs() < UNCHANGED_TOL).sum()),
    }


def empty_summary() -> dict[str, object]:
    """Return the row values for a bin that holds no objects (blank cells, zero counts)."""
    return {c: "" for c in OUT_COLUMNS[5:]} | {"objects": 0, "videos": 0}


def rows_for(
    frame: pd.DataFrame, model: str, test_set: str, method: Method, seed: int, resamples: int
) -> list[dict[str, object]]:
    """Build the four bin rows (lost, partial, good, all) for one method on one test set.

    Args:
        frame: Paired table from paired().
        model: SAM 2 size of the method.
        test_set: Test set name.
        method: The method being summarised.
        seed: Bootstrap random seed.
        resamples: Number of bootstrap resamples.

    Returns:
        One row dict per bin, in the order of BINS; the last one is the all row.
    """
    head = {"model": model, "test_set": test_set, "method": method.name, "oracle": method.oracle}
    rows = []
    for name in BINS:
        part = frame if name == "all" else frame[frame["bin"] == name]
        body = summarise(part, seed, resamples) if len(part) else empty_summary()
        rows.append({**head, "bin": name, **body})
    return rows


def reference_delta(results_dir: Path, method: Method, test_set: str) -> float:
    """Look up the delta J&F of a method in its committed table.

    Args:
        results_dir: Folder holding the committed tables.
        method: Method whose table (e2, e5, e7 or e8) is read.
        test_set: Test set name.

    Returns:
        The delta J&F versus SAM 2 alone stored in that table.

    Raises:
        E12Error: If the table is missing or does not hold exactly one matching row.
    """
    path = results_dir / f"{method.table}.csv"
    if not path.is_file():
        raise E12Error(f"missing input file: {path}")
    table = pd.read_csv(path)
    # The committed tables name their delta column differently.
    column = {"e2": "delta_JF", "e7": "delta_JF", "e5": "delta_vs_alone", "e8": "delta_JF"}[
        method.table
    ]
    # Pick the single row of this method and test set (and model, where there is one).
    chosen = table[(table["method"] == method.name) & (table["test_set"] == test_set)]
    if "model" in table.columns:
        chosen = chosen[chosen["model"] == method.model]
    if len(chosen) != 1:
        raise E12Error(f"{path}: expected one row for {method.name} on {test_set}")
    return float(chosen[column].iloc[0])


def check_all_row(results_dir: Path, method: Method, test_set: str, delta: float) -> None:
    """Check that the all-bin delta equals the delta in the committed table.

    Args:
        results_dir: Folder holding the committed tables.
        method: Method being checked.
        test_set: Test set name.
        delta: Delta J&F computed here for the all bin.

    Raises:
        E12Error: If the two values differ by more than CHECK_TOL.
    """
    expected = reference_delta(results_dir, method, test_set)
    if abs(expected - delta) > CHECK_TOL:
        raise E12Error(
            f"{method.name} on {test_set}: all-bin delta {delta} differs from {expected} "
            f"in results/{method.table}.csv"
        )


def build(
    results_dir: Path,
    methods: Sequence[Method] = METHODS,
    test_sets: Sequence[str] = TEST_SETS,
    seed: int = DEFAULT_SEED,
    resamples: int = DEFAULT_RESAMPLES,
) -> list[dict[str, object]]:
    """Compute the by-quality rows for every method and test set.

    Args:
        results_dir: Folder holding the per-object score CSVs and committed tables.
        methods: Methods to compare with SAM 2 alone.
        test_sets: Test sets to cover.
        seed: Bootstrap random seed.
        resamples: Number of bootstrap resamples.

    Returns:
        All output rows, four bins per method and test set.

    Raises:
        E12Error: If an input is missing or the all row disagrees with a committed table.
    """
    out: list[dict[str, object]] = []
    # One pass per method and test set; a method that was not run is skipped only if optional.
    for method in methods:
        for test_set in test_sets:
            alone_path = results_dir / f"sam2_{method.model}_{test_set}.csv"
            method_path = results_dir / f"{method.name}_{test_set}.csv"
            if method.optional and not method_path.is_file():
                continue
            label = f"{method.name}/{test_set}"
            # Pair per object, split into bins, then summarise each bin.
            frame = paired(read_scores(alone_path), read_scores(method_path), label)
            rows = rows_for(frame, method.model, test_set, method, seed, resamples)
            # Safety check: the all row must agree with the published table before it is kept.
            check_all_row(results_dir, method, test_set, cast(float, rows[-1]["delta_jf"]))
            out.extend(rows)
    return out


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write the rows to a CSV with the OUT_COLUMNS header; floats are written with repr."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUT_COLUMNS)
        for row in rows:
            writer.writerow(
                [repr(v) if isinstance(v, float) else v for v in (row[c] for c in OUT_COLUMNS)]
            )


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point: write the by-quality table.

    Reads the per-object score CSVs and committed tables in --results-dir. Writes --out
    (default results/e12_by_quality.csv) and prints the row count. Failures raise E12Error,
    which ends the program with a traceback and a non-zero exit code.

    Args:
        argv: Command line arguments; None means sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    # Build every row first so nothing is written when an input is wrong.
    rows = build(args.results_dir)
    write_rows(args.out, rows)
    print(f"{len(rows)} rows -> {args.out} (analysis; bins use ground-truth scores)")


if __name__ == "__main__":
    main()
