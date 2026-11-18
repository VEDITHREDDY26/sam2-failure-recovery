"""E3 table: J&F and detector recall with each of the four rules turned off in turn (Tiny).

    python -m experiments.e3.table [--results-dir results] [--out results/e3.csv]
        [--splits davis_val sav_test]

Reads results/sam2_tiny_<split>.csv, the scored recovery runs results/<method>_<split>.csv with
their events_ files (experiments.e2.score) and results/e3_detector_<split>.csv (experiments.e3.
detector_eval). removed_rule "none" is the full system (the E2 Tiny run fix_rules_csrt_tiny).
Deltas are J&F differences, versus the full system and versus SAM 2 alone. Detector precision and
recall are over annotation windows with the frozen thresholds. All runs must score the same
videos and objects.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.e2.table import load_events, load_scores, means
from experiments.e3.detector_eval import NONE, REMOVED, RESULT_COLUMNS, detector_path
from src.eval.rows import ScoreRow

TEST_SETS = ("davis_val", "sav_test")
MODEL = "tiny"
ALONE = f"sam2_{MODEL}"
TABLE_COLUMNS = (
    "removed_rule",
    "method",
    "test_set",
    "objects",
    "J",
    "F",
    "J&F",
    "delta_vs_full",
    "delta_vs_alone",
    "reprompts",
    "windows",
    "positives",
    "flagged",
    "tp",
    "precision",
    "recall",
)
DEFAULT_OUT = "results/e3.csv"


class E3Error(ValueError):
    """An E3 input is missing or inconsistent."""


def method_for(removed: str) -> str:
    """Name of the E2 recovery run that has one rule removed.

    Args:
        removed: Rule name, or "none" for the full system.

    Returns:
        Method name such as "fix_rules_csrt_tiny" or "fix_rules_no_<rule>_csrt_tiny".
    """
    if removed == NONE:
        return f"fix_rules_csrt_{MODEL}"
    return f"fix_rules_no_{removed}_csrt_{MODEL}"


def load_detector(path: Path) -> pd.DataFrame:
    """Load and validate the E3 detector result CSV of one split.

    Args:
        path: Path of results/e3_detector_<split>.csv.

    Returns:
        The table, with one row per removed rule in the order of REMOVED.

    Raises:
        E3Error: If the file is missing, or its columns or rows are not the expected ones.
    """
    if not path.is_file():
        raise E3Error(f"missing input file: {path}")
    table = pd.read_csv(path)
    if list(table.columns) != list(RESULT_COLUMNS):
        raise E3Error(f"{path}: columns are {list(table.columns)}")
    if list(table["removed_rule"]) != list(REMOVED):
        raise E3Error(f"{path}: rows are {list(table['removed_rule'])}, expected {list(REMOVED)}")
    return table


def _keys(rows: Sequence[ScoreRow]) -> set[tuple[str, int]]:
    """Set of (video, object) pairs covered by a list of score rows."""
    return {(r.video, r.object) for r in rows}


def build_table(results_dir: Path, splits: Sequence[str] = TEST_SETS) -> list[dict[str, object]]:
    """Combine the scores of every ablation run with the detector counts, per test set.

    Args:
        results_dir: Folder with the scored runs, their event files and the detector CSVs.
        splits: Test sets to include.

    Returns:
        One row per (test set, removed rule) with J, F, J&F, the J&F deltas versus the full
        system and versus SAM 2 alone, the number of re-prompts, and the detector counts.

    Raises:
        E3Error: If an input is missing, or a run scores other videos or objects than
        SAM 2 alone.
    """
    out: list[dict[str, object]] = []
    for split in splits:
        # SAM 2 alone is the reference for the deltas and for the object set.
        alone = load_scores(results_dir / f"{ALONE}_{split}.csv")
        detector = load_detector(detector_path(results_dir, split))
        alone_jf = means(alone)["J&F"]
        # Score every ablation run and count its re-prompts from the events file.
        runs = {}
        for removed in REMOVED:
            name = f"{method_for(removed)}_{split}"
            scores = load_scores(results_dir / f"{name}.csv")
            if _keys(scores) != _keys(alone):
                raise E3Error(f"{name}: scores different videos or objects than SAM 2 alone")
            events = load_events(results_dir / f"events_{name}.csv")
            runs[removed] = (means(scores), int((events["action"] == "reprompted").sum()))
        # The full system (nothing removed) is the reference for delta_vs_full.
        full_jf = runs[NONE][0]["J&F"]
        for removed in REMOVED:
            score, reprompts = runs[removed]
            # Attach the detector counts of this removed rule.
            det = detector[detector["removed_rule"] == removed].iloc[0]
            out.append(
                {
                    "removed_rule": removed,
                    "method": method_for(removed),
                    "test_set": split,
                    "objects": len(alone),
                    **score,
                    "delta_vs_full": score["J&F"] - full_jf,
                    "delta_vs_alone": score["J&F"] - alone_jf,
                    "reprompts": reprompts,
                    **{c: det[c].item() for c in TABLE_COLUMNS[10:]},
                }
            )
    return out


def write_table(path: Path, table: Sequence[dict[str, object]]) -> None:
    """Write the E3 table as a CSV in the column order TABLE_COLUMNS, atomically.

    Args:
        path: Output CSV path; its folder is created if needed.
        table: Rows from build_table. Floats are written with repr so no precision is lost.
    """
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
    """Command-line entry point that builds results/e3.csv.

    Reads the scored runs, their event files and the detector CSVs under --results-dir and
    writes the table to --out. Exits with code 0 on success; an inconsistent or missing input
    raises E3Error.

    Args:
        argv: Argument list; None means sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--splits", nargs="+", choices=TEST_SETS, default=list(TEST_SETS))
    args = parser.parse_args(argv)
    write_table(Path(args.out), build_table(Path(args.results_dir), args.splits))


if __name__ == "__main__":
    main()
