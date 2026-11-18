"""E5 table: rules vs the learned logistic regression, Tiny.

    python -m experiments.e5.table [--results-dir results] [--out results/e5.csv]
        [--splits davis_val sav_test]

Reads results/e5_detector.csv and results/e5_weights.csv (experiments.e5.train), the SAM 2 alone
scores, the E2 Tiny run (rules) and the scored learned run results/fix_learned_csrt_tiny_<split>.csv
with its events file. One row per test set and method. J&F columns are test-set means over
objects; delta_vs_rules is the J&F difference to the E2 Tiny run. The detector columns (windows,
precision, recall, F1, PR-AUC) are on the held-out SA-V val videos, the same for both test sets;
w_<feature> are the standardised logistic-regression weights (learned row only).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.e2.table import E2Error, load_events, load_scores, means
from experiments.e5.train import DETECTOR_COLUMNS, WEIGHT_COLUMNS
from src.detection.learned import FEATURE_NAMES
from src.eval.rows import ScoreRow

TEST_SETS = ("davis_val", "sav_test")
MODEL = "tiny"
METHODS = {"rules": f"fix_rules_csrt_{MODEL}", "learned": f"fix_learned_csrt_{MODEL}"}
DETECTOR_ROW = {"rules": "rules", "learned": "logreg"}
ALONE = f"sam2_{MODEL}"
DETECTOR_FIELDS = ("windows", "positives", "flagged", "tp", "precision", "recall", "f1", "pr_auc")
TABLE_COLUMNS = (
    "detector",
    "method",
    "test_set",
    "objects",
    "J",
    "F",
    "J&F",
    "delta_vs_alone",
    "delta_vs_rules",
    "reprompts",
    *DETECTOR_FIELDS,
    "threshold",
    *(f"w_{name}" for name in FEATURE_NAMES),
)
DEFAULT_OUT = "results/e5.csv"


class E5Error(ValueError):
    """An E5 input is missing or inconsistent."""


def _keys(rows: Sequence[ScoreRow]) -> set[tuple[str, int]]:
    """Set of (video, object) pairs covered by a list of score rows."""
    return {(r.video, r.object) for r in rows}


def load_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read a CSV file and check that its columns are exactly the expected ones.

    Args:
        path: CSV file to read.
        columns: Expected column names, in order.

    Returns:
        The table.

    Raises:
        E5Error: If the file is missing or its columns differ.
    """
    if not path.is_file():
        raise E5Error(f"missing input file: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != list(columns):
        raise E5Error(f"{path}: columns are {list(frame.columns)}")
    return frame


def build_table(results_dir: Path, splits: Sequence[str] = TEST_SETS) -> list[dict[str, object]]:
    """Combine the rule and learned runs with the held-out detector metrics, per test set.

    Args:
        results_dir: Folder with the detector CSVs, the scored runs and their event files.
        splits: Test sets to include.

    Returns:
        One row per (test set, detector) with J, F, J&F, the J&F deltas versus SAM 2 alone and
        versus the rules run, re-prompt count, detector metrics, threshold and, for the learned
        row, the feature weights.

    Raises:
        E5Error: If an input is missing or inconsistent, or a run scores other videos or
            objects than SAM 2 alone.
    """
    # Held-out detector metrics and weights come from the training step.
    detector = load_csv(results_dir / "e5_detector.csv", DETECTOR_COLUMNS)
    weights = load_csv(results_dir / "e5_weights.csv", WEIGHT_COLUMNS)
    # Keep only the frozen Tiny rows; they are the same for both test sets.
    detector = detector[(detector["model"] == MODEL) & detector["frozen"]].set_index("method")
    weights = weights[weights["model"] == MODEL].set_index("feature")["weight"]
    if sorted(detector.index) != sorted(DETECTOR_ROW.values()) or list(weights.index) != list(
        FEATURE_NAMES
    ):
        raise E5Error(f"e5_detector / e5_weights lack the {MODEL} rows or features")
    # Per test set: SAM 2 alone is the reference, then score both recovery runs.
    out: list[dict[str, object]] = []
    for split in splits:
        alone = load_scores(results_dir / f"{ALONE}_{split}.csv")
        alone_jf = means(alone)["J&F"]
        # Score each run and count its re-prompts from its events file.
        runs = {}
        for name, method in METHODS.items():
            scores = load_scores(results_dir / f"{method}_{split}.csv")
            if _keys(scores) != _keys(alone):
                raise E5Error(f"{method}_{split}: scores different videos or objects than SAM 2")
            events = load_events(results_dir / f"events_{method}_{split}.csv")
            runs[name] = (means(scores), int((events["action"] == "reprompted").sum()))
        # The rules run is the reference for delta_vs_rules.
        rules_jf = runs["rules"][0]["J&F"]
        for name, method in METHODS.items():
            score, reprompts = runs[name]
            det = detector.loc[DETECTOR_ROW[name]]
            row: dict[str, object] = {
                "detector": name,
                "method": method,
                "test_set": split,
                "objects": len(alone),
                **score,
                "delta_vs_alone": score["J&F"] - alone_jf,
                "delta_vs_rules": score["J&F"] - rules_jf,
                "reprompts": reprompts,
                **{c: det[c].item() for c in DETECTOR_FIELDS},
                "threshold": det["threshold"].item(),
            }
            # Weights apply to the learned detector only; the rules row leaves them empty.
            for feature in FEATURE_NAMES:
                row[f"w_{feature}"] = float(weights[feature]) if name == "learned" else ""
            out.append(row)
    return out


def write_table(path: Path, table: Sequence[dict[str, object]]) -> None:
    """Write the E5 table as a CSV in the column order TABLE_COLUMNS, atomically.

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
    """Command-line entry point that builds results/e5.csv.

    Reads the detector and weight CSVs and the scored runs under --results-dir and writes the
    table to --out. Exits with a message (SystemExit) if an input is missing or inconsistent.

    Args:
        argv: Argument list; None means sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--splits", nargs="+", choices=TEST_SETS, default=list(TEST_SETS))
    args = parser.parse_args(argv)
    try:
        write_table(Path(args.out), build_table(Path(args.results_dir), args.splits))
    except (E5Error, E2Error) as error:
        raise SystemExit(f"e5 table: {error}") from error


if __name__ == "__main__":
    main()
