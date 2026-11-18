"""E8 table: CSRT re-initialised from the last good frame vs continuous, Tiny, with the box check.

    python -m experiments.e8.table [--results-dir results] [--out results/e8.csv]
        [--splits davis_val sav_test]

Reads results/sam2_tiny_<split>.csv, results/<method>_<split>.csv, events_<method>_<split>.csv and
<method>_<split>_reprompts.csv (written by experiments.e8.wrong_reprompts). Variants: reinit is
the E2 Tiny run (fix_rules_csrt_tiny), continuous is fix_continuous_rules_csrt_tiny. A wrong
re-prompt is one whose mask at the re-prompted frame has IoU < 0.5 with ground truth; re-prompts
with no annotated frame to judge are counted as unscored, never as right.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.e2.table import load_events, load_scores, means
from experiments.e8.wrong_reprompts import REPROMPT_COLUMNS
from src.eval.rows import ScoreRow

TEST_SETS = ("davis_val", "sav_test")
ALONE = "sam2_tiny"
VARIANTS = {"reinit": "fix_rules_csrt_tiny", "continuous": "fix_continuous_rules_csrt_tiny"}
TABLE_COLUMNS = (
    "variant",
    "method",
    "test_set",
    "objects",
    "J",
    "F",
    "J&F",
    "delta_JF",
    "reprompts",
    "rejected",
    "wrong_reprompts",
    "unscored_reprompts",
)
DEFAULT_OUT = "results/e8.csv"


class E8Error(ValueError):
    """An E8 input is missing or inconsistent."""


def load_reprompts(path: Path) -> pd.DataFrame:
    """Read a re-prompt file written by experiments.e8.wrong_reprompts.

    Raises:
        E8Error: If the file is missing or its columns are not REPROMPT_COLUMNS.
    """
    if not path.is_file():
        raise E8Error(f"missing input file: {path}")
    table = pd.read_csv(path, dtype={"video": str})
    if list(table.columns) != list(REPROMPT_COLUMNS):
        raise E8Error(f"{path}: columns are {list(table.columns)}")
    return table


def _keys(rows: Sequence[ScoreRow]) -> set[tuple[str, int]]:
    """Return the set of (video, object) pairs that a list of score rows covers."""
    return {(r.video, r.object) for r in rows}


def build_rows(results_dir: Path, splits: Sequence[str] = TEST_SETS) -> list[dict[str, object]]:
    """Build the E8 table rows: SAM 2 alone, then each tracker variant per split.

    Args:
        results_dir: Folder holding scores, events and re-prompt files.
        splits: Test sets to cover.

    Returns:
        One row per split for SAM 2 alone and per variant, with mean J, F, J&F, the change
        versus SAM 2 alone and the re-prompt, rejected, wrong and unscored counts.

    Raises:
        E8Error: If a variant scores other objects than SAM 2 alone, or its re-prompt file
            disagrees with its events.
    """
    table: list[dict[str, object]] = []
    # Per split: the SAM 2 alone row is the reference for all changes.
    for split in splits:
        alone = load_scores(results_dir / f"{ALONE}_{split}.csv")
        base = means(alone)
        table.append(
            {"variant": "sam2_alone", "method": ALONE, "test_set": split, "objects": len(alone)}
            | base
            | {"delta_JF": 0.0}
            | {c: "" for c in TABLE_COLUMNS[8:]}
        )
        # Then one row per tracker variant, after checking its inputs agree.
        for variant, method in VARIANTS.items():
            name = f"{method}_{split}"
            rows = load_scores(results_dir / f"{name}.csv")
            if _keys(rows) != _keys(alone):
                raise E8Error(f"{split}/{variant}: scores a different set of objects")
            events = load_events(results_dir / f"events_{name}.csv")
            reprompts = load_reprompts(results_dir / f"{name}_reprompts.csv")
            # The event log and the re-prompt file must count the same re-prompts.
            actions = events["action"]
            if len(reprompts) != int((actions == "reprompted").sum()):
                raise E8Error(f"{split}/{variant}: re-prompt file and events disagree")
            new = means(rows)
            table.append(
                {"variant": variant, "method": method, "test_set": split, "objects": len(rows)}
                | new
                | {
                    "delta_JF": new["J&F"] - base["J&F"],
                    "reprompts": len(reprompts),
                    "rejected": int((actions == "rejected").sum()),
                    "wrong_reprompts": int((reprompts["wrong"] == True).sum()),  # noqa: E712
                    "unscored_reprompts": int(reprompts["wrong"].isna().sum()),
                }
            )
    return table


def write_table(path: Path, table: Sequence[dict[str, object]]) -> None:
    """Write the table to a CSV with the TABLE_COLUMNS header.

    The file is written under a temporary name and moved into place, so a failure never leaves
    a half-written table. Floats are written with repr.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first, then replace the target in one step.
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
    """Command line entry point: write the E8 table.

    Reads the score, events and re-prompt files in --results-dir for the chosen --splits and
    writes --out (default results/e8.csv). Inconsistent inputs raise E8Error (traceback,
    non-zero exit).

    Args:
        argv: Command line arguments; None means sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--splits", nargs="+", choices=TEST_SETS, default=list(TEST_SETS))
    args = parser.parse_args(argv)
    write_table(Path(args.out), build_rows(Path(args.results_dir), args.splits))


if __name__ == "__main__":
    main()
