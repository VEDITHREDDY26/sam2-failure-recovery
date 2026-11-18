"""E7 table: oracle detector / oracle box vs the real pieces, Tiny, DAVIS val and SA-V test.

    python -m experiments.e7.table [--results-dir results] [--out results/e7.csv]
        [--splits davis_val sav_test]

Four arms (rule detector or oracle flags) x (CSRT box or oracle box); the rules + CSRT arm is the
E2 Tiny run, reused. Every arm that uses ground truth has "oracle" in its name and oracle=True,
so the upper bounds are never read as results of the real system. SAM 2 alone is the reference.
Run experiments.e7.check_arms first; the table refuses to build if the check file is missing.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

from experiments.e2.table import load_events, load_scores, means
from experiments.e7.arms import ALONE, ARMS, TEST_SETS

TABLE_COLUMNS = (
    "arm",
    "method",
    "detector",
    "box_source",
    "oracle",
    "test_set",
    "objects",
    "J",
    "F",
    "J&F",
    "delta_JF",
    "reprompts",
)
CHECK_FILE = "e7_arms_check.csv"


class E7Error(ValueError):
    """An E7 input is missing."""


def build_rows(results_dir: Path, splits: Sequence[str] = TEST_SETS) -> list[dict[str, object]]:
    """Build the E7 rows: SAM 2 alone and each of the four arms, per test set.

    Args:
        results_dir: Folder with the scored runs, their event files and the arms check file.
        splits: Test sets to include.

    Returns:
        One row per (test set, arm) with J, F, J&F, the J&F delta versus SAM 2 alone, the
        number of re-prompts and the oracle flag.

    Raises:
        E7Error: If the arms check file is missing, or an arm's oracle label does not match
            its pieces.
    """
    # The arms must have been checked as comparable before the table is built.
    if not (results_dir / CHECK_FILE).is_file():
        raise E7Error(f"{results_dir / CHECK_FILE} missing: run experiments.e7.check_arms first")
    table: list[dict[str, object]] = []
    for split in splits:
        # SAM 2 alone is the reference row; it uses no detector or box source.
        alone = load_scores(results_dir / f"{ALONE}_{split}.csv")
        base = means(alone)
        table.append(
            {
                "arm": ALONE,
                "method": ALONE,
                "detector": "none",
                "box_source": "none",
                "oracle": False,
                "test_set": split,
                "objects": len(alone),
                "delta_JF": 0.0,
                "reprompts": "",
            }
            | base
        )
        # Every arm that uses ground truth must be labelled oracle in name and flag.
        for arm in ARMS:
            if ("oracle" in arm.name) != arm.oracle or ("oracle" in arm.method) != arm.oracle:
                raise E7Error(f"{arm.name}: oracle label does not match its pieces")
            name = f"{arm.method}_{split}"
            rows = load_scores(results_dir / f"{name}.csv")
            events = load_events(results_dir / f"events_{name}.csv")
            new = means(rows)
            table.append(
                {
                    "arm": arm.name,
                    "method": arm.method,
                    "detector": arm.detector,
                    "box_source": arm.box_source,
                    "oracle": arm.oracle,
                    "test_set": split,
                    "objects": len(rows),
                    "delta_JF": new["J&F"] - base["J&F"],
                    "reprompts": int((events["action"] == "reprompted").sum()),
                }
                | new
            )
    return table


def write_table(path: Path, table: Sequence[dict[str, object]]) -> None:
    """Write the E7 table as a CSV in the column order TABLE_COLUMNS, atomically.

    Args:
        path: Output CSV path; its folder is created if needed.
        table: Rows from build_rows. Floats are written with repr so no precision is lost.
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
    """Command-line entry point that builds results/e7.csv.

    Reads the scored runs, their event files and the arms check file under --results-dir and
    writes the table to --out. Raises E7Error if the check file is missing.

    Args:
        argv: Argument list; None means sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/e7.csv")
    parser.add_argument("--splits", nargs="+", choices=TEST_SETS, default=list(TEST_SETS))
    args = parser.parse_args(argv)
    write_table(Path(args.out), build_rows(Path(args.results_dir), args.splits))


if __name__ == "__main__":
    main()
