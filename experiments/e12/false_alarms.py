"""E12 part 2: re-prompts on real failures versus false alarms (SA-V val, analysis only).

    python -m experiments.e12.false_alarms [--results-dir results] \
        [--out results/e12_false_alarms.csv]

Uses the accept-all candidates of the rules + CSRT fix on SA-V val (the tuning data). A candidate
is a real failure when SAM 2 alone was already wrong at the matched labelled frame (IoU < 0.5),
a false alarm otherwise, and unscored when no gain could be computed. The split uses ground truth,
so it describes the candidates and is not available to the method.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.e12.candidates import (
    MODELS,
    REAL_FAILURE_BELOW,
    load_candidates,
    write_rows,
)

CLASSES = ("real_failure", "false_alarm", "unscored")
OUT_COLUMNS = (
    "model",
    "split",
    "class",
    "candidates",
    "share",
    "gain_sum",
    "gain_mean",
    "helped",
    "hurt",
    "unchanged",
    "wrong_after",
)
SPLIT = "sav_val"
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUT = Path("results/e12_false_alarms.csv")


def classify(candidates: pd.DataFrame) -> pd.Series:
    """Label each candidate as a real failure, a false alarm or unscored.

    Args:
        candidates: Candidate table with iou_alone and gain columns.

    Returns:
        A series of "real_failure" (SAM 2 alone IoU below 0.5), "false_alarm", or "unscored"
        (gain or iou_alone missing), aligned with the candidate rows.
    """
    # Start with false alarm; real failures overwrite it; unscored overwrites both.
    kind = pd.Series("false_alarm", index=candidates.index)
    kind[candidates["iou_alone"] < REAL_FAILURE_BELOW] = "real_failure"
    kind[candidates["gain"].isna() | candidates["iou_alone"].isna()] = "unscored"
    return kind


def summarise(part: pd.DataFrame, total: int, model: str, name: str) -> dict[str, object]:
    """Summarise the candidates of one class for one model.

    Args:
        part: The candidates of the class.
        total: Number of candidates of the model (for the share).
        model: SAM 2 size.
        name: Class name.

    Returns:
        One output row: counts, summed and mean gain, helped / hurt / unchanged counts, and how
        many candidates were still wrong (IoU below 0.5) after the re-prompt.
    """
    # Candidates without a gain are left out of the gain statistics.
    gain = part["gain"].dropna()
    return {
        "model": model,
        "split": SPLIT,
        "class": name,
        "candidates": len(part),
        "share": len(part) / total if total else 0.0,
        "gain_sum": float(gain.sum()),
        "gain_mean": float(gain.mean()) if len(gain) else "",
        "helped": int((gain > 0).sum()),
        "hurt": int((gain < 0).sum()),
        "unchanged": int((gain == 0).sum()),
        "wrong_after": int((part["iou_method"] < REAL_FAILURE_BELOW).sum()),
    }


def build(results_dir: Path, models: Sequence[str] = MODELS) -> list[dict[str, object]]:
    """Build the class summary rows for each model.

    Args:
        results_dir: Folder holding the labelled candidate files.
        models: SAM 2 sizes to cover.

    Returns:
        One row per model and class.
    """
    rows = []
    # Each model has its own candidate file; every class gets a row, even an empty one.
    for model in models:
        candidates = load_candidates(results_dir, model)
        kind = classify(candidates)
        for name in CLASSES:
            rows.append(summarise(candidates[kind == name], len(candidates), model, name))
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point: write the real failure versus false alarm table.

    Reads results/verify_candidates_<model>_sav_val.csv in --results-dir. Writes --out
    (default results/e12_false_alarms.csv) and prints the row count. A missing or malformed
    input raises CandidateError (traceback, non-zero exit).

    Args:
        argv: Command line arguments; None means sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    # Build all rows before writing so a bad input leaves no partial file.
    rows = build(args.results_dir)
    write_rows(args.out, OUT_COLUMNS, rows)
    print(f"{len(rows)} rows -> {args.out} (analysis; split uses ground truth)")


if __name__ == "__main__":
    main()
