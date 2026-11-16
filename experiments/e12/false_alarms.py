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
    kind = pd.Series("false_alarm", index=candidates.index)
    kind[candidates["iou_alone"] < REAL_FAILURE_BELOW] = "real_failure"
    kind[candidates["gain"].isna() | candidates["iou_alone"].isna()] = "unscored"
    return kind


def summarise(part: pd.DataFrame, total: int, model: str, name: str) -> dict[str, object]:
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
    rows = []
    for model in models:
        candidates = load_candidates(results_dir, model)
        kind = classify(candidates)
        for name in CLASSES:
            rows.append(summarise(candidates[kind == name], len(candidates), model, name))
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    rows = build(args.results_dir)
    write_rows(args.out, OUT_COLUMNS, rows)
    print(f"{len(rows)} rows -> {args.out} (analysis; split uses ground truth)")


if __name__ == "__main__":
    main()
