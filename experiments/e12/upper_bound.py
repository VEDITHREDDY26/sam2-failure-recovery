"""E12 part 3: what a perfect verifier could gain, and how well each signal separates (analysis).

    python -m experiments.e12.upper_bound [--results-dir results] \
        [--out results/e12_upper_bound.csv]

Per model on the SA-V val accept-all candidates: summed gain of accept-all, a perfect verifier
(accepts only candidates with gain > 0), the frozen choice (accepts nothing) and the best of the
tuning grid. Per signal: ROC AUC of helped (gain > 0) against hurt (gain < 0), unscored and
gain == 0 candidates and non-finite signal values excluded, with higher value meaning helped.
iou_alone is a ground-truth reference, not a usable signal.
Accept-all must match the committed tuning table.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.e12.candidates import MODELS, CandidateError, load_candidates, write_rows

SIGNALS = ("cand_score", "orig_score", "area_ratio", "box_iou", "mask_iou_orig")
GROUND_TRUTH_SIGNAL = "iou_alone"
STRATEGIES = ("accept_all", "perfect_verifier", "frozen", "best_grid")
OUT_COLUMNS = (
    "model",
    "kind",
    "name",
    "ground_truth_reference",
    "gain_sum",
    "auc_higher_helped",
    "auc_best",
    "direction",
    "helped",
    "hurt",
)
CHECK_TOL = 1e-6
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUT = Path("results/e12_upper_bound.csv")


def grid_rows(results_dir: Path, model: str) -> pd.DataFrame:
    """Read the tuning-grid rows of one model from the tuning table.

    Args:
        results_dir: Folder holding results/verify_sav_val.csv.
        model: SAM 2 size.

    Returns:
        The rows of that model whose setting is a grid number (the accept_nothing row is left out).

    Raises:
        CandidateError: If the tuning table is missing.
    """
    path = results_dir / "verify_sav_val.csv"
    if not path.is_file():
        raise CandidateError(f"missing input file: {path}")
    grid = pd.read_csv(path, dtype={"setting": str})
    grid = grid[grid["model"] == model]
    return grid[pd.to_numeric(grid["setting"], errors="coerce").notna()]


def strategy_rows(
    candidates: pd.DataFrame, grid: pd.DataFrame, model: str
) -> list[dict[str, object]]:
    """Build the summed-gain rows of the four verification strategies for one model.

    Args:
        candidates: Labelled candidates of the model.
        grid: Tuning-grid rows of the model (setting 0 is accept all).
        model: SAM 2 size.

    Returns:
        One row per strategy: accept all, perfect verifier (ground truth: accepts only gain > 0),
        frozen (accepts nothing, gain 0) and the best grid setting.

    Raises:
        CandidateError: If accept-all gain differs from the tuning table.
    """
    gain = candidates["gain"].dropna()
    accept_all = float(gain.sum())
    # Setting 0 has every check off, so it equals accepting all candidates.
    setting_zero = grid[grid["setting"] == "0"]
    if (
        len(setting_zero) != 1
        or abs(float(setting_zero["gain_sum"].iloc[0]) - accept_all) > CHECK_TOL
    ):
        raise CandidateError(f"{model}: accept-all gain differs from the tuning table")
    # Summed gain of each strategy; the perfect verifier keeps only candidates that gained.
    values = {
        "accept_all": accept_all,
        "perfect_verifier": float(gain[gain > 0].sum()),
        "frozen": 0.0,
        "best_grid": float(grid["gain_sum"].max()),
    }
    return [
        {
            "model": model,
            "kind": "strategy",
            "name": name,
            "ground_truth_reference": name == "perfect_verifier",
            "gain_sum": values[name],
            "auc_higher_helped": "",
            "auc_best": "",
            "direction": "",
            "helped": int((gain > 0).sum()) if name == "perfect_verifier" else "",
            "hurt": "",
        }
        for name in STRATEGIES
    ]


def signal_row(candidates: pd.DataFrame, signal: str, model: str) -> dict[str, object]:
    """Measure how well one signal separates helped from hurt candidates.

    Args:
        candidates: Labelled candidates of the model.
        signal: Column to use as the score.
        model: SAM 2 size.

    Returns:
        A row with the helped and hurt counts and the ROC AUC (higher value = helped), the
        better of AUC and 1 - AUC, and the direction; AUC cells stay blank when only one
        outcome is present.
    """
    # Only candidates that helped or hurt count; zero-gain and non-finite values are dropped.
    used = candidates[candidates["gain"].notna() & (candidates["gain"] != 0)]
    used = used[np.isfinite(used[signal])]
    helped = used["gain"] > 0
    row: dict[str, object] = {
        "model": model,
        "kind": "signal",
        "name": signal,
        "ground_truth_reference": signal == GROUND_TRUTH_SIGNAL,
        "gain_sum": "",
        "helped": int(helped.sum()),
        "hurt": int((~helped).sum()),
    }
    if helped.all() or not helped.any():
        return row | {"auc_higher_helped": "", "auc_best": "", "direction": ""}
    # AUC below 0.5 means that lower values helped, so the direction is recorded too.
    auc = float(roc_auc_score(helped, used[signal]))
    direction = "higher_helped" if auc >= 0.5 else "lower_helped"
    return row | {"auc_higher_helped": auc, "auc_best": max(auc, 1 - auc), "direction": direction}


def build(results_dir: Path, models: Sequence[str] = MODELS) -> list[dict[str, object]]:
    """Build the strategy and signal rows for every model.

    Args:
        results_dir: Folder holding the candidate files and the tuning table.
        models: SAM 2 sizes to cover.

    Returns:
        The output rows.
    """
    rows: list[dict[str, object]] = []
    # Strategies first, then one row per signal including the ground-truth reference.
    for model in models:
        candidates = load_candidates(results_dir, model)
        rows += strategy_rows(candidates, grid_rows(results_dir, model), model)
        rows += [signal_row(candidates, s, model) for s in (*SIGNALS, GROUND_TRUTH_SIGNAL)]
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point: write the upper bound and signal separation table.

    Reads the candidate files and results/verify_sav_val.csv in --results-dir. Writes --out
    (default results/e12_upper_bound.csv) and prints the row count. A missing input or a
    mismatch with the tuning table raises CandidateError (traceback, non-zero exit).

    Args:
        argv: Command line arguments; None means sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    rows = build(args.results_dir)
    write_rows(args.out, OUT_COLUMNS, rows)
    print(f"{len(rows)} rows -> {args.out} (analysis; some rows use ground truth)")


if __name__ == "__main__":
    main()
