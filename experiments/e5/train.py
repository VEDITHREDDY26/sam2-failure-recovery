"""E5 detector side: fit the logistic-regression failure detector on SA-V val (video-level split).

    python -m experiments.e5.train [--models tiny] [--data-dir DIR]
        [--rules configs/detector/rules.yaml] [--learned-dir configs/learned]
        [--results-dir results] [--overwrite]

Same SA-V val export and refusals as the rule tuning. Videos are split 80 / 20 by sorted name with
random.Random(5415); the two parts are asserted disjoint. The regression (class_weight balanced,
fixed seed, standardised features) is fitted on the annotated frames of the train videos only
(label = failure at that frame). Its frozen decision threshold is the best window-level F1 on the
train videos (grid 0.05 .. 0.95, ties -> the higher threshold) among thresholds that flag at most
FROZEN_MAX_FLAG_RATE of the train windows (same budget as the frozen rules); the saved model holds
only this threshold, and the uncapped threshold appears as a second, unfrozen csv row. The held-out
videos are then scored with window-level precision, recall, F1 and PR-AUC (average precision of
each window's maximum probability), next to the frozen rules on the same held-out videos. Writes
<learned-dir>/logreg_<model>.json with its .sha256 (record the hash before any test run),
results/e5_detector.csv and results/e5_weights.csv. An existing model file is never overwritten
without --overwrite.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from experiments.tune_rules.tune import (
    EXPECTED_VIDEOS,
    FROZEN_MAX_FLAG_RATE,
    SELECTION_CAPPED,
    SPLIT,
    TuneInputError,
    Windows,
    build_windows,
    load_model_inputs,
    window_any,
)
from src.common.config import DATA_ROOT_ENV
from src.detection.learned import (
    DEFAULT_DIR,
    FEATURE_NAMES,
    LearnedModel,
    model_path,
    save_model,
    table_features,
)
from src.detection.rules import DEFAULT_RULES_PATH, batch_flags, load_rules_config

SEED = 5415
HELD_OUT_FRACTION = 0.2
THRESHOLD_GRID = tuple(round(0.05 * k, 2) for k in range(1, 20))
DETECTOR_COLUMNS = (
    "model",
    "method",
    "train_videos",
    "heldout_videos",
    "windows",
    "positives",
    "flagged",
    "tp",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "threshold",
    "selection",
    "frozen",
)
SELECTION_PROTOCOL = "f1_protocol"
SELECTION_RULES = "rules_frozen"
WEIGHT_COLUMNS = ("model", "feature", "weight", "mean", "std")
FloatArray = npt.NDArray[np.float64]


def split_videos(names: Sequence[str]) -> tuple[list[str], list[str]]:
    """(train, held-out) video names; deterministic, disjoint, together all videos."""
    ordered = sorted(set(names))
    shuffled = list(ordered)
    random.Random(SEED).shuffle(shuffled)
    n_held = max(1, round(HELD_OUT_FRACTION * len(shuffled)))
    held, train = sorted(shuffled[:n_held]), sorted(shuffled[n_held:])
    assert not set(held) & set(train), "train and held-out videos overlap"
    assert sorted(held + train) == ordered, "split loses videos"
    return train, held


def subset(
    table: pd.DataFrame, labels: pd.DataFrame, videos: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = set(videos)
    return (
        table[table["video"].isin(keep)].reset_index(drop=True),
        labels[labels["video"].isin(keep)].reset_index(drop=True),
    )


def training_rows(
    table: pd.DataFrame, labels: pd.DataFrame
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Features and failure label of every annotated frame after the object's prompt frame."""
    features, has_prev = table_features(table)
    frame = table[["video", "object_id", "frame_index"]].copy()
    frame["row"] = np.arange(len(table))
    frame = frame[has_prev]
    merged = frame.merge(
        labels[["video", "object_id", "frame_index", "failure"]],
        on=["video", "object_id", "frame_index"],
        how="inner",
        validate="one_to_one",
    )
    return features[merged["row"].to_numpy()], merged["failure"].to_numpy(dtype=bool)


def fit(features: FloatArray, failure: npt.NDArray[np.bool_], sam2_model: str) -> LearnedModel:
    """Standardised, class-weighted logistic regression; the threshold is set later."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    clf = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
    clf.fit((features - mean) / std, failure)
    return LearnedModel(
        sam2_model,
        tuple(float(v) for v in mean),
        tuple(float(v) for v in std),
        tuple(float(v) for v in clf.coef_[0]),
        float(clf.intercept_[0]),
        0.5,
    )


def window_max(probability: FloatArray, windows: Windows) -> FloatArray:
    best = np.full(len(windows.truth), -1.0)
    inside = windows.frame_window >= 0
    np.maximum.at(best, windows.frame_window[inside], probability[inside])
    return best


def window_counts(hit: npt.NDArray[np.bool_], truth: npt.NDArray[np.bool_]) -> dict[str, object]:
    flagged, tp, positives = int(hit.sum()), int((hit & truth).sum()), int(truth.sum())
    denominator = flagged + positives
    return {
        "windows": len(truth),
        "positives": positives,
        "flagged": flagged,
        "tp": tp,
        "precision": tp / flagged if flagged else 0.0,
        "recall": tp / positives if positives else 0.0,
        "f1": 2 * tp / denominator if denominator else 0.0,
    }


def choose_threshold(
    best: FloatArray, truth: npt.NDArray[np.bool_], max_flag_rate: float | None = None
) -> float:
    """Best window F1 on the grid (ties -> higher threshold); with a cap, only thresholds
    flagging at most max_flag_rate of the windows are eligible, and none eligible is an error."""
    scored = [
        (float(window_counts(best >= t, truth)["f1"]), t)
        for t in THRESHOLD_GRID
        if max_flag_rate is None or (best >= t).sum() <= max_flag_rate * len(truth)
    ]
    if not scored:
        raise ValueError(f"no threshold flags at most {max_flag_rate:.0%} of the train windows")
    return max(scored)[1]


def pr_auc(best: FloatArray, truth: npt.NDArray[np.bool_]) -> float:
    return float(average_precision_score(truth, best)) if truth.any() else float("nan")


def train_model(
    model: str,
    table: pd.DataFrame,
    labels: pd.DataFrame,
    rules_path: Path,
) -> tuple[LearnedModel, list[dict[str, object]]]:
    train, held = split_videos(sorted(set(table["video"])))
    t_table, t_labels = subset(table, labels, train)
    h_table, h_labels = subset(table, labels, held)
    features, failure = training_rows(t_table, t_labels)
    fitted = fit(features, failure, model)
    t_windows = build_windows(t_table, t_labels)
    train_best = window_max(fitted.table_probabilities(t_table), t_windows)
    uncapped = choose_threshold(train_best, t_windows.truth)
    threshold = choose_threshold(train_best, t_windows.truth, FROZEN_MAX_FLAG_RATE)
    final = LearnedModel(
        fitted.sam2_model, fitted.mean, fitted.std, fitted.coef, fitted.intercept, threshold
    )
    h_windows = build_windows(h_table, h_labels)
    best = window_max(final.table_probabilities(h_table), h_windows)
    rules = load_rules_config(model, rules_path)
    rule_hit = window_any(batch_flags(h_table, rules)["flag"].to_numpy(dtype=bool), h_windows)
    common = {"model": model, "train_videos": len(train), "heldout_videos": len(held)}
    auc = pr_auc(best, h_windows.truth)
    rows = [
        {**common, "method": "logreg", **window_counts(best >= t, h_windows.truth),
         "pr_auc": auc, "threshold": t, "selection": name, "frozen": frozen}
        for t, name, frozen in (
            (uncapped, SELECTION_PROTOCOL, False),
            (threshold, SELECTION_CAPPED, True),
        )
    ] + [
        {**common, "method": "rules", **window_counts(rule_hit, h_windows.truth),
         "pr_auc": float("nan"), "threshold": float("nan"),
         "selection": SELECTION_RULES, "frozen": True},
    ]  # fmt: skip
    return final, rows


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def run(
    models: Sequence[str],
    data_dir: Path,
    rules_path: Path,
    learned_dir: Path,
    results_dir: Path,
    expect_videos: int = EXPECTED_VIDEOS,
    overwrite: bool = False,
) -> dict[str, str]:
    """Fit each model; returns {model: sha256}. Nothing is written unless every model is fitted."""
    if not rules_path.is_file():
        raise TuneInputError(f"detector not frozen yet: {rules_path} does not exist")
    for model in models:
        if model_path(model, learned_dir).exists() and not overwrite:
            raise TuneInputError(f"{model_path(model, learned_dir)} exists; use --overwrite")
    fitted: dict[str, LearnedModel] = {}
    rows: list[dict[str, object]] = []
    for model in models:
        table, labels = load_model_inputs(data_dir, model, expect_videos)
        fitted[model], model_rows = train_model(model, table, labels, rules_path)
        rows += model_rows
    weights = [
        {"model": m, "feature": name, "weight": w, "mean": mu, "std": sd}
        for m, fm in fitted.items()
        for name, w, mu, sd in zip(FEATURE_NAMES, fm.coef, fm.mean, fm.std, strict=True)
    ]
    hashes = {m: save_model(model_path(m, learned_dir), fm) for m, fm in fitted.items()}
    _write(results_dir / "e5_detector.csv", pd.DataFrame(rows, columns=list(DETECTOR_COLUMNS)))
    _write(results_dir / "e5_weights.csv", pd.DataFrame(weights, columns=list(WEIGHT_COLUMNS)))
    return hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--models", nargs="+", choices=("tiny", "large"), default=["tiny"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--learned-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--expect-videos", type=int, default=EXPECTED_VIDEOS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    root = os.environ.get(DATA_ROOT_ENV)
    data_dir = args.data_dir or (Path(root) / "tune_rules" if root else None)
    if data_dir is None:
        parser.error(f"--data-dir or {DATA_ROOT_ENV} is required")
    try:
        hashes = run(
            args.models, data_dir, args.rules, args.learned_dir, args.results_dir,
            args.expect_videos, args.overwrite,
        )  # fmt: skip
    except (TuneInputError, ValueError, OSError) as error:
        sys.stdout.write(f"e5 train: refused: {error}\n")
        return 1
    for model, digest in hashes.items():
        sys.stdout.write(f"{model} ({SPLIT}): sha256 {digest}  record it before any test-set run\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
