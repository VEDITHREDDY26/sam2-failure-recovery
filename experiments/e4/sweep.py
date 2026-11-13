"""E4: precision / recall of the rule detector as the score and area thresholds are swept.

    python -m experiments.e4.sweep [--models tiny large] [--data-dir DIR]
        [--rules configs/detector/rules.yaml] [--out results/e4.csv]

Detector level only (no SAM 2 reruns), on the SA-V val export used for the tuning (same loading
and refusals as experiments.tune_rules.tune). The jump, disappearance-fraction and border
thresholds stay at their frozen values from the rules config (refused if it is missing). Three
sweeps per model: score_threshold alone, area_ratio alone, and the 2-D grid of both. Precision,
recall and F1 are over annotation windows (build_windows / window_any of the tuning). The frozen
operating point is marked and is always on the grids.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tune_rules.tune import (
    EXPECTED_VIDEOS,
    SPLIT,
    BoolArray,
    TuneInputError,
    Windows,
    build_windows,
    load_model_inputs,
    window_any,
)
from src.common.config import DATA_ROOT_ENV
from src.detection.rules import (
    DEFAULT_RULES_PATH,
    MODELS,
    RuleThresholds,
    batch_evidence,
    load_rules_config,
)

SCORE_GRID: tuple[float, ...] = tuple(v / 2 for v in range(-24, 9))
AREA_RATIO_GRID: tuple[float, ...] = (
    1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.5, 10.0, 15.0, 20.0,
)  # fmt: skip
SWEEPS = ("score_threshold", "area_ratio", "both")
COLUMNS = (
    "model",
    "swept",
    "score_threshold",
    "area_ratio",
    "windows",
    "positives",
    "flagged",
    "tp",
    "precision",
    "recall",
    "f1",
    "frozen",
)


def with_frozen(grid: Sequence[float], value: float) -> list[float]:
    return sorted({*grid, value})


def _row(
    model: str,
    swept: str,
    score: float,
    area: float,
    hit: BoolArray,
    truth: BoolArray,
    frozen: RuleThresholds,
) -> dict[str, object]:
    flagged, tp, positives = int(hit.sum()), int((hit & truth).sum()), int(truth.sum())
    denominator = flagged + positives
    return {
        "model": model,
        "swept": swept,
        "score_threshold": score,
        "area_ratio": area,
        "windows": len(truth),
        "positives": positives,
        "flagged": flagged,
        "tp": tp,
        "precision": tp / flagged if flagged else 0.0,
        "recall": tp / positives if positives else 0.0,
        "f1": 2 * tp / denominator if denominator else 0.0,
        "frozen": score == frozen.score_threshold and area == frozen.area_ratio,
    }


def sweep_model(
    model: str, table: pd.DataFrame, labels: pd.DataFrame, frozen: RuleThresholds
) -> list[dict[str, object]]:
    windows: Windows = build_windows(table, labels)
    ev = batch_evidence(table)
    score, ratio, jump = (ev[c].to_numpy(dtype=float) for c in ("score", "area_ratio", "jump"))
    fraction = ev["prev_area_fraction"].to_numpy(dtype=float)
    margin = ev["prev_margin"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        fixed = (jump > frozen.jump_ratio) | (
            (fraction >= frozen.min_area_fraction) & (margin >= frozen.border_margin_px)
        )
        score_grid = with_frozen(SCORE_GRID, frozen.score_threshold)
        area_grid = with_frozen(AREA_RATIO_GRID, frozen.area_ratio)
        low = {v: window_any(score < v, windows) for v in score_grid}
        area = {v: window_any(ratio > v, windows) for v in area_grid}
    rest = window_any(fixed, windows)
    rows = []
    for v in score_grid:
        rows.append(
            _row(model, "score_threshold", v, frozen.area_ratio,
                 low[v] | area[frozen.area_ratio] | rest, windows.truth, frozen)
        )  # fmt: skip
    for v in area_grid:
        rows.append(
            _row(model, "area_ratio", frozen.score_threshold, v,
                 low[frozen.score_threshold] | area[v] | rest, windows.truth, frozen)
        )  # fmt: skip
    for s in score_grid:
        for a in area_grid:
            rows.append(_row(model, "both", s, a, low[s] | area[a] | rest, windows.truth, frozen))
    return rows


def run(
    models: Sequence[str],
    data_dir: Path,
    rules_path: Path,
    out: Path,
    expect_videos: int = EXPECTED_VIDEOS,
) -> pd.DataFrame:
    if not rules_path.is_file():
        raise TuneInputError(f"detector not frozen yet: {rules_path} does not exist")
    rows: list[dict[str, object]] = []
    for model in models:
        table, labels = load_model_inputs(data_dir, model, expect_videos)
        rows += sweep_model(model, table, labels, load_rules_config(model, rules_path))
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, out)
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--out", type=Path, default=Path("results/e4.csv"))
    parser.add_argument("--expect-videos", type=int, default=EXPECTED_VIDEOS)
    args = parser.parse_args(argv)
    root = os.environ.get(DATA_ROOT_ENV)
    data_dir = args.data_dir or (Path(root) / "tune_rules" if root else None)
    if data_dir is None:
        parser.error(f"--data-dir or {DATA_ROOT_ENV} is required")
    try:
        frame = run(args.models, data_dir, args.rules, args.out, args.expect_videos)
    except (TuneInputError, ValueError, OSError) as error:
        sys.stdout.write(f"e4 sweep: refused: {error}\n")
        return 1
    for _, row in frame[frame["frozen"] & (frame["swept"] == "both")].iterrows():
        sys.stdout.write(
            f"{row['model']} ({SPLIT}) frozen point: precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1={row['f1']:.3f}\n"
        )
    sys.stdout.write(f"wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
