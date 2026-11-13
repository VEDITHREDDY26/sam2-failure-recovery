"""Tune the rule-detector thresholds on SA-V val only (protocol fixed before any label was read).

    python -m experiments.tune_rules.tune --models tiny large [--data-dir DIR]
        [--config-out configs/detector/rules.yaml] [--results-out results/rules_sav_val.csv]

Inputs in the data dir (default $SAM2FR_DATA_ROOT/tune_rules), per model:
    labels_<model>_sav_val.csv, boxes_<model>_sav_val.csv,
    signals/sam2_<model>/sav_val/signals/<video>.csv

Unit = annotation window: for each object and each annotated frame t_a after its prompt frame, the
window is the frames (previous annotated frame, t_a]; it is positive when any rule fires in it and
its truth is the failure label at t_a. Precision / recall / F1 are pooled over all windows.
Search = exhaustive over the grids below, all four rules on; ties -> fewer flagged windows, then
the earlier grid point (order: score, area, jump, min fraction, border; each grid ascending).
Two points are computed per model: the protocol point (no cap) and the capped point, the best F1
among grid points that flag at most FROZEN_MAX_FLAG_RATE of all windows (2026-11-13 decision, a
module constant with no command-line option). Only the capped thresholds go to the config (the
frozen detector); the results CSV has both rows (`selection`, `frozen`).
Nothing is written unless every check passes.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from src.common.config import DATA_ROOT_ENV
from src.common.signals import read_signals
from src.detection.rules import (
    ALL_RULES,
    AREA_CHANGE,
    CENTRE_JUMP,
    DISAPPEARANCE,
    LOW_SCORE,
    MODELS,
    RuleThresholds,
    batch_evidence,
)
from src.detection.signals import count_overlapped, join_signals_boxes

FROZEN_MAX_FLAG_RATE = 0.10
SELECTION_PROTOCOL = "f1_protocol"
SELECTION_CAPPED = f"f1_flag_cap_{FROZEN_MAX_FLAG_RATE:.2f}"
SPLIT = "sav_val"
EXPECTED_VIDEOS = 155
FORBIDDEN = ("davis", "sav_test")

SCORE_GRID: tuple[float, ...] = tuple(float(v) for v in range(-8, 3))
AREA_RATIO_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0)
JUMP_RATIO_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
MIN_AREA_FRACTION_GRID: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1)
BORDER_MARGIN_GRID: tuple[float, ...] = (0.0, 4.0, 8.0, 16.0, 32.0)

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


class TuneInputError(Exception):
    """The input is not the SA-V val export, or is inconsistent."""


@dataclasses.dataclass(frozen=True)
class Windows:
    """Per-frame window ids (-1 = in no window) and the truth of each window."""

    frame_window: IntArray
    truth: BoolArray


@dataclasses.dataclass(frozen=True)
class Choice:
    thresholds: RuleThresholds
    windows: int
    positives: int
    flagged: int
    tp: int
    per_rule_fires: dict[str, int]
    per_rule_tp: dict[str, int]

    @property
    def fp(self) -> int:
        return self.flagged - self.tp

    @property
    def fn(self) -> int:
        return self.positives - self.tp

    @property
    def precision(self) -> float:
        return self.tp / self.flagged if self.flagged else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.positives if self.positives else 0.0

    @property
    def f1(self) -> float:
        return _f1(self.tp, self.flagged, self.positives)


def _f1_array(tp: IntArray, flagged: IntArray, positives: int) -> npt.NDArray[np.float64]:
    denominator = flagged + positives
    return np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 0.0)


def _f1(tp: int, flagged: int, positives: int) -> float:
    denominator = flagged + positives
    return 2 * tp / denominator if denominator else 0.0


def check_paths(paths: Sequence[Path], require_split: bool = True) -> None:
    for path in paths:
        text = str(path).lower()
        bad = [word for word in FORBIDDEN if word in text]
        if bad:
            raise TuneInputError(f"refusing {path}: contains {bad[0]} (only SA-V val is allowed)")
        if require_split and SPLIT not in text:
            raise TuneInputError(f"refusing {path}: path does not name {SPLIT}")


def input_paths(data_dir: Path, model: str) -> tuple[Path, Path, Path]:
    labels = data_dir / f"labels_{model}_{SPLIT}.csv"
    boxes = data_dir / f"boxes_{model}_{SPLIT}.csv"
    signals = data_dir / "signals" / f"sam2_{model}" / SPLIT / "signals"
    return labels, boxes, signals


def build_windows(table: pd.DataFrame, labels: pd.DataFrame) -> Windows:
    """Window per frame of the joined table; the prompt frame is the object's first table row."""
    first = table.groupby(["video", "object_id"], as_index=False)["frame_index"].min()
    prompt = {
        (str(v), int(o)): int(f)
        for v, o, f in zip(first["video"], first["object_id"], first["frame_index"], strict=False)
    }
    ann = labels.sort_values(["video", "object_id", "frame_index"], kind="stable")
    keys = list(zip(ann["video"].astype(str), ann["object_id"].astype(int), strict=False))
    if any(k not in prompt for k in keys):
        raise TuneInputError("labels for an object that has no boxes rows")
    ann = ann[
        [f > prompt[k] for k, f in zip(keys, ann["frame_index"].astype(int), strict=False)]
    ].reset_index(drop=True)
    ann_keys = list(zip(ann["video"].astype(str), ann["object_id"].astype(int), strict=False))
    ann_frames_all = ann["frame_index"].to_numpy(dtype=np.int64)
    offset: dict[tuple[str, int], int] = {}
    count: dict[tuple[str, int], int] = {}
    for i, k in enumerate(ann_keys):
        offset.setdefault(k, i)
        count[k] = count.get(k, 0) + 1
    frame_window = np.full(len(table), -1, dtype=np.int64)
    frame_index = table["frame_index"].to_numpy(dtype=np.int64)
    table_keys = list(zip(table["video"].astype(str), table["object_id"].astype(int), strict=False))
    if table.empty:
        raise TuneInputError("no frames in the joined table")
    starts = [0] + [i for i in range(1, len(table_keys)) if table_keys[i] != table_keys[i - 1]]
    for begin, end in zip(starts, starts[1:] + [len(table_keys)], strict=False):
        key = table_keys[begin]
        if key not in offset:
            continue
        base = offset[key]
        targets = ann_frames_all[base : base + count[key]]
        frames = frame_index[begin:end]
        idx = np.searchsorted(targets, frames, side="left")
        valid = (idx < len(targets)) & (frames > prompt[key])
        frame_window[begin:end][valid] = base + idx[valid]
    return Windows(frame_window, ann["failure"].to_numpy(dtype=bool))


def window_any(frame_flags: BoolArray, windows: Windows) -> BoolArray:
    inside = windows.frame_window[frame_flags & (windows.frame_window >= 0)]
    return np.bincount(inside, minlength=len(windows.truth)) > 0


def rule_vectors(
    table: pd.DataFrame, windows: Windows
) -> tuple[list[BoolArray], list[BoolArray], list[BoolArray], list[BoolArray]]:
    """Window-level bool vectors per grid value (disappearance: fraction x margin)."""
    ev = batch_evidence(table)
    score, ratio, jump = (ev[c].to_numpy(dtype=float) for c in ("score", "area_ratio", "jump"))
    frac, margin = ev["prev_area_fraction"].to_numpy(dtype=float), ev["prev_margin"].to_numpy(
        dtype=float
    )
    with np.errstate(invalid="ignore"):
        low = [window_any(score < v, windows) for v in SCORE_GRID]
        area = [window_any(ratio > v, windows) for v in AREA_RATIO_GRID]
        move = [window_any(jump > v, windows) for v in JUMP_RATIO_GRID]
        gone = [
            window_any((frac >= f) & (margin >= b), windows)
            for f in MIN_AREA_FRACTION_GRID
            for b in BORDER_MARGIN_GRID
        ]
    return low, area, move, gone


def search(
    low: list[BoolArray],
    area: list[BoolArray],
    move: list[BoolArray],
    gone: list[BoolArray],
    truth: BoolArray,
    max_flag_rate: float | None = None,
) -> tuple[tuple[int, int, int, int, int], int, int]:
    """Best grid point (indices in grid order), its flagged and tp counts.

    With max_flag_rate, only points flagging at most that share of all windows are eligible.
    """
    positives = int(truth.sum())
    budget = None if max_flag_rate is None else int(np.floor(max_flag_rate * len(truth) + 1e-9))
    grid_shape = (
        len(low),
        len(area),
        len(move),
        len(MIN_AREA_FRACTION_GRID),
        len(BORDER_MARGIN_GRID),
    )
    gone_stack = np.stack(gone)
    move_gone = np.stack([m | gone_stack for m in move])  # (move, gone, W)
    best: tuple[float, int, int] | None = None
    best_point: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    best_counts = (0, 0)
    n_borders = len(BORDER_MARGIN_GRID)
    for i_low, v_low in enumerate(low):
        for i_area, v_area in enumerate(area):
            base = v_low | v_area
            flags = base | move_gone  # (move, gone, W)
            flagged = flags.sum(axis=2, dtype=np.int64)
            tp = (flags & truth).sum(axis=2, dtype=np.int64)
            f1 = _f1_array(tp, flagged, positives)
            if budget is not None:
                f1 = np.where(flagged <= budget, f1, -1.0)
            top = float(f1.max())
            if top < 0:
                continue
            cand = np.argwhere(f1 == top)
            if best is not None and top < best[0]:
                continue
            for i_move, i_gone in cand:
                key_flagged = int(flagged[i_move, i_gone])
                f_i, b_i = divmod(int(i_gone), n_borders)
                point = (i_low, i_area, int(i_move), f_i, b_i)
                rank = int(np.ravel_multi_index(point, grid_shape))
                if best is None or (top, -key_flagged, -rank) > (best[0], -best[1], -best[2]):
                    best = (top, key_flagged, rank)
                    best_point = point
                    best_counts = (key_flagged, int(tp[i_move, i_gone]))
    if best is None:
        raise TuneInputError(f"no grid point flags at most {max_flag_rate} of the windows")
    return best_point, best_counts[0], best_counts[1]


def thresholds_at(point: tuple[int, int, int, int, int]) -> RuleThresholds:
    i_s, i_a, i_j, i_f, i_b = point
    return RuleThresholds(
        SCORE_GRID[i_s],
        AREA_RATIO_GRID[i_a],
        JUMP_RATIO_GRID[i_j],
        MIN_AREA_FRACTION_GRID[i_f],
        BORDER_MARGIN_GRID[i_b],
    )


def rule_window_flags(
    table: pd.DataFrame, windows: Windows, cfg: RuleThresholds
) -> dict[str, BoolArray]:
    ev = batch_evidence(table)
    with np.errstate(invalid="ignore"):
        frame = {
            LOW_SCORE: ev["score"].to_numpy(dtype=float) < cfg.score_threshold,
            AREA_CHANGE: ev["area_ratio"].to_numpy(dtype=float) > cfg.area_ratio,
            CENTRE_JUMP: ev["jump"].to_numpy(dtype=float) > cfg.jump_ratio,
            DISAPPEARANCE: (ev["prev_area_fraction"].to_numpy(dtype=float) >= cfg.min_area_fraction)
            & (ev["prev_margin"].to_numpy(dtype=float) >= cfg.border_margin_px),
        }
    return {name: window_any(frame[name], windows) for name in ALL_RULES}


def _choice_at(
    table: pd.DataFrame,
    windows: Windows,
    point: tuple[int, int, int, int, int],
    flagged: int,
    tp: int,
) -> Choice:
    cfg = thresholds_at(point)
    per_rule = rule_window_flags(table, windows, cfg)
    return Choice(
        cfg,
        len(windows.truth),
        int(windows.truth.sum()),
        flagged,
        tp,
        {name: int(per_rule[name].sum()) for name in ALL_RULES},
        {name: int((per_rule[name] & windows.truth).sum()) for name in ALL_RULES},
    )


def tune_model(table: pd.DataFrame, labels: pd.DataFrame) -> tuple[Choice, Choice]:
    """(protocol choice without a cap, choice under FROZEN_MAX_FLAG_RATE)."""
    windows = build_windows(table, labels)
    vectors = rule_vectors(table, windows)
    protocol = search(*vectors, windows.truth)
    capped = search(*vectors, windows.truth, FROZEN_MAX_FLAG_RATE)
    return _choice_at(table, windows, *protocol), _choice_at(table, windows, *capped)


def load_model_inputs(
    data_dir: Path, model: str, expect_videos: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_path, boxes_path, signals_dir = input_paths(data_dir, model)
    check_paths([data_dir], require_split=False)
    check_paths([labels_path, boxes_path, signals_dir])
    for path in (labels_path, boxes_path):
        if not path.is_file():
            raise TuneInputError(f"missing input {path}")
    if not signals_dir.is_dir():
        raise TuneInputError(f"missing signals folder {signals_dir}")
    labels = pd.read_csv(labels_path, dtype={"video": str})
    boxes = pd.read_csv(boxes_path, dtype={"video": str})
    files = {p.stem: p for p in sorted(signals_dir.glob("*.csv"))}
    sets = {
        "labels": set(labels["video"]),
        "boxes": set(boxes["video"]),
        "signals": set(files),
    }
    if not (sets["labels"] == sets["boxes"] == sets["signals"]):
        detail = {k: len(v) for k, v in sets.items()}
        raise TuneInputError(
            f"{model}: video sets differ between labels, boxes and signals {detail}"
        )
    if len(files) != expect_videos:
        raise TuneInputError(f"{model}: {len(files)} videos, expected {expect_videos}")
    signals = pd.concat([read_signals(p) for p in files.values()], ignore_index=True)
    return join_signals_boxes(signals, boxes, allow_overlapped=True), labels


def results_row(
    model: str,
    choice: Choice,
    videos: int,
    overlapped_rows: int = 0,
    selection: str = SELECTION_CAPPED,
    frozen: bool = True,
) -> dict[str, object]:
    cfg = choice.thresholds
    row: dict[str, object] = {
        "model": model,
        "split": SPLIT,
        "selection": selection,
        "frozen": frozen,
        "videos": videos,
        "overlapped_rows": overlapped_rows,
        **dataclasses.asdict(cfg),
        "windows": choice.windows,
        "positives": choice.positives,
        "flagged": choice.flagged,
        "tp": choice.tp,
        "fp": choice.fp,
        "fn": choice.fn,
        "precision": choice.precision,
        "recall": choice.recall,
        "f1": choice.f1,
    }
    for name in ALL_RULES:
        row[f"fires_{name}"] = choice.per_rule_fires[name]
        row[f"tp_{name}"] = choice.per_rule_tp[name]
    return row


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def config_text(choices: dict[str, Choice]) -> str:
    body = {"kind": "rules"}
    for model in MODELS:
        body[model] = dataclasses.asdict(choices[model].thresholds)  # type: ignore[assignment]
    return yaml.safe_dump(body, sort_keys=False)


def run(
    models: Sequence[str],
    data_dir: Path,
    config_out: Path,
    results_out: Path,
    expect_videos: int = EXPECTED_VIDEOS,
) -> list[dict[str, object]]:
    if set(models) != set(MODELS) or len(models) != len(MODELS):
        raise TuneInputError(f"both models are needed to write the config: {list(MODELS)}")
    inputs = {m: load_model_inputs(data_dir, m, expect_videos) for m in models}
    both = {m: tune_model(*inputs[m]) for m in models}
    choices = {m: both[m][1] for m in models}
    overlapped = {m: count_overlapped(inputs[m][0]) for m in models}
    for m in models:
        sys.stdout.write(f"{m}: overlapped rows (area > 0, no box) = {overlapped[m]}\n")
    rows = [
        results_row(m, choice, int(inputs[m][1]["video"].nunique()), overlapped[m], sel, frozen)
        for m in models
        for choice, sel, frozen in (
            (both[m][0], SELECTION_PROTOCOL, False),
            (both[m][1], SELECTION_CAPPED, True),
        )
    ]
    _atomic(config_out, config_text(choices))
    _atomic(results_out, pd.DataFrame(rows).to_csv(index=False))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config-out", type=Path, default=Path("configs/detector/rules.yaml"))
    parser.add_argument("--results-out", type=Path, default=Path("results/rules_sav_val.csv"))
    parser.add_argument("--expect-videos", type=int, default=EXPECTED_VIDEOS)
    args = parser.parse_args(argv)
    root = os.environ.get(DATA_ROOT_ENV)
    data_dir = args.data_dir or (Path(root) / "tune_rules" if root else None)
    if data_dir is None:
        parser.error(f"--data-dir or {DATA_ROOT_ENV} is required")
    try:
        rows = run(args.models, data_dir, args.config_out, args.results_out, args.expect_videos)
    except (TuneInputError, ValueError, OSError) as error:
        sys.stdout.write(f"tune: refused: {error}\n")
        return 1
    for row in rows:
        sys.stdout.write(
            f"{row['model']} {row['selection']}: windows={row['windows']} "
            f"positives={row['positives']} "
            f"flagged={row['flagged']} precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1={row['f1']:.3f}\n"
        )
    sys.stdout.write(f"wrote {args.config_out} and {args.results_out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
