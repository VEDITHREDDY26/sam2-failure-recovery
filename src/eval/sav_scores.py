"""Per-object J and F from the official SA-V evaluator (pinned, unmodified).

SA-V GT: <gt_root>/<video>/<object folder>/<stem>.png, binary, every 4th 24 fps frame.
Our stored predictions are indexed label maps, one per 24 fps frame. For each video the
label maps are split into per-object binary PNGs at the GT frame names in a temporary folder
and scored with `VideoEvaluator(..., skip_first_and_last=True)`. The official scores are in
percent and are divided by 100 so both datasets report values in [0,1].

Known limitation: SA-V objects can overlap but a stored label map holds one object per pixel,
so overlapping objects can lose pixels. Scoring per-object GT directly is not affected.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.masks import MAX_OBJECT_ID, LabelMap, read_label_map
from src.eval.completeness import check_completeness, sav_gt_frames, sav_object_folders
from src.eval.evaluators import load_sav_benchmark
from src.eval.rows import ScoreRow, ScoringError, sorted_rows

PERCENT = 100.0
BINARY_ON = 255


def sav_object_id(folder: str) -> int:
    """Our object id for an SA-V object folder: "000" -> 1 (proposed; matches the SA-V loader)."""
    return int(folder) + 1


def _checked_id(video: str, folder: str) -> int:
    obj_id = sav_object_id(folder)
    if not 1 <= obj_id <= MAX_OBJECT_ID:
        raise ScoringError(f"{video}/{folder}: object id {obj_id} outside 1..{MAX_OBJECT_ID}")
    return obj_id


def write_binary_predictions(gt_root: Path, pred_root: Path, video: str, out_root: Path) -> None:
    """Label maps of one video -> <out_root>/<video>/<object folder>/<stem>.png (0 or 255)."""
    folders = sav_object_folders(gt_root, video)
    ids = {folder: _checked_id(video, folder) for folder in folders}
    for stem in sav_gt_frames(gt_root, video):
        label: LabelMap = read_label_map(pred_root / video / f"{stem}.png")
        for folder, obj_id in ids.items():
            binary = (label == obj_id).astype(np.uint8) * BINARY_ON
            target = out_root / video / folder / f"{stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(binary).save(target)


def _score_video(gt_root: Path, pred_root: Path, video: str) -> list[ScoreRow]:
    evaluator = load_sav_benchmark().VideoEvaluator(
        str(gt_root), str(pred_root), skip_first_and_last=True
    )
    try:
        _, iou, boundary_f = evaluator(video)
    except AssertionError as error:
        raise ScoringError(f"{video}: the SA-V evaluator rejected the input: {error}") from error
    return [
        ScoreRow(video, sav_object_id(folder), float(iou[folder]) / PERCENT, float(f) / PERCENT)
        for folder, f in boundary_f.items()
    ]


def score_sav(gt_root: Path, pred_root: Path, videos: Sequence[str]) -> list[ScoreRow]:
    check_completeness(gt_root, pred_root, videos, sav=True)
    rows: list[ScoreRow] = []
    for video in videos:
        with tempfile.TemporaryDirectory() as temp:
            write_binary_predictions(gt_root, pred_root, video, Path(temp))
            rows.extend(_score_video(gt_root, Path(temp), video))
    return sorted_rows(rows)
