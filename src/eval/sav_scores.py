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
    """Our object id for an SA-V object folder, checked to be in range.

    Args:
        video: Video name, used in the error message.
        folder: Object folder name such as "000".

    Raises:
        ScoringError: If the id is outside 1..MAX_OBJECT_ID.
    """
    obj_id = sav_object_id(folder)
    if not 1 <= obj_id <= MAX_OBJECT_ID:
        raise ScoringError(f"{video}/{folder}: object id {obj_id} outside 1..{MAX_OBJECT_ID}")
    return obj_id


def write_binary_predictions(gt_root: Path, pred_root: Path, video: str, out_root: Path) -> None:
    """Label maps of one video -> <out_root>/<video>/<object folder>/<stem>.png (0 or 255)."""
    folders = sav_object_folders(gt_root, video)
    ids = {folder: _checked_id(video, folder) for folder in folders}
    # Only frames that have ground truth are written; the evaluator compares these.
    for stem in sav_gt_frames(gt_root, video):
        label: LabelMap = read_label_map(pred_root / video / f"{stem}.png")
        for folder, obj_id in ids.items():
            # An object's mask is the pixels carrying its id in the label map.
            binary = (label == obj_id).astype(np.uint8) * BINARY_ON
            target = out_root / video / folder / f"{stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(binary).save(target)


def _score_video(gt_root: Path, pred_root: Path, video: str) -> list[ScoreRow]:
    """Score one video with the official SA-V evaluator.

    Args:
        gt_root: Ground-truth folder (per-object binary PNGs).
        pred_root: Folder of per-object binary predictions in the same layout.
        video: Video name.

    Returns:
        One ScoreRow per object folder, J and F as fractions in [0, 1] (the evaluator's
        percentages divided by 100).

    Raises:
        ScoringError: If the evaluator rejects the input with an assertion.
    """
    # skip_first_and_last matches the DAVIS protocol of ignoring the end frames.
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
    """Score predicted label maps with the official SA-V evaluator.

    Args:
        gt_root: SA-V ground-truth folder (one subfolder per video, then per object).
        pred_root: Folder with one subfolder of predicted label-map PNGs per video.
        videos: Video names to score.

    Returns:
        One ScoreRow per (video, object), J and F as fractions in [0, 1], sorted.

    Raises:
        IncompleteError: If predictions are missing or malformed (checked before scoring).
        ScoringError: If the evaluator rejects a video.
    """
    check_completeness(gt_root, pred_root, videos, sav=True)
    rows: list[ScoreRow] = []
    # Per video: split label maps into binary PNGs in a temp folder, then score them.
    for video in videos:
        with tempfile.TemporaryDirectory() as temp:
            write_binary_predictions(gt_root, pred_root, video, Path(temp))
            rows.extend(_score_video(gt_root, Path(temp), video))
    return sorted_rows(rows)
