"""Failure labels: a frame is a failure for an object when its mask IoU with ground truth is < 0.5.

This is the definition of a failure, not a detector setting. Only evaluation code reads it; no
detector may import this module (checked by tests/detection/test_label_isolation.py).
Frames without annotation (SA-V) get no label at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.layout import mask_path
from src.common.masks import BoolMask, read_label_map
from src.data.video import NeverVisibleError, VideoSample

FAILURE_IOU = 0.5
LABEL_COLUMNS = ("video", "object_id", "frame_index", "iou", "failure")


def mask_iou(pred: BoolMask, gt: BoolMask) -> float:
    """Intersection over union; two empty masks count as a perfect match (1.0)."""
    union = int(np.count_nonzero(pred | gt))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(pred & gt)) / union


def is_failure(iou: float) -> bool:
    return iou < FAILURE_IOU


def video_labels(video: VideoSample, output_root: Path, method: str, split: str) -> pd.DataFrame:
    """One row per (object, annotated frame at or after the object's prompt frame).

    Objects that are never visible were never run and are skipped.
    """
    rows: list[tuple[str, int, int, float, bool]] = []
    for obj_id in video.object_ids:
        try:
            start = video.first_box(obj_id).frame_index
        except NeverVisibleError:
            continue
        for frame_index in video.annotated_frames(obj_id):
            if frame_index < start:
                continue
            gt = video.gt_mask(obj_id, frame_index)
            if gt is None:
                continue
            stem = video.frame_stems[frame_index]
            label = read_label_map(mask_path(output_root, method, split, video.name, stem))
            iou = mask_iou(label == obj_id, gt)
            rows.append((video.name, obj_id, frame_index, iou, is_failure(iou)))
    frame = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    return frame.astype({"failure": bool})


def count_skipped_objects(video: VideoSample) -> int:
    skipped = 0
    for obj_id in video.object_ids:
        try:
            video.first_box(obj_id)
        except NeverVisibleError:
            skipped += 1
    return skipped
