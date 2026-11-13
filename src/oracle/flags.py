"""Oracle failure flags and boxes from ground truth, for oracle runs only.

Nothing under src/recovery or src/detection may import this package. An unannotated frame
(SA-V) gives None: it is unknown, never treated as an empty mask.
"""

from __future__ import annotations

import numpy as np

from src.common.boxes import Box, box_from_mask
from src.common.masks import BoolMask
from src.data.video import VideoSample

ORACLE_FAILURE_IOU = 0.5


def oracle_iou(pred_mask: BoolMask, gt_mask: BoolMask) -> float:
    """Intersection over union; two empty masks count as a perfect match (1.0)."""
    union = int(np.count_nonzero(pred_mask | gt_mask))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(pred_mask & gt_mask)) / union


def oracle_flag(pred_mask: BoolMask, gt_mask: BoolMask | None) -> bool | None:
    """True when the prediction fails against ground truth; None on an unannotated frame."""
    if gt_mask is None:
        return None
    return oracle_iou(pred_mask, gt_mask) < ORACLE_FAILURE_IOU


def oracle_box(gt_mask: BoolMask | None) -> Box | None:
    """Tight box of the ground-truth mask; None if unannotated or the object is absent."""
    if gt_mask is None:
        return None
    return box_from_mask(gt_mask)


def oracle_flag_at(
    video: VideoSample, obj_id: int, frame_index: int, pred_mask: BoolMask
) -> bool | None:
    return oracle_flag(pred_mask, video.gt_mask(obj_id, frame_index))


def oracle_box_at(video: VideoSample, obj_id: int, frame_index: int) -> Box | None:
    return oracle_box(video.gt_mask(obj_id, frame_index))
