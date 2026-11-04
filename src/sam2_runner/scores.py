"""Per-frame object score for one object, read from SAM 2's inference state.

SAM 2 keeps `object_score_logits` (shape 1x1) in the output dict of every tracked frame
(sam2/sam2_video_predictor.py:794-801), under `output_dict_per_obj[obj_idx]` in
`cond_frame_outputs` (prompted frames) or `non_cond_frame_outputs`. The object index comes from
`obj_id_to_idx`. Only the current frame is ever read.

If the score is missing, the fallback is the mean mask logit over the predicted-foreground
pixels (logit > 0); for an empty mask it is the maximum logit, which is <= 0, so the sign still
says present or absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from src.sam2_runner.predictor import to_numpy

STORAGE_KEYS = ("cond_frame_outputs", "non_cond_frame_outputs")
SCORE_KEY = "object_score_logits"


def read_stored_score(state: Mapping[str, Any], obj_id: int, frame_idx: int) -> float | None:
    """SAM 2's object-score logit for this object at this frame, or None if it is missing."""
    obj_idx = state.get("obj_id_to_idx", {}).get(obj_id)
    outputs = state.get("output_dict_per_obj", {})
    if obj_idx is None or obj_idx not in outputs:
        return None
    for key in STORAGE_KEYS:
        frame_out = outputs[obj_idx].get(key, {}).get(frame_idx)
        if frame_out is None:
            continue
        value = frame_out.get(SCORE_KEY)
        if value is None:
            return None
        values = to_numpy(value).reshape(-1)
        if values.size != 1:
            raise ValueError(f"{SCORE_KEY} has {values.size} values, expected 1")
        return float(values[0])
    return None


def fallback_score(obj_logits: npt.NDArray[np.floating]) -> float:
    """Mean logit over pixels with logit > 0; the maximum logit when there are none."""
    foreground = obj_logits > 0
    if foreground.any():
        return float(obj_logits[foreground].astype(np.float64).mean())
    return float(obj_logits.max())


def object_score(
    state: Mapping[str, Any], obj_id: int, frame_idx: int, obj_logits: npt.NDArray[np.floating]
) -> tuple[float, bool]:
    """(score, True if it is SAM 2's own object score, False if the fallback was used)."""
    stored = read_stored_score(state, obj_id, frame_idx)
    if stored is not None:
        return stored, True
    return fallback_score(obj_logits), False
