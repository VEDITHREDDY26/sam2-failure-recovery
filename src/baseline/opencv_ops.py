"""The only place that calls OpenCV for the baseline.

Frames are read as RGB. `rgb_to_bgr` is the single conversion and returns a `BgrImage`; the
tracker and GrabCut functions accept nothing else (checked by mypy). Boxes are xyxy outside and
xywh only at the tracker call, converted through src.common.boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from src.common.boxes import Box, clip, from_xywh, to_xywh
from src.common.masks import BoolMask

BgrImage = NewType("BgrImage", npt.NDArray[np.uint8])
RgbImage = npt.NDArray[np.uint8]


def rgb_to_bgr(image: RgbImage) -> BgrImage:
    """Explicit RGB -> BGR conversion; call it once per frame right before any OpenCV call."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 RGB image, got {image.dtype} {image.shape}")
    return BgrImage(np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), dtype=np.uint8))


@dataclass(frozen=True)
class TrackResult:
    ok: bool
    box: Box | None  # clipped to the frame; None when the tracker failed or the box is unusable


class BoxTracker(Protocol):
    def init(self, frame: BgrImage, box: Box) -> None: ...

    def update(self, frame: BgrImage) -> TrackResult: ...


def pad_to_min_side(box: Box, min_side: int, height: int, width: int) -> Box:
    """Grow a box around its centre so both sides are at least `min_side`, kept inside the frame."""

    def grow(lo: int, hi: int, limit: int) -> tuple[int, int]:
        extra = max(min_side - (hi - lo), 0)
        lo, hi = lo - extra // 2, hi + (extra - extra // 2)
        shift = max(-lo, 0) - max(hi - limit, 0)
        lo, hi = lo + shift, hi + shift
        return max(lo, 0), min(hi, limit)

    x0, x1 = grow(box.x0, box.x1, width)
    y0, y1 = grow(box.y0, box.y1, height)
    return Box(x0, y0, x1, y1)


class CsrtTracker:
    def __init__(self, min_side: int) -> None:
        self._min_side = min_side
        self._tracker = cv2.TrackerCSRT_create()  # type: ignore[attr-defined]
        self._shape = (0, 0)

    def init(self, frame: BgrImage, box: Box) -> None:
        self._shape = (frame.shape[0], frame.shape[1])
        self._tracker.init(frame, to_xywh(box))

    def update(self, frame: BgrImage) -> TrackResult:
        ok, rect = self._tracker.update(frame)
        if not ok:
            return TrackResult(False, None)
        x, y, w, h = (int(round(v)) for v in rect)
        if w <= 0 or h <= 0:
            return TrackResult(True, None)
        clipped = clip(from_xywh(x, y, w, h), *self._shape)
        if clipped is None or min(clipped.width, clipped.height) < self._min_side:
            return TrackResult(True, None)
        return TrackResult(True, clipped)


def grabcut_in_box(frame: BgrImage, box: Box, iterations: int, margin_frac: float) -> BoolMask:
    """GrabCut with a rectangle on a crop around the box; full-frame bool mask, empty on failure."""
    height, width = frame.shape[:2]
    mx, my = round(box.width * margin_frac), round(box.height * margin_frac)
    roi = clip(Box(box.x0 - mx, box.y0 - my, box.x1 + mx, box.y1 + my), height, width)
    out = np.zeros((height, width), dtype=np.bool_)
    if roi is None:
        return out
    crop = np.ascontiguousarray(frame[roi.y0 : roi.y1, roi.x0 : roi.x1])
    labels = np.zeros(crop.shape[:2], dtype=np.uint8)
    rect = (box.x0 - roi.x0, box.y0 - roi.y0, box.width, box.height)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(crop, labels, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return out
    foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    out[roi.y0 : roi.y1, roi.x0 : roi.x1] = foreground
    return out
