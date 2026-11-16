"""CSRT tracker kept aligned to SAM 2's last reliable mask.

The caller decides which frames are reliable (the detector did not flag them) and calls `reinit`
on each of them; `update` then follows the object on the frames in between. Only frames up to the
current one are ever seen. Frames are BGR uint8 HxWx3; boxes are xyxy outside and xywh only at
the OpenCV call, converted through src.common.boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

from src.common.boxes import Box, box_from_mask, clip, from_xywh, to_xywh

Status = Literal["ok", "lost", "not_initialised"]
ReinitStatus = Literal["reinitialised", "empty_mask", "init_failed"]


@dataclass(frozen=True)
class TrackResult:
    status: Status
    box: Box | None  # clipped to the frame; None unless status is "ok"
    raw_box: Box | None  # tracker output before clipping (may lie partly outside the frame)


def _check_frame(frame: npt.NDArray[np.uint8]) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 BGR frame, got {frame.dtype} {frame.shape}")


class AlignedCSRT:
    def __init__(self) -> None:
        self._tracker: cv2.Tracker | None = None
        self._shape = (0, 0)
        self.last_reliable_frame: int | None = None
        self.last_reliable_box: Box | None = None

    @property
    def initialised(self) -> bool:
        return self._tracker is not None

    def reinit(
        self, frame_bgr: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_], frame_idx: int
    ) -> ReinitStatus:
        """CSRT restarts on the tight box of a reliable mask.

        An empty mask keeps the old state. So does a box CSRT cannot start on: OpenCV's CSRT
        raises on boxes one pixel wide or tall (and on some two-pixel boxes at the image corner),
        which a thin sliver of SAM 2 mask can produce.
        """
        _check_frame(frame_bgr)
        box = box_from_mask(mask)
        if box is None:
            return "empty_mask"
        tracker = cv2.TrackerCSRT_create()  # type: ignore[attr-defined]
        try:
            tracker.init(frame_bgr, to_xywh(box))
        except cv2.error:
            return "init_failed"
        self._tracker = tracker
        self._shape = (frame_bgr.shape[0], frame_bgr.shape[1])
        self.last_reliable_frame = frame_idx
        self.last_reliable_box = box
        return "reinitialised"

    def update(self, frame_bgr: npt.NDArray[np.uint8]) -> TrackResult:
        _check_frame(frame_bgr)
        if self._tracker is None:
            return TrackResult("not_initialised", None, None)
        found, rect = self._tracker.update(frame_bgr)
        if not found:
            return TrackResult("lost", None, None)
        x, y, w, h = (int(round(v)) for v in rect)
        if w <= 0 or h <= 0:
            return TrackResult("lost", None, None)
        raw = from_xywh(x, y, w, h)
        clipped = clip(raw, *self._shape)
        if clipped is None:
            return TrackResult("lost", None, raw)
        return TrackResult("ok", clipped, raw)
