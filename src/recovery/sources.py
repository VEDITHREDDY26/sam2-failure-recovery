"""Seams the recovery loop talks to: the failure detector and the box source.

Both are protocols so the real system (rule detector, aligned CSRT) and oracle runs plug in
without this package importing the oracle or the detector code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from src.common.boxes import Box, box_from_mask, clip
from src.recovery.tracker import AlignedCSRT, TrackResult
from src.recovery.verify import box_iou
from src.recovery.vit_tracker import ViTFactory, ViTTracker

Mask = npt.NDArray[np.bool_]
Frame = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Decision:
    flag: bool
    fired: tuple[str, ...] = ()


class Detector(Protocol):
    def reset(self) -> None: ...

    def update(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> Decision: ...

    def observe(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> None:
        """Set the history without flagging (prompt frame, recovered frame)."""


class BoxSource(Protocol):
    def reset(self) -> None: ...

    def start(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """A reliable observation that restarts the source (prompt frame, recovered frame)."""

    def observe(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """A frame the detector did not flag."""

    def skip(self, obj_id: int, frame_index: int, frame: Frame) -> None:
        """A flagged frame inside the cooldown: no proposal is made, the frame only goes by."""

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult: ...

    def last_reliable_box(self, obj_id: int) -> Box | None: ...


class CSRTBoxSource:
    """One aligned CSRT tracker per object.

    reinit: the tracker restarts on every unflagged frame's mask. continuous: it restarts only
    at start and after a recovery, and follows the object on every unflagged frame.
    """

    def __init__(self, tracker_mode: str = "reinit") -> None:
        if tracker_mode not in ("reinit", "continuous"):
            raise ValueError(f"unknown tracker_mode '{tracker_mode}'")
        self.tracker_mode = tracker_mode
        self._trackers: dict[int, AlignedCSRT] = {}
        self._last_box: dict[int, Box] = {}

    def reset(self) -> None:
        self._trackers.clear()
        self._last_box.clear()

    def start(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        self._trackers.setdefault(obj_id, AlignedCSRT()).reinit(frame, mask, frame_index)
        self._remember(obj_id, mask)

    def observe(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        tracker = self._trackers.setdefault(obj_id, AlignedCSRT())
        if self.tracker_mode == "reinit":
            tracker.reinit(frame, mask, frame_index)
        else:
            tracker.update(frame)
        self._remember(obj_id, mask)

    def skip(self, obj_id: int, frame_index: int, frame: Frame) -> None:
        return None

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult:
        return self._trackers.setdefault(obj_id, AlignedCSRT()).update(frame)

    def last_reliable_box(self, obj_id: int) -> Box | None:
        return self._last_box.get(obj_id)

    def _remember(self, obj_id: int, mask: Mask) -> None:
        box = box_from_mask(mask)
        if box is not None:
            self._last_box[obj_id] = box


VIT_MODES = ("reinit", "continuous", "anchored")


class ViTBoxSource:
    """One ViT tracker per object, following the same start / observe / propose contract.

    reinit: restart on every unflagged mask. continuous: restart only at start and after a
    recovery; the tracker is updated on every frame, flagged ones inside the cooldown included.
    anchored: continuous, but an unflagged frame also restarts the tracker on SAM 2's mask box
    when the tracker's own box overlaps it by at least anchor_iou, or when the tracker has no box
    (lost or not started); an empty mask never restarts it. continuous never restarts.
    """

    def __init__(self, factory: ViTFactory, mode: str = "reinit", anchor_iou: float = 0.5) -> None:
        if mode not in VIT_MODES:
            raise ValueError(f"unknown ViT mode '{mode}'")
        self.mode = mode
        self.anchor_iou = anchor_iou
        self._factory = factory
        self._trackers: dict[int, ViTTracker] = {}
        self._last_box: dict[int, Box] = {}

    def reset(self) -> None:
        self._trackers.clear()
        self._last_box.clear()

    def start(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        self._tracker(obj_id).reinit(frame, mask, frame_index)
        self._remember(obj_id, mask)

    def observe(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        tracker = self._tracker(obj_id)
        if self.mode == "reinit":
            tracker.reinit(frame, mask, frame_index)
        else:
            result = tracker.update(frame)
            if self.mode == "anchored" and (
                result.box is None or box_iou(result.box, box_from_mask(mask)) >= self.anchor_iou
            ):
                tracker.reinit(frame, mask, frame_index)
        self._remember(obj_id, mask)

    def skip(self, obj_id: int, frame_index: int, frame: Frame) -> None:
        if self.mode != "reinit":
            self._tracker(obj_id).update(frame)

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult:
        return self._tracker(obj_id).update(frame)

    def last_reliable_box(self, obj_id: int) -> Box | None:
        return self._last_box.get(obj_id)

    def _tracker(self, obj_id: int) -> ViTTracker:
        if obj_id not in self._trackers:
            self._trackers[obj_id] = ViTTracker(self._factory)
        return self._trackers[obj_id]

    def _remember(self, obj_id: int, mask: Mask) -> None:
        box = box_from_mask(mask)
        if box is not None:
            self._last_box[obj_id] = box


class MotionBoxSource:
    """Constant-velocity extrapolation of the last two reliable SAM 2 mask boxes; no appearance."""

    def __init__(self) -> None:
        self._history: dict[int, list[tuple[int, Box]]] = {}

    def reset(self) -> None:
        self._history.clear()

    def start(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        self._push(obj_id, frame_index, mask)

    def observe(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        self._push(obj_id, frame_index, mask)

    def skip(self, obj_id: int, frame_index: int, frame: Frame) -> None:
        return None

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult:
        history = self._history.get(obj_id, [])
        if not history:
            return TrackResult("not_initialised", None, None)
        frame_b, box_b = history[-1]
        if len(history) == 1:
            raw: Box | None = box_b
        else:
            frame_a, box_a = history[-2]
            raw = _extrapolate(box_a, box_b, (frame_index - frame_b) / (frame_b - frame_a))
        if raw is None:
            return TrackResult("lost", None, None)
        clipped = clip(raw, frame.shape[0], frame.shape[1])
        if clipped is None:
            return TrackResult("lost", None, raw)
        return TrackResult("ok", clipped, raw)

    def last_reliable_box(self, obj_id: int) -> Box | None:
        history = self._history.get(obj_id)
        return history[-1][1] if history else None

    def _push(self, obj_id: int, frame_index: int, mask: Mask) -> None:
        box = box_from_mask(mask)
        if box is not None:
            self._history.setdefault(obj_id, []).append((frame_index, box))
            del self._history[obj_id][:-2]


def _extrapolate(a: Box, b: Box, steps: float) -> Box | None:
    """b + steps * (b - a) per coordinate; None when the result has no area."""
    x0, y0, x1, y1 = (
        round(q + steps * (q - p))
        for p, q in ((a.x0, b.x0), (a.y0, b.y0), (a.x1, b.x1), (a.y1, b.y1))
    )
    return Box(x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
