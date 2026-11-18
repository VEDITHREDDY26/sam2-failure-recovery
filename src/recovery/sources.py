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
    """Outcome of the detector for one object at one frame.

    Attributes:
        flag: True if the detector says SAM 2 has lost the object here.
        fired: Names of the rules or detector parts that fired (empty when not flagged).
    """

    flag: bool
    fired: tuple[str, ...] = ()


class Detector(Protocol):
    """Protocol of a failure detector, fed SAM 2's own outputs frame by frame (no ground truth).

    Implemented by the rule detector, the learned detector adapter and the oracle detector.
    """

    def reset(self) -> None:
        """Clear all per-object history before a new video."""
        ...

    def update(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> Decision:
        """Look at one object at one frame and decide whether SAM 2 has lost it.

        Args:
            obj_id: Object id.
            frame_index: 0-based frame index.
            mask: SAM 2's boolean mask for the object at this frame.
            object_score: SAM 2's object-score logit at this frame.
            frame_shape: (height, width) of the frame in pixels.

        Returns:
            A Decision with the flag and the names of the signals that fired.
        """
        ...

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
    """Protocol of a source of prompt boxes for flagged frames (CSRT, ViT, motion or oracle).

    The loop calls start on reliable frames that restart the source, observe on unflagged frames,
    skip on flagged frames inside the cooldown, and propose on flagged frames that may be
    re-prompted.
    """

    def reset(self) -> None:
        """Forget all per-object state before a new video."""
        ...

    def start(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """A reliable observation that restarts the source (prompt frame, recovered frame)."""

    def observe(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """A frame the detector did not flag."""

    def skip(self, obj_id: int, frame_index: int, frame: Frame) -> None:
        """A flagged frame inside the cooldown: no proposal is made, the frame only goes by."""

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult:
        """Propose a box for a flagged frame.

        Args:
            obj_id: Object id.
            frame_index: 0-based frame index.
            frame: BGR uint8 image of this frame.

        Returns:
            A TrackResult; its raw box is the unclipped xyxy proposal and its box is clipped to the
            frame (None unless the status is "ok").
        """
        ...

    def last_reliable_box(self, obj_id: int) -> Box | None:
        """Tight box of the last reliable SAM 2 mask of the object.

        Returns:
            The box in xyxy pixels, or None if no reliable mask was seen yet.
        """
        ...


class CSRTBoxSource:
    """One aligned CSRT tracker per object.

    reinit: the tracker restarts on every unflagged frame's mask. continuous: it restarts only
    at start and after a recovery, and follows the object on every unflagged frame.
    """

    def __init__(self, tracker_mode: str = "reinit") -> None:
        """Create an empty source.

        Args:
            tracker_mode: "reinit" or "continuous" (see the class docstring).

        Raises:
            ValueError: If the mode is unknown.
        """
        if tracker_mode not in ("reinit", "continuous"):
            raise ValueError(f"unknown tracker_mode '{tracker_mode}'")
        self.tracker_mode = tracker_mode
        self._trackers: dict[int, AlignedCSRT] = {}
        self._last_box: dict[int, Box] = {}

    def reset(self) -> None:
        """Drop every tracker and remembered box."""
        self._trackers.clear()
        self._last_box.clear()

    def start(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """Restart the object's tracker on the reliable mask and remember its box."""
        self._trackers.setdefault(obj_id, AlignedCSRT()).reinit(frame, mask, frame_index)
        self._remember(obj_id, mask)

    def observe(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """Handle an unflagged frame: restart the tracker (reinit) or follow the object."""
        tracker = self._trackers.setdefault(obj_id, AlignedCSRT())
        if self.tracker_mode == "reinit":
            tracker.reinit(frame, mask, frame_index)
        else:
            tracker.update(frame)
        self._remember(obj_id, mask)

    def skip(self, obj_id: int, frame_index: int, frame: Frame) -> None:
        """Do nothing for a flagged frame inside the cooldown."""
        return None

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult:
        """Run the object's tracker on this frame and return its result."""
        return self._trackers.setdefault(obj_id, AlignedCSRT()).update(frame)

    def last_reliable_box(self, obj_id: int) -> Box | None:
        """Box of the last reliable mask of the object, or None."""
        return self._last_box.get(obj_id)

    def _remember(self, obj_id: int, mask: Mask) -> None:
        """Store the tight box of a mask as the last reliable box (empty masks are ignored)."""
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
        """Create an empty source.

        Args:
            factory: Callable that builds a new raw ViT tracker.
            mode: One of VIT_MODES (see the class docstring).
            anchor_iou: Minimum IoU between the tracker box and SAM 2's mask box for the anchored
                mode to restart the tracker.

        Raises:
            ValueError: If the mode is unknown.
        """
        if mode not in VIT_MODES:
            raise ValueError(f"unknown ViT mode '{mode}'")
        self.mode = mode
        self.anchor_iou = anchor_iou
        self._factory = factory
        self._trackers: dict[int, ViTTracker] = {}
        self._last_box: dict[int, Box] = {}

    def reset(self) -> None:
        """Drop every tracker and remembered box."""
        self._trackers.clear()
        self._last_box.clear()

    def start(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """Restart the object's tracker on the reliable mask and remember its box."""
        self._tracker(obj_id).reinit(frame, mask, frame_index)
        self._remember(obj_id, mask)

    def observe(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """Handle an unflagged frame according to the mode (reinit, continuous or anchored)."""
        tracker = self._tracker(obj_id)
        if self.mode == "reinit":
            tracker.reinit(frame, mask, frame_index)
        else:
            # Follow the object; in anchored mode restart on SAM 2's mask when the tracker agrees
            # with it or has no box.
            result = tracker.update(frame)
            if self.mode == "anchored" and (
                result.box is None or box_iou(result.box, box_from_mask(mask)) >= self.anchor_iou
            ):
                tracker.reinit(frame, mask, frame_index)
        self._remember(obj_id, mask)

    def skip(self, obj_id: int, frame_index: int, frame: Frame) -> None:
        """Keep a non-reinit tracker following the object on a flagged frame inside the cooldown."""
        if self.mode != "reinit":
            self._tracker(obj_id).update(frame)

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult:
        """Run the object's tracker on this frame and return its result (box and score)."""
        return self._tracker(obj_id).update(frame)

    def last_reliable_box(self, obj_id: int) -> Box | None:
        """Box of the last reliable mask of the object, or None."""
        return self._last_box.get(obj_id)

    def _tracker(self, obj_id: int) -> ViTTracker:
        """Return the object's tracker, creating it on first use."""
        if obj_id not in self._trackers:
            self._trackers[obj_id] = ViTTracker(self._factory)
        return self._trackers[obj_id]

    def _remember(self, obj_id: int, mask: Mask) -> None:
        """Store the tight box of a mask as the last reliable box (empty masks are ignored)."""
        box = box_from_mask(mask)
        if box is not None:
            self._last_box[obj_id] = box


class MotionBoxSource:
    """Constant-velocity extrapolation of the last two reliable SAM 2 mask boxes; no appearance."""

    def __init__(self) -> None:
        """Create a source with no history."""
        self._history: dict[int, list[tuple[int, Box]]] = {}

    def reset(self) -> None:
        """Forget the box history of every object."""
        self._history.clear()

    def start(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """Record the box of the reliable mask at the prompt or recovered frame."""
        self._push(obj_id, frame_index, mask)

    def observe(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """Record the box of an unflagged frame's mask."""
        self._push(obj_id, frame_index, mask)

    def skip(self, obj_id: int, frame_index: int, frame: Frame) -> None:
        """Do nothing for a flagged frame inside the cooldown."""
        return None

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult:
        """Extrapolate the object's box to this frame from the last two reliable boxes.

        With one reliable box it is returned as is. The result is clipped to the frame.

        Args:
            obj_id: Object id.
            frame_index: 0-based frame index.
            frame: BGR uint8 image of this frame; only its height and width are used.

        Returns:
            "ok" with the box, "lost" if the extrapolation has no area or leaves the image, or
            "not_initialised" if no reliable box was seen.
        """
        history = self._history.get(obj_id, [])
        if not history:
            return TrackResult("not_initialised", None, None)
        # Newest reliable box (b) and, if present, the one before it (a) give a constant velocity.
        frame_b, box_b = history[-1]
        if len(history) == 1:
            raw: Box | None = box_b
        else:
            frame_a, box_a = history[-2]
            raw = _extrapolate(box_a, box_b, (frame_index - frame_b) / (frame_b - frame_a))
        if raw is None:
            return TrackResult("lost", None, None)
        # Same clip-or-lost handling as the other box sources.
        clipped = clip(raw, frame.shape[0], frame.shape[1])
        if clipped is None:
            return TrackResult("lost", None, raw)
        return TrackResult("ok", clipped, raw)

    def last_reliable_box(self, obj_id: int) -> Box | None:
        """Box of the last reliable mask of the object, or None."""
        history = self._history.get(obj_id)
        return history[-1][1] if history else None

    def _push(self, obj_id: int, frame_index: int, mask: Mask) -> None:
        """Append a reliable mask's box to the history, keeping only the last two."""
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
