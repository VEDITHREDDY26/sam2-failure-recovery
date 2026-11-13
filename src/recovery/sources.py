"""Seams the recovery loop talks to: the failure detector and the box source.

Both are protocols so the real system (rule detector, aligned CSRT) and oracle runs plug in
without this package importing the oracle or the detector code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from src.common.boxes import Box, box_from_mask
from src.recovery.tracker import AlignedCSRT, TrackResult

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

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult:
        return self._trackers.setdefault(obj_id, AlignedCSRT()).update(frame)

    def last_reliable_box(self, obj_id: int) -> Box | None:
        return self._last_box.get(obj_id)

    def _remember(self, obj_id: int, mask: Mask) -> None:
        box = box_from_mask(mask)
        if box is not None:
            self._last_box[obj_id] = box
