"""Oracle detector and oracle box source for the recovery loop (oracle runs only).

They read ground truth through a VideoSample and satisfy the protocols in
src.recovery.sources. Nothing under src/recovery or src/detection imports this module.
An unannotated frame gives no flag; an object absent from ground truth gives no box, which the
loop treats like a lost tracker (recorded, never replaced).
"""

from __future__ import annotations

from src.common.boxes import Box, box_from_mask, clip
from src.data.video import VideoSample
from src.oracle.flags import oracle_box_at, oracle_flag_at
from src.recovery.sources import Decision, Frame, Mask
from src.recovery.tracker import TrackResult

FIRED = ("oracle",)


class OracleDetector:
    """Failure detector that flags a frame when the mask disagrees with ground truth.

    Implements the Detector protocol from src.recovery.sources. It keeps no history.

    Attributes:
        video: Video sample that provides the ground-truth masks.
    """

    def __init__(self, video: VideoSample) -> None:
        """Store the video sample whose ground truth is read."""
        self.video = video

    def reset(self) -> None:
        """Nothing to reset; the oracle keeps no per-video state."""
        return None

    def update(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> Decision:
        """Flag the frame if the mask's IoU with ground truth is below the oracle threshold.

        Args:
            obj_id: Object id.
            frame_index: 0-based frame index.
            mask: Predicted boolean mask at this frame.
            object_score: SAM 2 object score (unused by the oracle).
            frame_shape: (height, width) in pixels (unused by the oracle).

        Returns:
            Decision with flag True and fired = ("oracle",) on failure; flag False otherwise, which
            includes unannotated frames.
        """
        flag = oracle_flag_at(self.video, obj_id, frame_index, mask)
        return Decision(True, FIRED) if flag else Decision(False)

    def observe(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> None:
        """Do nothing; the oracle has no history to set."""
        return None


class OracleBoxSource:
    """Box source that proposes the ground-truth box (oracle runs only).

    Implements the BoxSource protocol from src.recovery.sources. The reliable-box memory still
    comes from SAM 2's own masks, as in the real system.

    Attributes:
        video: Video sample that provides the ground-truth masks.
    """

    def __init__(self, video: VideoSample) -> None:
        """Store the video sample and start with no remembered boxes."""
        self.video = video
        self._last_box: dict[int, Box] = {}

    def reset(self) -> None:
        """Forget the remembered mask boxes of every object."""
        self._last_box.clear()

    def start(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """Remember the box of the reliable mask at the prompt or recovered frame."""
        self._remember(obj_id, mask)

    def observe(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        """Remember the box of an unflagged frame's mask."""
        self._remember(obj_id, mask)

    def skip(self, obj_id: int, frame_index: int, frame: Frame) -> None:
        """Do nothing for a flagged frame inside the cooldown."""
        return None

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult:
        """Return the ground-truth box at this frame, clipped to the image.

        Args:
            obj_id: Object id.
            frame_index: 0-based frame index.
            frame: BGR uint8 image of this frame; only its height and width are used.

        Returns:
            TrackResult "ok" with the clipped box and the raw box, or "lost" when ground truth
            has no object here or the box lies outside the image.
        """
        raw = oracle_box_at(self.video, obj_id, frame_index)
        if raw is None:
            return TrackResult("lost", None, None)
        # Ground truth may touch the border or leave the image; clip, and report a lost result if
        # nothing is left.
        clipped = clip(raw, frame.shape[0], frame.shape[1])
        if clipped is None:
            return TrackResult("lost", None, raw)
        return TrackResult("ok", clipped, raw)

    def last_reliable_box(self, obj_id: int) -> Box | None:
        """Box of the last reliable mask of the object, or None if there is none."""
        return self._last_box.get(obj_id)

    def _remember(self, obj_id: int, mask: Mask) -> None:
        """Store the tight box of a mask as the last reliable box (empty masks are ignored)."""
        box = box_from_mask(mask)
        if box is not None:
            self._last_box[obj_id] = box
