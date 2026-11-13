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
    def __init__(self, video: VideoSample) -> None:
        self.video = video

    def reset(self) -> None:
        return None

    def update(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> Decision:
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
        return None


class OracleBoxSource:
    def __init__(self, video: VideoSample) -> None:
        self.video = video
        self._last_box: dict[int, Box] = {}

    def reset(self) -> None:
        self._last_box.clear()

    def start(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        self._remember(obj_id, mask)

    def observe(self, obj_id: int, frame_index: int, frame: Frame, mask: Mask) -> None:
        self._remember(obj_id, mask)

    def propose(self, obj_id: int, frame_index: int, frame: Frame) -> TrackResult:
        raw = oracle_box_at(self.video, obj_id, frame_index)
        if raw is None:
            return TrackResult("lost", None, None)
        clipped = clip(raw, frame.shape[0], frame.shape[1])
        if clipped is None:
            return TrackResult("lost", None, raw)
        return TrackResult("ok", clipped, raw)

    def last_reliable_box(self, obj_id: int) -> Box | None:
        return self._last_box.get(obj_id)

    def _remember(self, obj_id: int, mask: Mask) -> None:
        box = box_from_mask(mask)
        if box is not None:
            self._last_box[obj_id] = box
