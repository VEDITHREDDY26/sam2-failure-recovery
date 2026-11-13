"""One event row per recovery trigger (a flagged object at a frame)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.common.boxes import Box

Action = Literal["reprompted", "rejected", "tracker_lost", "cooldown"]
EVENT_COLUMNS = (
    "video",
    "object_id",
    "frame_index",
    "fired",
    "action",
    "reason",
    "box_x0",
    "box_y0",
    "box_x1",
    "box_y1",
    "restart_frames",
)


@dataclass(frozen=True)
class Event:
    video: str
    object_id: int
    frame_index: int
    fired: tuple[str, ...]
    action: Action
    reason: str = ""
    raw_box: Box | None = None
    restart_frames: int = 0

    def as_row(self) -> tuple[object, ...]:
        box = self.raw_box
        corners = (box.x0, box.y0, box.x1, box.y1) if box else ("", "", "", "")
        return (
            self.video,
            self.object_id,
            self.frame_index,
            "+".join(self.fired),
            self.action,
            self.reason,
            *corners,
            self.restart_frames,
        )
