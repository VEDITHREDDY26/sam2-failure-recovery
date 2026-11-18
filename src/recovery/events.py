"""One event row per recovery trigger (a flagged object at a frame)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.common.boxes import Box

Action = Literal["reprompted", "rejected", "tracker_lost", "cooldown", "verify_rejected", "gated"]
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
    """One recovery trigger: what happened when the detector flagged an object at a frame.

    Attributes:
        video: Video name.
        object_id: Object id.
        frame_index: 0-based frame index of the flag.
        fired: Names of the detector rules that fired.
        action: What the loop did: "reprompted", "rejected" (box check), "tracker_lost",
            "cooldown", "verify_rejected" or "gated".
        reason: Why a box was rejected, gated or lost (empty when re-prompted).
        raw_box: Unclipped proposed box in xyxy pixels, or None if there was none.
        restart_frames: Number of frames replayed to restart SAM 2 for a re-prompt (frame_index
            minus the prompt frame), else 0.
    """

    video: str
    object_id: int
    frame_index: int
    fired: tuple[str, ...]
    action: Action
    reason: str = ""
    raw_box: Box | None = None
    restart_frames: int = 0

    def as_row(self) -> tuple[object, ...]:
        """Flatten the event into a tuple in EVENT_COLUMNS order.

        Returns:
            Values for one CSV row; fired names are joined with "+" and a missing box gives four
            empty strings.
        """
        box = self.raw_box
        # The four box columns stay empty when the event has no proposed box.
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
