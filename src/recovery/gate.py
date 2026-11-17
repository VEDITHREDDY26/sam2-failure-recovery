"""Gate on a proposed box: decides, before the box check, whether a flagged frame is re-prompted.

Uses only the tracker's result and SAM 2's own mask box at the flagged frame. A threshold that is
None (or non-finite in the config file) is off; with both off the gate never blocks.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

from src.common.boxes import Box
from src.common.config import ConfigError, load_config, register
from src.recovery.tracker import TrackResult
from src.recovery.verify import box_iou

LOW_SCORE = "low_score"
AGREES = "agrees_with_sam2"


@dataclasses.dataclass(frozen=True)
class BoxGate:
    s_min: float | None = None
    a_max: float | None = None

    @property
    def active(self) -> bool:
        return self.s_min is not None or self.a_max is not None

    def reason(self, result: TrackResult, sam2_box: Box | None) -> str | None:
        """'low_score', 'agrees_with_sam2' or None (let the box through)."""
        if self.s_min is not None and (result.score is None or not result.score >= self.s_min):
            return LOW_SCORE
        if self.a_max is not None and box_iou(result.box, sam2_box) >= self.a_max:
            return AGREES
        return None


@register("box_gate")
@dataclasses.dataclass(frozen=True)
class BoxGateConfig:
    """File form of the gate: a non-finite value (.inf or -.inf) switches that test off."""

    s_min: float
    a_max: float

    def gate(self) -> BoxGate:
        return BoxGate(
            self.s_min if math.isfinite(self.s_min) else None,
            self.a_max if math.isfinite(self.a_max) else None,
        )


def load_box_gate(path: Path) -> BoxGateConfig:
    cfg = load_config(path)
    if not isinstance(cfg, BoxGateConfig):
        raise ConfigError(f"{path}: not a box_gate config")
    return cfg
