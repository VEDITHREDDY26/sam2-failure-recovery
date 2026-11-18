"""Sanity check on a CSRT box before it is used as a SAM 2 prompt.

Rules run in a fixed order and the first one that fires is the reason: none, out_of_frame,
too_small, too_large, bad_aspect, size_jump, centre_jump. The last three compare with the box
from the last reliable frame and are skipped when there is none. A rejected box is recorded by
the caller as an event; it is never replaced by another box.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

from src.common.boxes import Box, clip
from src.common.config import ConfigError, load_config, register

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "recovery" / "box_check.yaml"
REASONS = (
    "none",
    "out_of_frame",
    "too_small",
    "too_large",
    "bad_aspect",
    "size_jump",
    "centre_jump",
)


@register("box_check")
@dataclasses.dataclass(frozen=True)
class BoxCheckConfig:
    """Thresholds of the box sanity check, loaded from configs/recovery/box_check.yaml.

    Attributes:
        min_visible_fraction: Least fraction of the box area that must lie inside the frame.
        min_side_px: Shortest allowed box side in pixels.
        max_area_fraction: Largest allowed box area as a fraction of the frame area.
        min_aspect: Smallest allowed width / height.
        max_aspect: Largest allowed width / height.
        max_aspect_change: Largest allowed factor between this box's aspect ratio and the last
            reliable box's (either direction).
        max_size_ratio: Largest allowed factor between this box's area and the last reliable
            box's area (either direction).
        max_centre_jump_diagonals: Largest allowed centre move from the last reliable box, in
            units of that box's diagonal.
    """

    min_visible_fraction: float
    min_side_px: int
    max_area_fraction: float
    min_aspect: float
    max_aspect: float
    max_aspect_change: float
    max_size_ratio: float
    max_centre_jump_diagonals: float

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises:
            ConfigError: If a threshold is outside its allowed range.
        """
        # Refuse impossible thresholds when the config is loaded, not later during a run.
        if not 0 < self.min_visible_fraction <= 1:
            raise ConfigError("min_visible_fraction must be in (0, 1]")
        if not 0 < self.max_area_fraction <= 1:
            raise ConfigError("max_area_fraction must be in (0, 1]")
        if self.min_side_px < 1:
            raise ConfigError("min_side_px must be at least 1")
        if not 0 < self.min_aspect <= self.max_aspect:
            raise ConfigError("need 0 < min_aspect <= max_aspect")
        for name in ("max_aspect_change", "max_size_ratio"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1")
        if self.max_centre_jump_diagonals <= 0:
            raise ConfigError("max_centre_jump_diagonals must be positive")


def load_box_check_config(path: Path = CONFIG_PATH) -> BoxCheckConfig:
    """Load and validate the box check thresholds.

    Args:
        path: YAML file; defaults to the committed one.

    Returns:
        The thresholds.

    Raises:
        ConfigError: If the file is invalid or is not a box_check config.
    """
    cfg = load_config(path)
    if not isinstance(cfg, BoxCheckConfig):
        raise ConfigError(f"{path}: not a box_check config")
    return cfg


@dataclasses.dataclass(frozen=True)
class BoxCheckResult:
    """Verdict of the box check.

    Attributes:
        ok: True if the box may be used as a prompt.
        reason: Name of the rule that fired, one of REASONS ("none" means the tracker gave no
            box), or None when the box is ok.
    """

    ok: bool
    reason: str | None


_OK = BoxCheckResult(True, None)


def _reject(reason: str) -> BoxCheckResult:
    """Build a rejecting result with the given reason."""
    return BoxCheckResult(False, reason)


def _centre(box: Box) -> tuple[float, float]:
    """Centre (x, y) of a box in pixels."""
    return (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2


class BoxCheck:
    """Sanity check that rejects implausible tracker boxes before they are used as SAM 2 prompts.

    Attributes:
        config: Thresholds in use.
    """

    def __init__(self, config: BoxCheckConfig | None = None) -> None:
        """Use the given thresholds, or load the committed ones when none are given."""
        self.config = config or load_box_check_config()

    def check(
        self, box: Box | None, frame_shape: tuple[int, int], last_reliable_box: Box | None
    ) -> BoxCheckResult:
        """`box` is the unclipped tracker box; `frame_shape` is (height, width)."""
        if box is None:
            return _reject("none")
        cfg, (height, width) = self.config, frame_shape
        # Rules run in a fixed order; the first one that fires is the reason.
        inside = clip(box, height, width)
        if inside is None or inside.area / box.area < cfg.min_visible_fraction:
            return _reject("out_of_frame")
        if min(box.width, box.height) < cfg.min_side_px:
            return _reject("too_small")
        if box.area / (height * width) > cfg.max_area_fraction:
            return _reject("too_large")
        # Shape of the box on its own; the checks against the last reliable box follow.
        aspect = box.width / box.height
        if not cfg.min_aspect <= aspect <= cfg.max_aspect:
            return _reject("bad_aspect")
        if last_reliable_box is None:
            return _OK
        return self._check_relative(box, aspect, last_reliable_box)

    def _check_relative(self, box: Box, aspect: float, last: Box) -> BoxCheckResult:
        """Compare a box with the last reliable box: aspect change, size jump, then centre jump.

        Args:
            box: Proposed box in xyxy pixels.
            aspect: Width / height of the proposed box.
            last: Last reliable box in xyxy pixels.

        Returns:
            BoxCheckResult for the first relative rule that fires, or ok.
        """
        cfg = self.config
        change = aspect / (last.width / last.height)
        if max(change, 1 / change) > cfg.max_aspect_change:
            return _reject("bad_aspect")
        # Size may grow or shrink, so test the larger of ratio and its inverse.
        ratio = box.area / last.area
        if max(ratio, 1 / ratio) > cfg.max_size_ratio:
            return _reject("size_jump")
        # Centre move is measured in diagonals of the last reliable box.
        (cx, cy), (lx, ly) = _centre(box), _centre(last)
        if math.hypot(cx - lx, cy - ly) > cfg.max_centre_jump_diagonals * math.hypot(
            last.width, last.height
        ):
            return _reject("centre_jump")
        return _OK
