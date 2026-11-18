"""Adapter that plugs the streaming rule detector into the recovery loop.

One RuleDetector per object id; observations are built from the bool mask, SAM 2's object score
and the frame shape, all available at frame t. Satisfies src.recovery.sources.Detector.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.detection.rules import (
    ALL_RULES,
    DEFAULT_RULES_PATH,
    RuleDetector,
    RuleThresholds,
    load_rules_config,
)
from src.detection.signals import FrameObs, obs_from_mask
from src.recovery.sources import Decision

Mask = npt.NDArray[np.bool_]


class OnlineRuleDetector:
    """Rule detector for the live recovery loop, with one streaming detector per object id.

    Only the mask, SAM 2's object score and the frame shape at the current frame are used, so no
    ground truth and no future frames are seen.

    Args:
        thresholds: Rule thresholds (see RuleThresholds).
        enabled: Names of the rules that may fire; all four by default.

    Attributes:
        thresholds: The thresholds in use.
        enabled: The enabled rule names.
    """

    def __init__(self, thresholds: RuleThresholds, enabled: Iterable[str] = ALL_RULES) -> None:
        """Store the settings and check the rule names; per-object detectors are made on demand."""
        self.thresholds = thresholds
        self.enabled = tuple(enabled)
        # Build one detector now so unknown rule names fail at construction.
        RuleDetector(thresholds, self.enabled)
        self._detectors: dict[int, RuleDetector] = {}

    @classmethod
    def from_config(
        cls, model: str, path: str | Path = DEFAULT_RULES_PATH, enabled: Iterable[str] = ALL_RULES
    ) -> OnlineRuleDetector:
        """Build a detector with the thresholds of one SAM 2 model from the rules config file.

        Args:
            model: "tiny" or "large".
            path: Rules YAML file; the frozen detector settings by default.
            enabled: Names of the rules that may fire.
        """
        return cls(load_rules_config(model, path), enabled)

    def reset(self) -> None:
        """Forget all per-object history (call before a new video)."""
        self._detectors.clear()

    def update(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> Decision:
        """Feed one frame of one object and return the detector's decision for it.

        Args:
            obj_id: Object id.
            frame_index: 0-based frame index (not used by the rules, which only compare with the
                previously fed frame).
            mask: Boolean HxW mask of the object on this frame.
            object_score: SAM 2's object score for this frame.
            frame_shape: Frame (height, width) in pixels; must equal the mask shape.

        Returns:
            A Decision with the flag and the names of the rules that fired.

        Raises:
            ValueError: If the mask shape differs from the frame shape.
        """
        obs = self._obs(mask, object_score, frame_shape)
        # Each object has its own detector, so one object's history never affects another.
        result = self._detector(obj_id).update(obs)
        return Decision(result.flag, result.fired)

    def observe(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> None:
        """Feed one frame like update but discard the decision; it only advances the history."""
        self._detector(obj_id).update(self._obs(mask, object_score, frame_shape))

    def _detector(self, obj_id: int) -> RuleDetector:
        """Return the streaming detector of an object, creating it on first use."""
        if obj_id not in self._detectors:
            self._detectors[obj_id] = RuleDetector(self.thresholds, self.enabled)
        return self._detectors[obj_id]

    @staticmethod
    def _obs(mask: Mask, object_score: float, frame_shape: tuple[int, int]) -> FrameObs:
        """Build a FrameObs from a mask and score, checking the mask matches the frame shape."""
        if mask.shape != frame_shape:
            raise ValueError(f"mask shape {mask.shape} does not match frame shape {frame_shape}")
        return obs_from_mask(mask, object_score)
