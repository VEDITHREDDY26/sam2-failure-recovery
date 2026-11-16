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
    def __init__(self, thresholds: RuleThresholds, enabled: Iterable[str] = ALL_RULES) -> None:
        self.thresholds = thresholds
        self.enabled = tuple(enabled)
        RuleDetector(thresholds, self.enabled)
        self._detectors: dict[int, RuleDetector] = {}

    @classmethod
    def from_config(
        cls, model: str, path: str | Path = DEFAULT_RULES_PATH, enabled: Iterable[str] = ALL_RULES
    ) -> OnlineRuleDetector:
        return cls(load_rules_config(model, path), enabled)

    def reset(self) -> None:
        self._detectors.clear()

    def update(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> Decision:
        obs = self._obs(mask, object_score, frame_shape)
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
        self._detector(obj_id).update(self._obs(mask, object_score, frame_shape))

    def _detector(self, obj_id: int) -> RuleDetector:
        if obj_id not in self._detectors:
            self._detectors[obj_id] = RuleDetector(self.thresholds, self.enabled)
        return self._detectors[obj_id]

    @staticmethod
    def _obs(mask: Mask, object_score: float, frame_shape: tuple[int, int]) -> FrameObs:
        if mask.shape != frame_shape:
            raise ValueError(f"mask shape {mask.shape} does not match frame shape {frame_shape}")
        return obs_from_mask(mask, object_score)
