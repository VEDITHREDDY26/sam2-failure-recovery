"""Settings of the CSRT + GrabCut baseline, fixed up front and never tuned on evaluation data."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from src.common.config import ConfigError, load_config, register

METHOD = "baseline_csrt_grabcut"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "baseline" / "csrt_grabcut.yaml"


@register("baseline")
@dataclasses.dataclass(frozen=True)
class BaselineConfig:
    """Frozen settings of the CSRT + GrabCut baseline, loaded from the baseline YAML file.

    Attributes:
        grabcut_iterations: Number of GrabCut refinement iterations per frame (at least 1).
        roi_margin_frac: Fraction of the box width/height added on each side to form the GrabCut
            crop (0 or more).
        min_box_side: Smallest allowed box side in pixels; smaller prompt boxes are grown to it and
            smaller tracker boxes are treated as unusable (at least 1).
    """

    grabcut_iterations: int
    roi_margin_frac: float
    min_box_side: int

    def __post_init__(self) -> None:
        """Reject out-of-range settings with a ConfigError."""
        # Check the ranges when the config is built, so a bad YAML value fails before any run.
        if self.grabcut_iterations < 1:
            raise ConfigError("grabcut_iterations must be at least 1")
        if self.roi_margin_frac < 0:
            raise ConfigError("roi_margin_frac must not be negative")
        if self.min_box_side < 1:
            raise ConfigError("min_box_side must be at least 1")


def load_baseline_config(path: Path = CONFIG_PATH) -> BaselineConfig:
    """Load and type-check the baseline settings file.

    Args:
        path: YAML file to read; defaults to configs/baseline/csrt_grabcut.yaml in the repo.

    Returns:
        The parsed BaselineConfig.

    Raises:
        ConfigError: If the file does not describe a baseline config or a value is out of range.
    """
    cfg = load_config(path)
    if not isinstance(cfg, BaselineConfig):
        raise ConfigError(f"{path}: not a baseline config")
    return cfg
