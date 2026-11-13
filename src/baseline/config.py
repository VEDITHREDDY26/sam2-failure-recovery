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
    grabcut_iterations: int
    roi_margin_frac: float
    min_box_side: int

    def __post_init__(self) -> None:
        if self.grabcut_iterations < 1:
            raise ConfigError("grabcut_iterations must be at least 1")
        if self.roi_margin_frac < 0:
            raise ConfigError("roi_margin_frac must not be negative")
        if self.min_box_side < 1:
            raise ConfigError("min_box_side must be at least 1")


def load_baseline_config(path: Path = CONFIG_PATH) -> BaselineConfig:
    cfg = load_config(path)
    if not isinstance(cfg, BaselineConfig):
        raise ConfigError(f"{path}: not a baseline config")
    return cfg
