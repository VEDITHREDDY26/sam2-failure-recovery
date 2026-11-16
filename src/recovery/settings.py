"""Recovery settings: cooldown after a re-prompt and how the CSRT tracker follows SAM 2."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from src.common.config import ConfigError, load_config, register

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "recovery" / "recovery.yaml"
TRACKER_MODES = ("reinit", "continuous")


@register("recovery")
@dataclasses.dataclass(frozen=True)
class RecoverySettings:
    cooldown_frames: int
    tracker_mode: str

    def __post_init__(self) -> None:
        if self.cooldown_frames < 0:
            raise ConfigError("cooldown_frames must be >= 0")
        if self.tracker_mode not in TRACKER_MODES:
            raise ConfigError(f"tracker_mode must be one of {TRACKER_MODES}")


def load_recovery_settings(path: Path = CONFIG_PATH) -> RecoverySettings:
    cfg = load_config(path)
    if not isinstance(cfg, RecoverySettings):
        raise ConfigError(f"{path}: not a recovery settings config")
    return cfg
