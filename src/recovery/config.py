"""Run-config schema for recovery runs.

Holds strings only, so real-system code never imports the oracle package. Oracle pieces and
oracle method names must go together; the oracle entry script injects the oracle code.
Recovery settings are added later; this schema only says which detector and box source run.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from src.common.config import ConfigError, load_config, register
from src.common.layout import is_oracle_method, validate_method

DETECTORS = ("rules", "oracle")
BOX_SOURCES = ("csrt", "oracle")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "recovery"


@register("recovery_run")
@dataclasses.dataclass(frozen=True)
class RecoveryRunConfig:
    method: str
    detector: str
    box_source: str

    def __post_init__(self) -> None:
        try:
            validate_method(self.method)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if self.detector not in DETECTORS:
            raise ConfigError(f"detector must be one of {DETECTORS}, got '{self.detector}'")
        if self.box_source not in BOX_SOURCES:
            raise ConfigError(f"box_source must be one of {BOX_SOURCES}, got '{self.box_source}'")
        uses_oracle = "oracle" in (self.detector, self.box_source)
        if uses_oracle and not is_oracle_method(self.method):
            raise ConfigError(f"'{self.method}' uses oracle pieces but its name has no 'oracle'")
        if is_oracle_method(self.method) and not uses_oracle:
            raise ConfigError(f"'{self.method}' says oracle but uses no oracle piece")


def load_recovery_config(path: Path) -> RecoveryRunConfig:
    cfg = load_config(path)
    if not isinstance(cfg, RecoveryRunConfig):
        raise ConfigError(f"{path}: not a recovery run config")
    return cfg
