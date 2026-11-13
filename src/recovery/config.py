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
TRACKER_MODES = ("reinit", "continuous")
RULE_NAMES = ("low_score", "area_change", "centre_jump", "disappearance")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "recovery"


@register("recovery_run")
@dataclasses.dataclass(frozen=True)
class RecoveryRunConfig:
    method: str
    detector: str
    box_source: str
    tracker_mode: str = ""
    disabled_rules: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            validate_method(self.method)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if self.detector not in DETECTORS:
            raise ConfigError(f"detector must be one of {DETECTORS}, got '{self.detector}'")
        if self.box_source not in BOX_SOURCES:
            raise ConfigError(f"box_source must be one of {BOX_SOURCES}, got '{self.box_source}'")
        self._check_variant()
        uses_oracle = "oracle" in (self.detector, self.box_source)
        if uses_oracle and not is_oracle_method(self.method):
            raise ConfigError(f"'{self.method}' uses oracle pieces but its name has no 'oracle'")
        if is_oracle_method(self.method) and not uses_oracle:
            raise ConfigError(f"'{self.method}' says oracle but uses no oracle piece")

    def _check_variant(self) -> None:
        if self.tracker_mode and self.tracker_mode not in TRACKER_MODES:
            raise ConfigError(
                f"tracker_mode must be empty (fixed settings file) or one of {TRACKER_MODES}"
            )
        unknown = [r for r in self.disabled_rules if r not in RULE_NAMES]
        if unknown or len(set(self.disabled_rules)) != len(self.disabled_rules):
            raise ConfigError(f"disabled_rules must be distinct names from {RULE_NAMES}")
        if ("continuous" in self.method) != (self.tracker_mode == "continuous"):
            raise ConfigError(f"'{self.method}': 'continuous' in the name iff tracker_mode is")
        for rule in RULE_NAMES:
            if (f"no_{rule}" in self.method) != (rule in self.disabled_rules):
                raise ConfigError(f"'{self.method}': 'no_{rule}' in the name iff the rule is off")

    @property
    def enabled_rules(self) -> tuple[str, ...]:
        return tuple(r for r in RULE_NAMES if r not in self.disabled_rules)


def load_recovery_config(path: Path) -> RecoveryRunConfig:
    cfg = load_config(path)
    if not isinstance(cfg, RecoveryRunConfig):
        raise ConfigError(f"{path}: not a recovery run config")
    return cfg
