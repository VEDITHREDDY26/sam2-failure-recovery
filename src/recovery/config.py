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

DETECTORS = ("rules", "learned", "oracle")
VIT_SOURCES = ("vit_reinit", "vit_continuous", "vit_anchored")
BOX_SOURCES = ("csrt", "oracle", "motion", *VIT_SOURCES)
TRACKER_MODES = ("reinit", "continuous")
RULE_NAMES = ("low_score", "area_change", "centre_jump", "disappearance")
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "recovery"


@register("recovery_run")
@dataclasses.dataclass(frozen=True)
class RecoveryRunConfig:
    """Schema of a recovery run: which detector and box source to run and under what method name.

    Holds strings only, so real-system code never needs the oracle package.

    Attributes:
        method: Output folder name of the run; lowercase letters, digits and underscores.
        detector: One of DETECTORS ("rules", "learned", "oracle").
        box_source: One of BOX_SOURCES ("csrt", "oracle", "motion" or a ViT source).
        tracker_mode: "reinit" or "continuous" for the CSRT source; empty for every other source
            or when the fixed settings file decides.
        disabled_rules: Detector rules switched off (ablation); distinct names from RULE_NAMES.
    """

    method: str
    detector: str
    box_source: str
    tracker_mode: str
    disabled_rules: list[str]

    def __post_init__(self) -> None:
        """Validate the config.

        Oracle pieces and oracle method names must go together, so a real-system run can never
        use ground truth by accident.

        Raises:
            ConfigError: On an invalid method name, detector or box source, a name/setting mismatch
                (see _check_variant) or a missing/extra "oracle" in the method name.
        """
        try:
            validate_method(self.method)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if self.detector not in DETECTORS:
            raise ConfigError(f"detector must be one of {DETECTORS}, got '{self.detector}'")
        if self.box_source not in BOX_SOURCES:
            raise ConfigError(f"box_source must be one of {BOX_SOURCES}, got '{self.box_source}'")
        self._check_variant()
        # Oracle pieces and oracle method names must go together, in both directions.
        uses_oracle = "oracle" in (self.detector, self.box_source)
        if uses_oracle and not is_oracle_method(self.method):
            raise ConfigError(f"'{self.method}' uses oracle pieces but its name has no 'oracle'")
        if is_oracle_method(self.method) and not uses_oracle:
            raise ConfigError(f"'{self.method}' says oracle but uses no oracle piece")

    def _check_variant(self) -> None:
        """Check that the method name and the settings describe the same variant.

        Each marker in a method name ("learned", "continuous", "vit", "motion", "gated", "verified",
        "no_<rule>") must appear exactly when the matching setting is chosen, so a result folder
        name always says what was run.

        Raises:
            ConfigError: On any mismatch, an unknown or repeated disabled rule, or a tracker mode
                that is not allowed for the box source.
        """
        if self.box_source != "csrt" and self.tracker_mode:
            raise ConfigError(f"'{self.method}': tracker_mode applies to the csrt box source only")
        if self.tracker_mode and self.tracker_mode not in TRACKER_MODES:
            raise ConfigError(
                f"tracker_mode must be empty (fixed settings file) or one of {TRACKER_MODES}"
            )
        # Disabled rules must be known names with no repeats.
        unknown = [r for r in self.disabled_rules if r not in RULE_NAMES]
        if unknown or len(set(self.disabled_rules)) != len(self.disabled_rules):
            raise ConfigError(f"disabled_rules must be distinct names from {RULE_NAMES}")
        if ("learned" in self.method) != (self.detector == "learned"):
            raise ConfigError(f"'{self.method}': 'learned' in the name iff detector is learned")
        # Method-name markers must agree with the settings that they describe.
        continuous = self.tracker_mode == "continuous" or self.box_source == "vit_continuous"
        if ("continuous" in self.method) != continuous:
            raise ConfigError(f"'{self.method}': 'continuous' in the name iff the source is")
        for source in ("vit_reinit", "vit_anchored", "motion"):
            if (source in self.method) != (self.box_source == source):
                raise ConfigError(f"'{self.method}': '{source}' in the name iff box_source is")
        if ("vit" in self.method) != (self.box_source in VIT_SOURCES):
            raise ConfigError(f"'{self.method}': 'vit' in the name iff box_source is a ViT one")
        if ("gated" in self.method) != isinstance(self, GatedRunConfig):
            raise ConfigError(f"'{self.method}': 'gated' in the name iff a gated config")
        if ("verified" in self.method) != isinstance(self, VerifiedRunConfig):
            raise ConfigError(f"'{self.method}': 'verified' in the name iff a verified config")
        for rule in RULE_NAMES:
            if (f"no_{rule}" in self.method) != (rule in self.disabled_rules):
                raise ConfigError(f"'{self.method}': 'no_{rule}' in the name iff the rule is off")

    @property
    def enabled_rules(self) -> tuple[str, ...]:
        """Names of the detector rules that are on, in RULE_NAMES order."""
        return tuple(r for r in RULE_NAMES if r not in self.disabled_rules)


@register("verified_recovery_run")
@dataclasses.dataclass(frozen=True)
class VerifiedRunConfig(RecoveryRunConfig):
    """A real-system run whose re-prompts are verified with the frozen thresholds."""

    verify_config: str

    def __post_init__(self) -> None:
        """Validate the base config and require the rule detector with the CSRT source.

        Raises:
            ConfigError: If the detector/source pair is not rules + csrt or verify_config is empty.
        """
        super().__post_init__()
        if (self.detector, self.box_source) != ("rules", "csrt"):
            raise ConfigError(f"'{self.method}': verified recovery is rules + csrt only")
        if not self.verify_config:
            raise ConfigError(f"'{self.method}': verify_config must name the thresholds file")

    def verify_path(self) -> Path:
        """Absolute path of the thresholds file."""
        return ROOT / self.verify_config


@register("gated_recovery_run")
@dataclasses.dataclass(frozen=True)
class GatedRunConfig(RecoveryRunConfig):
    """A real-system run whose proposed boxes pass the gate before the box check."""

    gate_config: str

    def __post_init__(self) -> None:
        """Validate the base config and require the rule detector without the oracle.

        Raises:
            ConfigError: If the detector is not rules, the box source is the oracle, or gate_config
                is empty.
        """
        super().__post_init__()
        if self.detector != "rules" or self.box_source == "oracle":
            raise ConfigError(f"'{self.method}': a gated run uses the rule detector, no oracle")
        if not self.gate_config:
            raise ConfigError(f"'{self.method}': gate_config must name the gate file")

    def gate_path(self) -> Path:
        """Absolute path of the gate file."""
        return ROOT / self.gate_config


def load_recovery_config(path: Path) -> RecoveryRunConfig:
    """Load and validate a recovery run config.

    Args:
        path: YAML file of any of the three recovery run kinds.

    Returns:
        The matching config object.

    Raises:
        ConfigError: If the file is invalid or is not a recovery run config.
    """
    cfg = load_config(path)
    if not isinstance(cfg, RecoveryRunConfig):
        raise ConfigError(f"{path}: not a recovery run config")
    return cfg
