from pathlib import Path

import pytest

from src.common.config import ConfigError
from src.recovery.config import RULE_NAMES, RecoveryRunConfig, load_recovery_config

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = sorted(
    p
    for p in (ROOT / "configs" / "recovery").glob("*.yaml")
    if p.name not in ("box_check.yaml", "recovery.yaml")
) + sorted((ROOT / "configs" / "oracle").glob("*.yaml"))


def test_shipped_configs_exist() -> None:
    assert len(SHIPPED) == 7


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_every_shipped_config_loads(path: Path) -> None:
    cfg = load_recovery_config(path)
    assert ("oracle" in cfg.method) == (path.parent.name == "oracle")


def test_real_system_config_is_not_oracle() -> None:
    cfg = load_recovery_config(ROOT / "configs" / "recovery" / "rules_csrt_tiny.yaml")
    assert (cfg.detector, cfg.box_source) == ("rules", "csrt")


def test_real_system_with_oracle_flags_raises() -> None:
    with pytest.raises(ConfigError):
        RecoveryRunConfig(method="fix_rules_csrt_tiny", detector="oracle", box_source="csrt")


def test_real_system_with_oracle_box_raises() -> None:
    with pytest.raises(ConfigError):
        RecoveryRunConfig(method="fix_rules_csrt_tiny", detector="rules", box_source="oracle")


def test_oracle_name_without_oracle_piece_raises() -> None:
    with pytest.raises(ConfigError):
        RecoveryRunConfig(method="fix_oracle_tiny", detector="rules", box_source="csrt")


@pytest.mark.parametrize(
    ("method", "detector", "box"),
    [("fix_x", "learned", "csrt"), ("fix_x", "rules", "gt"), ("Bad Name", "rules", "csrt")],
)
def test_unknown_values_raise(method: str, detector: str, box: str) -> None:
    with pytest.raises(ConfigError):
        RecoveryRunConfig(method=method, detector=detector, box_source=box)


def test_yaml_with_oracle_flags_in_real_method_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "kind: recovery_run\nmethod: fix_rules_csrt_tiny\ndetector: oracle\nbox_source: csrt\n"
    )
    with pytest.raises(ConfigError):
        load_recovery_config(path)


def test_continuous_config_sets_the_tracker_mode() -> None:
    cfg = load_recovery_config(ROOT / "configs" / "recovery" / "rules_csrt_continuous_tiny.yaml")
    assert cfg.tracker_mode == "continuous" and cfg.enabled_rules == RULE_NAMES


def test_plain_config_leaves_the_fixed_settings() -> None:
    cfg = load_recovery_config(ROOT / "configs" / "recovery" / "rules_csrt_tiny.yaml")
    assert cfg.tracker_mode == "" and cfg.disabled_rules == []


def test_name_must_say_continuous_iff_tracker_mode_is() -> None:
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_rules_csrt_tiny", "rules", "csrt", tracker_mode="continuous")
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_continuous_rules_csrt_tiny", "rules", "csrt")
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_continuous_x_tiny", "rules", "csrt", tracker_mode="sideways")


def test_name_must_say_which_rule_is_off() -> None:
    cfg = RecoveryRunConfig("fix_no_low_score_tiny", "rules", "csrt", disabled_rules=["low_score"])
    assert "low_score" not in cfg.enabled_rules and len(cfg.enabled_rules) == 3
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_tiny", "rules", "csrt", disabled_rules=["low_score"])
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_no_low_score_tiny", "rules", "csrt")
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_no_bogus_tiny", "rules", "csrt", disabled_rules=["bogus"])
