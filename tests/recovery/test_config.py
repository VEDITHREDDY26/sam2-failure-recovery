from pathlib import Path

import pytest

from src.common.config import ConfigError
from src.recovery.config import (
    RULE_NAMES,
    RecoveryRunConfig,
    VerifiedRunConfig,
    load_recovery_config,
)
from src.recovery.gate import load_box_gate

EXTRA = 'tracker_mode: ""\ndisabled_rules: []\n'

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = sorted(
    p
    for p in (ROOT / "configs" / "recovery").glob("*.yaml")
    if p.name not in ("box_check.yaml", "recovery.yaml", "vit.yaml")
    and not p.name.startswith(("verified_", "box_gate_"))
) + sorted((ROOT / "configs" / "oracle").glob("*.yaml"))
GATES = sorted((ROOT / "configs" / "recovery").glob("box_gate_*.yaml"))


def test_shipped_configs_exist() -> None:
    assert len(SHIPPED) == 15
    assert [p.name for p in GATES] == ["box_gate_large.yaml", "box_gate_tiny.yaml"]


@pytest.mark.parametrize("path", GATES, ids=lambda p: p.name)
def test_every_shipped_gate_loads(path: Path) -> None:
    gate = load_box_gate(path).gate()
    assert gate.s_min is not None or gate.a_max is not None


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_every_shipped_config_loads(path: Path) -> None:
    cfg = load_recovery_config(path)
    assert ("oracle" in cfg.method) == (path.parent.name == "oracle")


def test_real_system_config_is_not_oracle() -> None:
    cfg = load_recovery_config(ROOT / "configs" / "recovery" / "rules_csrt_tiny.yaml")
    assert (cfg.detector, cfg.box_source) == ("rules", "csrt")


def test_real_system_with_oracle_flags_raises() -> None:
    with pytest.raises(ConfigError):
        RecoveryRunConfig(
            method="fix_rules_csrt_tiny",
            detector="oracle",
            box_source="csrt",
            tracker_mode="",
            disabled_rules=[],
        )


def test_real_system_with_oracle_box_raises() -> None:
    with pytest.raises(ConfigError):
        RecoveryRunConfig(
            method="fix_rules_csrt_tiny",
            detector="rules",
            box_source="oracle",
            tracker_mode="",
            disabled_rules=[],
        )


def test_oracle_name_without_oracle_piece_raises() -> None:
    with pytest.raises(ConfigError):
        RecoveryRunConfig(
            method="fix_oracle_tiny",
            detector="rules",
            box_source="csrt",
            tracker_mode="",
            disabled_rules=[],
        )


@pytest.mark.parametrize(
    ("method", "detector", "box"),
    [("fix_x", "learned", "csrt"), ("fix_x", "rules", "gt"), ("Bad Name", "rules", "csrt")],
)
def test_unknown_values_raise(method: str, detector: str, box: str) -> None:
    with pytest.raises(ConfigError):
        RecoveryRunConfig(
            method=method, detector=detector, box_source=box, tracker_mode="", disabled_rules=[]
        )


def test_yaml_with_oracle_flags_in_real_method_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "kind: recovery_run\nmethod: fix_rules_csrt_tiny\ndetector: oracle\nbox_source: csrt\n"
        + EXTRA
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
        RecoveryRunConfig("fix_rules_csrt_tiny", "rules", "csrt", "continuous", [])
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_continuous_rules_csrt_tiny", "rules", "csrt", "", [])
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_continuous_x_tiny", "rules", "csrt", "sideways", [])


def test_name_must_say_which_rule_is_off() -> None:
    cfg = RecoveryRunConfig("fix_no_low_score_tiny", "rules", "csrt", "", ["low_score"])
    assert "low_score" not in cfg.enabled_rules and len(cfg.enabled_rules) == 3
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_tiny", "rules", "csrt", "", ["low_score"])
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_no_low_score_tiny", "rules", "csrt", "", [])
    with pytest.raises(ConfigError):
        RecoveryRunConfig("fix_no_bogus_tiny", "rules", "csrt", "", ["bogus"])


def _learned(method: str, detector: str) -> RecoveryRunConfig:
    return RecoveryRunConfig(
        method=method, detector=detector, box_source="csrt", tracker_mode="", disabled_rules=[]
    )


def test_learned_in_the_name_iff_the_detector_is_learned() -> None:
    assert _learned("fix_learned_csrt_tiny", "learned").detector == "learned"
    with pytest.raises(ConfigError):
        _learned("fix_rules_csrt_tiny", "learned")
    with pytest.raises(ConfigError):
        _learned("fix_learned_csrt_tiny", "rules")


VERIFIED = sorted((ROOT / "configs" / "recovery").glob("verified_*.yaml"))


def test_verified_configs_exist_for_both_models() -> None:
    assert [p.name for p in VERIFIED] == [
        "verified_rules_csrt_large.yaml",
        "verified_rules_csrt_tiny.yaml",
    ]


@pytest.mark.parametrize("path", VERIFIED, ids=lambda p: p.name)
def test_verified_config_loads_and_points_to_the_verify_file(path: Path) -> None:
    cfg = load_recovery_config(path)
    assert isinstance(cfg, VerifiedRunConfig)
    assert cfg.method == f"fix_verified_{path.stem.removeprefix('verified_')}"
    assert cfg.verify_path() == ROOT / "configs" / "verify" / "verify.yaml"
    assert (cfg.detector, cfg.box_source) == ("rules", "csrt")


def test_verified_name_and_kind_go_together() -> None:
    plain = dict(detector="rules", box_source="csrt", tracker_mode="", disabled_rules=[])
    with pytest.raises(ConfigError):
        RecoveryRunConfig(method="fix_verified_rules_csrt_tiny", **plain)
    with pytest.raises(ConfigError):
        VerifiedRunConfig(method="fix_rules_csrt_tiny", verify_config="x.yaml", **plain)
    with pytest.raises(ConfigError):
        VerifiedRunConfig(method="fix_verified_rules_csrt_tiny", verify_config="", **plain)


def test_verified_run_cannot_use_oracle_pieces() -> None:
    with pytest.raises(ConfigError):
        VerifiedRunConfig(
            method="fix_verified_oracle_tiny",
            detector="oracle",
            box_source="csrt",
            tracker_mode="",
            disabled_rules=[],
            verify_config="x.yaml",
        )
