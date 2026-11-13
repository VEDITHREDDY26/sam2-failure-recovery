from pathlib import Path

import pytest

from src.common.config import ConfigError
from src.recovery.config import RecoveryRunConfig, load_recovery_config

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = sorted((ROOT / "configs" / "recovery").glob("*.yaml")) + sorted(
    (ROOT / "configs" / "oracle").glob("*.yaml")
)


def test_shipped_configs_exist() -> None:
    assert len(SHIPPED) == 4


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
