from pathlib import Path

import pytest

from src.baseline.config import CONFIG_PATH, BaselineConfig, load_baseline_config
from src.common.config import ConfigError, load_config


def test_shipped_config_values() -> None:
    cfg = load_baseline_config()
    assert cfg == BaselineConfig(grabcut_iterations=5, roi_margin_frac=0.25, min_box_side=4)


def test_kind_is_registered_for_generic_loader() -> None:
    assert isinstance(load_config(CONFIG_PATH), BaselineConfig)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(CONFIG_PATH.read_text() + "tuned: 1\n")
    with pytest.raises(ConfigError, match="tuned"):
        load_baseline_config(path)


def test_invalid_value_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(CONFIG_PATH.read_text().replace("iterations: 5", "iterations: 0"))
    with pytest.raises(ConfigError, match="grabcut_iterations"):
        load_baseline_config(path)
