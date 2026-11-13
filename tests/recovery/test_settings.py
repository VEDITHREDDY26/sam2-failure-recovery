from pathlib import Path

import pytest

from src.common.config import ConfigError
from src.recovery.settings import CONFIG_PATH, RecoverySettings, load_recovery_settings


def test_shipped_settings() -> None:
    cfg = load_recovery_settings(CONFIG_PATH)
    assert (cfg.cooldown_frames, cfg.tracker_mode) == (12, "reinit")


@pytest.mark.parametrize(("cooldown", "mode"), [(-1, "reinit"), (12, "sometimes")])
def test_bad_values_raise(cooldown: int, mode: str) -> None:
    with pytest.raises(ConfigError):
        RecoverySettings(cooldown, mode)


def test_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text("kind: recovery\ncooldown_frames: 5\ntracker_mode: reinit\nextra: 1\n")
    with pytest.raises(ConfigError):
        load_recovery_settings(path)
