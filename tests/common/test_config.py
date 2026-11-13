from pathlib import Path

import pytest

from src.common.config import DATA_ROOT_ENV, ConfigError, DataConfig, load_config

CONFIG_FILES = sorted(Path("configs").rglob("*.yaml"))

VALID_DATA = """\
kind: data
data_root: ${SAM2FR_DATA_ROOT}
davis: {root: DAVIS, split: val, resolution: 480p, year: 2017}
sav: {root: SA-V, val_split: sav_val, test_split: sav_test}
output_root: outputs
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return path


def test_config_files_found() -> None:
    assert CONFIG_FILES


@pytest.mark.parametrize("path", CONFIG_FILES, ids=str)
def test_every_config_loads(path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert load_config(path) is not None


def test_data_paths_resolve_under_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = load_config("configs/data.yaml")
    assert isinstance(cfg, DataConfig)
    assert cfg.data_root == tmp_path
    assert cfg.resolve(cfg.davis.root) == tmp_path / "DAVIS"
    assert cfg.davis.year == 2017


def test_unknown_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    with pytest.raises(ConfigError, match="surprise"):
        load_config(write(tmp_path, VALID_DATA + "surprise: 1\n"))


def test_missing_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    with pytest.raises(ConfigError, match="output_root"):
        load_config(write(tmp_path, VALID_DATA.replace("output_root: outputs\n", "")))


def test_wrong_type(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    with pytest.raises(ConfigError, match="year"):
        load_config(write(tmp_path, VALID_DATA.replace("year: 2017", "year: abc")))


def test_missing_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(ConfigError, match=DATA_ROOT_ENV):
        load_config(write(tmp_path, VALID_DATA))


@pytest.mark.parametrize("value", ["", "  "])
def test_empty_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, value)
    with pytest.raises(ConfigError, match=DATA_ROOT_ENV):
        load_config(write(tmp_path, VALID_DATA))


def test_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mystery"):
        load_config(write(tmp_path, "kind: mystery\n"))
