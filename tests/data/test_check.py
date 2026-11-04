from __future__ import annotations

from pathlib import Path

import pytest

from src.data.check import EXPECTED_VIDEOS, main
from tests.data.helpers import build_davis_root

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data.yaml"


def _args(*extra: str) -> list[str]:
    return ["--dataset", "davis", "--split", "val", "--config", str(CONFIG), *extra]


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    build_davis_root(tmp_path)
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(tmp_path))
    return tmp_path


def test_expected_counts_are_dataset_facts() -> None:
    assert EXPECTED_VIDEOS == {("davis", "val"): 30, ("sav", "val"): 155, ("sav", "test"): 150}


def test_report_counts(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_args("--expected-videos", "2")) == 0
    out = capsys.readouterr().out
    assert "videos: 2 (expected 2)" in out
    assert "objects: 4" in out
    assert "frames: 40" in out
    assert "objects first visible after frame 0: 0" in out
    assert "problems: 0" in out


def test_full_mode(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_args("--full", "--expected-videos", "2")) == 0
    assert "problems: 0" in capsys.readouterr().out


def test_count_mismatch_exits_1(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_args()) == 1
    out = capsys.readouterr().out
    assert "videos: 2 (expected 30)" in out
    assert "FAIL: expected 30 videos, found 2" in out


def test_broken_video_is_reported(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (data_root / "DAVIS" / "Annotations" / "480p" / "bear" / "00003.png").unlink()
    assert main(_args("--expected-videos", "2")) == 1
    assert "bear: " in capsys.readouterr().out


def test_sav_not_built_yet() -> None:
    with pytest.raises(NotImplementedError):
        main(["--dataset", "sav", "--split", "val"])
