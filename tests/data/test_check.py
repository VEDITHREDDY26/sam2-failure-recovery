from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.check import EXPECTED_VIDEOS, main
from tests.data.helpers import build_davis_root, build_sav_root, write_binary_png

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


def _sav_args(split: str, *extra: str) -> list[str]:
    return ["--dataset", "sav", "--split", split, "--config", str(CONFIG), *extra]


def test_sav_report_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_sav_root(tmp_path, "sav_test", ("vid_a", "vid_b"))
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(tmp_path))
    assert main(_sav_args("test", "--expected-videos", "2")) == 0
    out = capsys.readouterr().out
    assert "videos: 2 (expected 2)" in out
    assert "objects: 6" in out
    assert "WARN: expected 278 masklets, found 6" in out
    assert "frames: 40" in out
    assert "annotated (object, frame) pairs: 30" in out
    assert "objects first visible after frame 0: 2" in out
    assert "problems: 0" in out
    assert main(_sav_args("test", "--full", "--expected-videos", "2")) == 0


def test_sav_video_count_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_sav_root(tmp_path, "sav_val")
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(tmp_path))
    assert main(_sav_args("val")) == 1
    out = capsys.readouterr().out
    assert "videos: 1 (expected 155)" in out
    assert "FAIL: expected 155 videos, found 1" in out


def test_sav_size_and_stem_problems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = build_sav_root(tmp_path, "sav_val", ("vid_a", "vid_b"))
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(tmp_path))
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(
        root / "Annotations_6fps" / "vid_a" / "001" / "00004.png"
    )
    write_binary_png(root / "Annotations_6fps" / "vid_b" / "000" / "00077.png", None)
    assert main(_sav_args("val", "--full", "--expected-videos", "2")) == 1
    out = capsys.readouterr().out
    assert "vid_a: object 2 frame 4 annotation size mismatch" in out
    assert "vid_b: " in out and "no matching frame" in out
