from pathlib import Path

import pytest

from src.common import layout

ROOT = Path("/out")


def test_paths() -> None:
    assert layout.mask_path(ROOT, "sam2_tiny", "davis_val", "bear", "00012") == Path(
        "/out/sam2_tiny/davis_val/masks/bear/00012.png"
    )
    assert layout.masks_dir(ROOT, "sam2_tiny", "sav_val", "sav_000001") == Path(
        "/out/sam2_tiny/sav_val/masks/sav_000001"
    )
    assert layout.signals_path(ROOT, "sam2_large", "sav_test", "v") == Path(
        "/out/sam2_large/sav_test/signals/v.csv"
    )
    assert layout.events_path(ROOT, "fix_tiny", "davis_val", "v") == Path(
        "/out/fix_tiny/davis_val/events/v.csv"
    )


@pytest.mark.parametrize("name", ["", "Tiny", "a-b", "a/b", "a b", "../x"])
def test_invalid_method(name: str) -> None:
    with pytest.raises(ValueError):
        layout.validate_method(name)


def test_invalid_split() -> None:
    with pytest.raises(ValueError):
        layout.signals_path(ROOT, "m", "davis_train", "v")


@pytest.mark.parametrize("video", ["", "a/b", "a\\b", "..", "."])
def test_invalid_video(video: str) -> None:
    with pytest.raises(ValueError):
        layout.signals_path(ROOT, "m", "davis_val", video)
    with pytest.raises(ValueError):
        layout.mask_path(ROOT, "m", "davis_val", video, "00000")


def test_invalid_frame_stem() -> None:
    with pytest.raises(ValueError):
        layout.mask_path(ROOT, "m", "davis_val", "v", "../00000")


def test_oracle_detection() -> None:
    assert layout.is_oracle_method("oracle")
    assert layout.is_oracle_method("fix_oracle_tiny")
    assert layout.is_oracle_method("sam2_tiny_oracle")
    assert not layout.is_oracle_method("sam2_tiny")
    assert layout.is_oracle_method("oracles")
    assert layout.is_oracle_method("myoracle")
    assert layout.is_oracle_method("oracle2")
