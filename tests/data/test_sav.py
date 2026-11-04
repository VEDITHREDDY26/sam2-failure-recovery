from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.boxes import Box
from src.data.sav import list_sav_videos, load_sav_video
from src.data.video import VideoSample
from tests.data.helpers import (
    SAV_ANNOTATED,
    SAV_NAME,
    build_sav_root,
    make_config,
    write_binary_png,
)


@pytest.fixture
def video(tmp_path: Path) -> VideoSample:
    build_sav_root(tmp_path)
    return load_sav_video(make_config(tmp_path), "val", SAV_NAME)


def test_list_and_ids(tmp_path: Path, video: VideoSample) -> None:
    assert list_sav_videos(make_config(tmp_path), "val") == (SAV_NAME,)
    assert video.object_ids == (1, 2, 3)
    assert video.object_names == {1: "000", 2: "001", 3: "002"}
    assert video.dataset == "sav"
    assert video.split == "sav_val"
    assert video.num_frames == 20


def test_annotated_frames_are_every_fourth(video: VideoSample) -> None:
    for obj_id in (1, 2, 3):
        assert video.annotated_frames(obj_id) == (0, 4, 8, 12, 16) == SAV_ANNOTATED


@pytest.mark.parametrize("frame", [1, 2, 3, 15])
def test_unannotated_frames_are_none_not_empty(video: VideoSample, frame: int) -> None:
    for obj_id in (1, 2, 3):
        assert video.gt_mask(obj_id, frame) is None
        assert not video.is_annotated(obj_id, frame)
    assert video.gt_masks(frame) == {}


def test_annotated_empty_frame_is_all_false(video: VideoSample) -> None:
    mask = video.gt_mask(1, 16)
    assert mask is not None
    assert mask.shape == (64, 96)
    assert not mask.any()
    assert set(video.gt_masks(16)) == {1, 2, 3}


def test_annotated_mask_values(video: VideoSample) -> None:
    mask = video.gt_mask(1, 0)
    assert mask is not None
    assert mask.dtype == bool
    assert int(mask.sum()) == 12 * 12
    assert video.first_box(1).box == Box(3, 20, 15, 32)


def test_late_object_first_box(video: VideoSample) -> None:
    first = video.first_box(3)
    assert first.frame_index == 8
    assert first.box == Box(10, 46, 20, 56)
    early = video.gt_mask(3, 0)
    assert early is not None and not early.any()


def test_overlap_is_kept_in_both_objects(video: VideoSample) -> None:
    masks = video.gt_masks(8)
    overlap = np.zeros((64, 96), dtype=bool)
    overlap[46:52, 10:16] = True
    assert overlap.sum() == 36
    assert masks[2][overlap].all()
    assert masks[3][overlap].all()


def test_missing_list_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="video list"):
        list_sav_videos(make_config(tmp_path), "val")


def test_unknown_video(tmp_path: Path) -> None:
    build_sav_root(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        load_sav_video(make_config(tmp_path), "val", "nope")


def test_annotation_without_frame(tmp_path: Path) -> None:
    root = build_sav_root(tmp_path)
    write_binary_png(root / "Annotations_6fps" / SAV_NAME / "000" / "00099.png", None)
    with pytest.raises(ValueError, match="no matching frame"):
        load_sav_video(make_config(tmp_path), "val", SAV_NAME)


def test_non_numeric_object_folder(tmp_path: Path) -> None:
    root = build_sav_root(tmp_path)
    write_binary_png(root / "Annotations_6fps" / SAV_NAME / "cat" / "00000.png", None)
    with pytest.raises(ValueError, match="not numeric"):
        load_sav_video(make_config(tmp_path), "val", SAV_NAME)


def test_object_id_above_255(tmp_path: Path) -> None:
    root = build_sav_root(tmp_path)
    write_binary_png(root / "Annotations_6fps" / SAV_NAME / "255" / "00000.png", None)
    with pytest.raises(ValueError, match="above 255"):
        load_sav_video(make_config(tmp_path), "val", SAV_NAME)


def test_test_split_uses_test_folder(tmp_path: Path) -> None:
    build_sav_root(tmp_path, split="sav_test")
    cfg = make_config(tmp_path)
    assert load_sav_video(cfg, "test", SAV_NAME).split == "sav_test"
    with pytest.raises(ValueError, match="invalid SA-V split"):
        list_sav_videos(cfg, "train")
