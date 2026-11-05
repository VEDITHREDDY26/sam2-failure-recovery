from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.boxes import Box
from src.data.davis import list_davis_videos, load_davis_video
from src.data.video import NeverVisibleError, PromptBox, VideoSample
from tests.data.helpers import VIDEO_NAMES, build_davis_root, make_config

HEIGHT, WIDTH = 64, 96


def _rect(box: Box) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[box.y0 : box.y1, box.x0 : box.x1] = True
    return mask


OBJECT2 = _rect(Box(8, 44, 16, 52))


@pytest.fixture
def video(tmp_path: Path) -> VideoSample:
    build_davis_root(tmp_path)
    return load_davis_video(make_config(tmp_path), "zebra")


def test_video_list_keeps_file_order(tmp_path: Path) -> None:
    build_davis_root(tmp_path)
    assert list_davis_videos(make_config(tmp_path)) == VIDEO_NAMES == ("zebra", "bear")


def test_basic_fields(video: VideoSample) -> None:
    assert (video.dataset, video.split, video.name) == ("davis", "davis_val", "zebra")
    assert video.num_frames == 20
    assert video.object_ids == (1, 2)
    assert video.object_names == {1: "1", 2: "2"}
    assert video.frame_path(3).name == "00003.jpg"
    assert video.frame_path(3).is_file()


def test_frame_order_is_numeric_not_lexicographic(tmp_path: Path) -> None:
    videos = build_davis_root(tmp_path, ("plain",))
    for folder, suffix in ((videos[0].frames_dir, ".jpg"), (videos[0].annotations_dir, ".png")):
        for path in list(folder.iterdir()):
            path.rename(folder / f"{int(path.stem)}{suffix}")
    loaded = load_davis_video(make_config(tmp_path), "plain")
    assert loaded.frame_stems[:3] == ("0", "1", "2")
    assert loaded.frame_stems[-1] == "19"
    assert loaded.frame_stems != tuple(sorted(loaded.frame_stems))
    assert loaded.frame_path(10).name == "10.jpg"
    assert loaded.first_box(1) == PromptBox(0, Box(3, 20, 15, 32))


def test_gt_mask_frame_zero(video: VideoSample) -> None:
    assert np.array_equal(video.gt_mask(1, 0), _rect(Box(3, 20, 15, 32)))
    assert np.array_equal(video.gt_mask(2, 0), OBJECT2)


def test_gt_mask_partly_occluded_frame(video: VideoSample) -> None:
    assert np.array_equal(video.gt_mask(1, 5), _rect(Box(18, 20, 27, 32)))


def test_gt_mask_vanished_is_empty_not_none(video: VideoSample) -> None:
    for frame in (9, 15):
        mask = video.gt_mask(1, frame)
        assert mask is not None
        assert mask.shape == (HEIGHT, WIDTH)
        assert not mask.any()
        assert video.is_annotated(1, frame)
    assert np.array_equal(video.gt_mask(2, 15), OBJECT2)


def test_gt_masks_per_frame(video: VideoSample) -> None:
    masks = video.gt_masks(15)
    assert sorted(masks) == [1, 2]
    assert not masks[1].any()
    assert np.array_equal(masks[2], OBJECT2)


def test_annotated_frames_cover_every_frame(video: VideoSample) -> None:
    assert video.annotated_frames(1) == tuple(range(20))


def test_first_boxes_are_tight_and_exclusive(video: VideoSample) -> None:
    assert video.first_box(1) == PromptBox(0, Box(3, 20, 15, 32))
    assert video.first_box(2) == PromptBox(0, Box(8, 44, 16, 52))


def test_never_visible_object_raises(video: VideoSample) -> None:
    class Blank:
        def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
            return (0, 4)

        def read_mask(self, obj_id: int, frame_index: int) -> np.ndarray:
            return np.zeros((HEIGHT, WIDTH), dtype=bool)

    blank = VideoSample(
        "sav", "sav_val", "x", video.frames_dir, video.frame_stems, (3,), {3: "a"}, Blank()
    )
    with pytest.raises(NeverVisibleError, match="never visible"):
        blank.first_box(3)
    assert blank.gt_mask(3, 1) is None
    assert blank.gt_mask(3, 4) is not None


def test_out_of_range_inputs(video: VideoSample) -> None:
    with pytest.raises(IndexError):
        video.frame_path(20)
    with pytest.raises(KeyError):
        video.first_box(7)


def test_missing_annotation_is_a_clear_error(tmp_path: Path) -> None:
    videos = build_davis_root(tmp_path)
    videos[0].annotation_path(7).unlink()
    with pytest.raises(FileNotFoundError, match="missing annotation.*'00007'"):
        load_davis_video(make_config(tmp_path), "zebra")


def test_missing_list_file(tmp_path: Path) -> None:
    build_davis_root(tmp_path)
    (tmp_path / "DAVIS" / "ImageSets" / "2017" / "val.txt").unlink()
    with pytest.raises(FileNotFoundError, match="video list not found"):
        list_davis_videos(make_config(tmp_path))


def test_video_not_in_list(tmp_path: Path) -> None:
    build_davis_root(tmp_path)
    with pytest.raises(KeyError, match="not in the DAVIS val list"):
        load_davis_video(make_config(tmp_path), "giraffe")


def test_missing_frames_folder(tmp_path: Path) -> None:
    videos = build_davis_root(tmp_path)
    for path in videos[0].frames_dir.iterdir():
        path.unlink()
    videos[0].frames_dir.rmdir()
    with pytest.raises(FileNotFoundError, match="folder not found"):
        load_davis_video(make_config(tmp_path), "zebra")
