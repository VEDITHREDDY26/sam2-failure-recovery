from pathlib import Path

import numpy as np
from PIL import Image

from src.common.boxes import Box, box_from_mask
from src.common.masks import read_label_map
from tests.fixtures.synthetic_video import (
    FULLY_OCCLUDED_FRAMES,
    HEIGHT,
    NUM_FRAMES,
    OBJECT1_RGB,
    PARTLY_OCCLUDED_FRAMES,
    VANISH_FRAMES,
    WIDTH,
    SyntheticVideo,
    make_synthetic_video,
)

JPEG_TOLERANCE = 40


def _area(video: SyntheticVideo, index: int, obj: int) -> int:
    return int((read_label_map(video.annotation_path(index)) == obj).sum())


def test_frame_count_and_names(synthetic_video: SyntheticVideo) -> None:
    names = sorted(p.name for p in synthetic_video.frames_dir.iterdir())
    assert names == [f"{i:05d}.jpg" for i in range(NUM_FRAMES)]
    ann = sorted(p.name for p in synthetic_video.annotations_dir.iterdir())
    assert ann == [f"{i:05d}.png" for i in range(NUM_FRAMES)]
    assert synthetic_video.num_frames == NUM_FRAMES == 20


def test_image_size_and_mode(synthetic_video: SyntheticVideo) -> None:
    with Image.open(synthetic_video.frame_path(0)) as image:
        assert image.mode == "RGB"
        assert image.size == (96, 64) == (WIDTH, HEIGHT)


def test_ground_truth_ids(synthetic_video: SyntheticVideo) -> None:
    for i in range(NUM_FRAMES):
        ids = set(np.unique(read_label_map(synthetic_video.annotation_path(i))).tolist())
        assert ids <= {0, 1, 2}
        assert 2 in ids


def test_first_frame_box_matches_literal(synthetic_video: SyntheticVideo) -> None:
    label = read_label_map(synthetic_video.annotation_path(0))
    assert box_from_mask(label == 1) == Box(3, 20, 15, 32)
    assert box_from_mask(label == 2) == Box(8, 44, 16, 52)


def test_written_masks_match_expected_facts(synthetic_video: SyntheticVideo) -> None:
    for i, facts in enumerate(synthetic_video.facts):
        label = read_label_map(synthetic_video.annotation_path(i))
        assert box_from_mask(label == 1) == facts.object1_box
        assert int((label == 1).sum()) == facts.object1_area


def test_area_drops_during_occlusion(synthetic_video: SyntheticVideo) -> None:
    full = _area(synthetic_video, 0, 1)
    assert full == 144
    for i in PARTLY_OCCLUDED_FRAMES:
        assert 0 < _area(synthetic_video, i, 1) < full
    for i in FULLY_OCCLUDED_FRAMES:
        assert _area(synthetic_video, i, 1) == 0


def test_disappearance_and_reappearance(synthetic_video: SyntheticVideo) -> None:
    for i in VANISH_FRAMES:
        assert _area(synthetic_video, i, 1) == 0
    after = VANISH_FRAMES[-1] + 1
    assert _area(synthetic_video, after, 1) == 144
    assert box_from_mask(read_label_map(synthetic_video.annotation_path(after)) == 1) == Box(
        54, 20, 66, 32
    )


def test_object_two_constant(synthetic_video: SyntheticVideo) -> None:
    for i in range(NUM_FRAMES):
        label = read_label_map(synthetic_video.annotation_path(i))
        assert box_from_mask(label == 2) == Box(8, 44, 16, 52)
        assert int((label == 2).sum()) == 64


def test_generation_is_deterministic(tmp_path: Path) -> None:
    a = make_synthetic_video(tmp_path / "a")
    b = make_synthetic_video(tmp_path / "b")
    for i in range(NUM_FRAMES):
        assert a.annotation_path(i).read_bytes() == b.annotation_path(i).read_bytes()
        assert a.frame_path(i).read_bytes() == b.frame_path(i).read_bytes()


def _is_object1_colour(pixel: np.ndarray) -> bool:
    return bool(np.abs(pixel.astype(int) - np.array(OBJECT1_RGB)).max() <= JPEG_TOLERANCE)


def _interior(box: Box) -> tuple[slice, slice]:
    return slice(box.y0 + 2, box.y1 - 2), slice(box.x0 + 2, box.x1 - 2)


def test_pixels_match_ground_truth_visibility(synthetic_video: SyntheticVideo) -> None:
    for index in (0, 6, 17):
        with Image.open(synthetic_video.frame_path(index)) as image:
            rgb = np.array(image)
        box = synthetic_video.facts[index].object1_box
        assert box is not None
        ys, xs = _interior(box)
        assert all(_is_object1_colour(p) for p in rgb[ys, xs].reshape(-1, 3))
        label = read_label_map(synthetic_video.annotation_path(index))
        outside = rgb[label != 1].reshape(-1, 3)
        near = np.abs(outside.astype(int) - np.array(OBJECT1_RGB)).max(axis=1) <= JPEG_TOLERANCE
        assert near.sum() == 0


def test_object_colour_absent_in_vanish_frame(synthetic_video: SyntheticVideo) -> None:
    with Image.open(synthetic_video.frame_path(15)) as image:
        rgb = np.array(image).reshape(-1, 3)
    near = np.abs(rgb.astype(int) - np.array(OBJECT1_RGB)).max(axis=1) <= JPEG_TOLERANCE
    assert near.sum() == 0


def test_partly_occluded_frame_literal_box(synthetic_video: SyntheticVideo) -> None:
    facts = synthetic_video.facts[6]
    assert facts.object1_box == Box(21, 20, 27, 32)
    assert facts.object1_area == 72
    label = read_label_map(synthetic_video.annotation_path(6))
    assert box_from_mask(label == 1) == Box(21, 20, 27, 32)
