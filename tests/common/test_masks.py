from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common.masks import (
    davis_palette,
    label_map_to_masks,
    masks_to_label_map,
    read_label_map,
    write_label_map,
)


def test_empty_frame_round_trip(tmp_path: Path) -> None:
    label = np.zeros((6, 9), dtype=np.uint8)
    path = tmp_path / "a" / "b" / "00000.png"
    write_label_map(path, label)
    assert np.array_equal(read_label_map(path), label)
    with Image.open(path) as image:
        assert image.mode == "P"
    assert label_map_to_masks(read_label_map(path)) == {}


def _three_objects() -> dict[int, np.ndarray]:
    masks = {i: np.zeros((8, 10), dtype=bool) for i in (1, 3, 7)}
    masks[1][0:3, 0:3] = True
    masks[3][4:6, 5:9] = True
    masks[7][7, :] = True
    return masks


def test_multi_object_round_trip(tmp_path: Path) -> None:
    masks = _three_objects()
    label = masks_to_label_map(masks, (8, 10))
    path = tmp_path / "f.png"
    write_label_map(path, label)
    back = read_label_map(path)
    assert np.array_equal(back, label)
    restored = label_map_to_masks(back)
    assert list(restored) == [1, 3, 7]
    for obj_id, mask in masks.items():
        assert np.array_equal(restored[obj_id], mask)
    assert np.array_equal(masks_to_label_map(restored, (8, 10)), label)


def test_overlap_by_id_and_by_score() -> None:
    a = np.zeros((2, 3), dtype=bool)
    b = np.zeros((2, 3), dtype=bool)
    a[:, :2] = True
    b[:, 1:] = True
    by_id = masks_to_label_map({2: b, 5: a}, (2, 3))
    assert by_id.tolist() == [[5, 2, 2], [5, 2, 2]]
    scores = {5: np.full((2, 3), 1.0), 2: np.array([[0, 0, 0], [0, 3.0, 0]])}
    by_score = masks_to_label_map({2: b, 5: a}, (2, 3), scores)
    assert by_score.tolist() == [[5, 5, 2], [5, 2, 2]]


def test_rgb_png_raises(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="mode P"):
        read_label_map(path)


def test_palette_matches_davis() -> None:
    palette = davis_palette()
    assert len(palette) == 768
    assert tuple(palette[0:3]) == (0, 0, 0)
    assert tuple(palette[3:6]) == (128, 0, 0)
    assert tuple(palette[6:9]) == (0, 128, 0)
    assert tuple(palette[21:24]) == (128, 128, 128)


def test_validation() -> None:
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        masks_to_label_map({0: mask}, (2, 2))
    with pytest.raises(ValueError):
        masks_to_label_map({256: mask}, (2, 2))
    with pytest.raises(ValueError):
        masks_to_label_map({1: mask}, (3, 2))
    with pytest.raises(ValueError):
        masks_to_label_map({1: mask}, (2, 2), {})
