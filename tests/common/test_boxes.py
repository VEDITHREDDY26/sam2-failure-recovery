import numpy as np
import pytest

from src.common.boxes import Box, box_from_mask, clip, from_xywh, to_xywh


def test_properties() -> None:
    box = Box(3, 2, 7, 5)
    assert (box.width, box.height, box.area) == (4, 3, 12)


@pytest.mark.parametrize("args", [(3, 2, 3, 5), (3, 2, 2, 5), (3, 2, 4, 2), (3, 2, 4, 1)])
def test_invalid_box_raises(args: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        Box(*args)


def test_single_pixel_mask() -> None:
    mask = np.zeros((6, 8), dtype=bool)
    mask[2, 3] = True
    assert box_from_mask(mask) == Box(3, 2, 4, 3)


def test_rectangle_mask_uint8() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:4, 2:7] = 1
    mask[8, 9] = 1
    assert box_from_mask(mask) == Box(2, 1, 10, 9)
    mask[8, 9] = 0
    assert box_from_mask(mask) == Box(2, 1, 7, 4)


def test_empty_mask_is_none() -> None:
    assert box_from_mask(np.zeros((5, 5), dtype=bool)) is None


def test_mask_must_be_2d() -> None:
    with pytest.raises(ValueError):
        box_from_mask(np.zeros((2, 3, 4), dtype=bool))


def test_xywh_round_trip() -> None:
    box = Box(3, 2, 7, 5)
    assert to_xywh(box) == (3, 2, 4, 3)
    assert from_xywh(3, 2, 4, 3) == box


def test_clip_partly_outside() -> None:
    assert clip(Box(-2, -3, 5, 4), height=10, width=4) == Box(0, 0, 4, 4)


def test_clip_inside_unchanged() -> None:
    assert clip(Box(1, 1, 3, 3), height=10, width=10) == Box(1, 1, 3, 3)


@pytest.mark.parametrize("box", [Box(10, 0, 12, 3), Box(-5, -5, 0, 2), Box(0, 8, 3, 9)])
def test_clip_fully_outside(box: Box) -> None:
    assert clip(box, height=8, width=10) is None
