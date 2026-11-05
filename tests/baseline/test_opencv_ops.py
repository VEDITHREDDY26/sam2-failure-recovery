from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.baseline.opencv_ops import (
    CsrtTracker,
    grabcut_in_box,
    pad_to_min_side,
    rgb_to_bgr,
)
from src.common.boxes import Box, from_xywh, to_xywh


def asymmetric_rgb() -> np.ndarray:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:] = (200, 30, 10)  # R, G, B all different
    return image


def test_rgb_to_bgr_swaps_channels() -> None:
    bgr = rgb_to_bgr(asymmetric_rgb())
    assert tuple(bgr[0, 0]) == (10, 30, 200)


def test_opencv_receives_bgr(tmp_path: Path) -> None:
    """OpenCV writes what it is given as BGR, so reading the file as RGB gives back the RGB."""
    path = tmp_path / "x.png"
    assert cv2.imwrite(str(path), rgb_to_bgr(asymmetric_rgb()))
    with Image.open(path) as image:
        assert tuple(np.array(image.convert("RGB"))[0, 0]) == (200, 30, 10)


def test_unconverted_rgb_would_be_wrong(tmp_path: Path) -> None:
    path = tmp_path / "y.png"
    cv2.imwrite(str(path), asymmetric_rgb())
    with Image.open(path) as image:
        assert tuple(np.array(image.convert("RGB"))[0, 0]) == (10, 30, 200)


def test_rgb_to_bgr_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        rgb_to_bgr(np.zeros((4, 6), dtype=np.uint8))
    with pytest.raises(ValueError):
        rgb_to_bgr(np.zeros((4, 6, 3), dtype=np.float32))  # type: ignore[arg-type]


def test_box_round_trip_at_the_boundary() -> None:
    box = Box(3, 5, 13, 11)
    assert from_xywh(*to_xywh(box)) == box
    assert to_xywh(box) == (3, 5, 10, 6)


def scene() -> np.ndarray:
    image = np.full((60, 80, 3), 230, dtype=np.uint8)
    image[20:40, 30:55] = (200, 30, 30)
    return image


def test_grabcut_finds_object_inside_box() -> None:
    mask = grabcut_in_box(rgb_to_bgr(scene()), Box(26, 16, 59, 44), 5, 0.25)
    assert mask.shape == (60, 80)
    assert mask[20:40, 30:55].mean() > 0.9
    assert not mask[:16].any() and not mask[:, :26].any() and not mask[44:].any()


def test_grabcut_on_degenerate_input_is_empty_not_an_error() -> None:
    flat = rgb_to_bgr(np.full((20, 20, 3), 100, dtype=np.uint8))
    mask = grabcut_in_box(flat, Box(0, 0, 20, 20), 5, 0.25)
    assert mask.shape == (20, 20)


def test_pad_to_min_side_grows_and_stays_inside() -> None:
    padded = pad_to_min_side(Box(0, 0, 2, 2), 4, 10, 10)
    assert padded.width >= 4 and padded.height >= 4 and padded.x0 == 0
    assert pad_to_min_side(Box(1, 1, 9, 9), 4, 10, 10) == Box(1, 1, 9, 9)


def test_csrt_follows_a_moving_square() -> None:
    def frame(x: int) -> np.ndarray:
        image = np.full((60, 120, 3), 230, dtype=np.uint8)
        rng = np.random.default_rng(0)
        image[20:40, x : x + 20] = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
        return image

    tracker = CsrtTracker(min_side=4)
    tracker.init(rgb_to_bgr(frame(10)), Box(10, 20, 30, 40))
    result = tracker.update(rgb_to_bgr(frame(14)))
    assert result.ok and result.box is not None
    assert abs(result.box.x0 - 14) <= 4
