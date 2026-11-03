"""Tiny deterministic DAVIS-style video used by tests.

Object 1 is a square moving right at constant speed. It passes behind a static grey bar
(partly, then fully hidden), later vanishes for a few frames and reappears. Object 2 is a
small static square. Expected facts are computed from the constants below, never from the
written files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.boxes import Box
from src.common.masks import LabelMap, write_label_map

NUM_FRAMES = 20
HEIGHT = 64
WIDTH = 96
JPEG_QUALITY = 95

BACKGROUND_RGB = (230, 230, 230)
OBJECT1_RGB = (200, 30, 30)
OBJECT2_RGB = (30, 30, 200)
OCCLUDER_RGB = (110, 110, 110)

OBJECT1_SIZE = 12
OBJECT1_X0 = 3
OBJECT1_Y0 = 20
OBJECT1_SPEED = 3
OBJECT1_ID = 1
OBJECT2_ID = 2
OBJECT2_BOX = Box(8, 44, 16, 52)

OCCLUDER_BOX = Box(27, 10, 45, 50)
VANISH_FRAMES = range(14, 17)
FULLY_OCCLUDED_FRAMES = range(8, 11)
PARTLY_OCCLUDED_FRAMES = (5, 6, 7, 11, 12)


@dataclass(frozen=True)
class FrameFacts:
    object1_box: Box | None
    object1_area: int
    object2_box: Box
    object2_area: int


@dataclass(frozen=True)
class SyntheticVideo:
    name: str
    root: Path
    frames_dir: Path
    annotations_dir: Path
    num_frames: int
    height: int
    width: int
    facts: tuple[FrameFacts, ...]

    def frame_path(self, index: int) -> Path:
        return self.frames_dir / f"{index:05d}.jpg"

    def annotation_path(self, index: int) -> Path:
        return self.annotations_dir / f"{index:05d}.png"


def _object1_columns(index: int) -> range:
    x0 = OBJECT1_X0 + OBJECT1_SPEED * index
    return range(x0, x0 + OBJECT1_SIZE)


def _visible_columns(index: int) -> list[int]:
    hidden = range(OCCLUDER_BOX.x0, OCCLUDER_BOX.x1)
    return [c for c in _object1_columns(index) if c not in hidden]


def _expected_facts(index: int) -> FrameFacts:
    box1: Box | None = None
    area1 = 0
    columns = _visible_columns(index)
    if index not in VANISH_FRAMES and columns:
        box1 = Box(columns[0], OBJECT1_Y0, columns[-1] + 1, OBJECT1_Y0 + OBJECT1_SIZE)
        area1 = len(columns) * OBJECT1_SIZE
    return FrameFacts(box1, area1, OBJECT2_BOX, OBJECT2_BOX.area)


def _fill(image: np.ndarray, box: Box, colour: tuple[int, int, int]) -> None:
    image[box.y0 : box.y1, box.x0 : box.x1] = colour


def _object1_box(index: int) -> Box:
    cols = _object1_columns(index)
    return Box(cols[0], OBJECT1_Y0, cols[-1] + 1, OBJECT1_Y0 + OBJECT1_SIZE)


def _render_frame(index: int) -> np.ndarray:
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = BACKGROUND_RGB
    _fill(image, OBJECT2_BOX, OBJECT2_RGB)
    if index not in VANISH_FRAMES:
        _fill(image, _object1_box(index), OBJECT1_RGB)
    _fill(image, OCCLUDER_BOX, OCCLUDER_RGB)
    return image


def _render_label(facts: FrameFacts) -> LabelMap:
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    if facts.object1_box is not None:
        b = facts.object1_box
        label[b.y0 : b.y1, b.x0 : b.x1] = OBJECT1_ID
    b2 = facts.object2_box
    label[b2.y0 : b2.y1, b2.x0 : b2.x1] = OBJECT2_ID
    return label


def make_synthetic_video(root: Path, name: str = "synth") -> SyntheticVideo:
    frames_dir = root / "JPEGImages" / "480p" / name
    annotations_dir = root / "Annotations" / "480p" / name
    frames_dir.mkdir(parents=True, exist_ok=True)
    facts = tuple(_expected_facts(i) for i in range(NUM_FRAMES))
    video = SyntheticVideo(
        name, root, frames_dir, annotations_dir, NUM_FRAMES, HEIGHT, WIDTH, facts
    )
    for i in range(NUM_FRAMES):
        Image.fromarray(_render_frame(i)).save(
            video.frame_path(i), format="JPEG", quality=JPEG_QUALITY
        )
        write_label_map(video.annotation_path(i), _render_label(facts[i]))
    return video
