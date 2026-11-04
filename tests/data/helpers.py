from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.boxes import Box
from src.common.config import DataConfig, DavisConfig, SavConfig
from tests.fixtures.synthetic_video import (
    HEIGHT,
    WIDTH,
    SyntheticVideo,
    make_synthetic_video,
)

VIDEO_NAMES = ("zebra", "bear")


def make_config(data_root: Path) -> DataConfig:
    return DataConfig(
        data_root=data_root,
        davis=DavisConfig(Path("DAVIS"), "val", "480p", 2017),
        sav=SavConfig(Path("SA-V"), "sav_val", "sav_test"),
        output_root=Path("out"),
    )


def build_davis_root(data_root: Path, names: tuple[str, ...] = VIDEO_NAMES) -> list[SyntheticVideo]:
    davis = data_root / "DAVIS"
    videos = [make_synthetic_video(davis, name) for name in names]
    sets = davis / "ImageSets" / "2017"
    sets.mkdir(parents=True)
    (sets / "val.txt").write_text("".join(f"{name}\n" for name in names))
    return videos


SAV_NAME = "vid_a"
SAV_ANNOTATED = (0, 4, 8, 12, 16)
LATE_BOX = Box(10, 46, 20, 56)
LATE_FIRST_FRAME = 8


def write_binary_png(path: Path, box: Box | None) -> None:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    if box is not None:
        mask[box.y0 : box.y1, box.x0 : box.x1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def build_sav_root(
    data_root: Path, split: str = "sav_val", names: tuple[str, ...] = (SAV_NAME,)
) -> Path:
    """SA-V layout: 20 frames, ground truth on frames 0, 4, 8, 12, 16 for objects 000-002."""
    root = data_root / "SA-V" / split
    scratch = data_root / "scratch"
    for name in names:
        video = make_synthetic_video(scratch, name)
        frames = root / "JPEGImages_24fps" / name
        frames.mkdir(parents=True)
        for i in range(video.num_frames):
            shutil.copy(video.frame_path(i), frames / f"{i:05d}.jpg")
        for i in SAV_ANNOTATED:
            ann = root / "Annotations_6fps" / name
            write_binary_png(ann / "000" / f"{i:05d}.png", video.facts[i].object1_box)
            write_binary_png(ann / "001" / f"{i:05d}.png", video.facts[i].object2_box)
            late = LATE_BOX if i >= LATE_FIRST_FRAME else None
            write_binary_png(ann / "002" / f"{i:05d}.png", late)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{split}.txt").write_text("".join(f"{name}\n" for name in names))
    return root
