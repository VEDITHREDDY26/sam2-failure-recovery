"""Builders for DAVIS-like and SA-V-like trees from the synthetic video."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.masks import LabelMap, read_label_map, write_label_map
from tests.fixtures.synthetic_video import NUM_FRAMES, SyntheticVideo

SAV_STEP = 4
OBJECT_IDS = (1, 2)


def make_davis_root(video: SyntheticVideo, root: Path) -> Path:
    """DAVIS tree (ImageSets, JPEGImages, Annotations) holding the synthetic video as val."""
    (root / "ImageSets" / "2017").mkdir(parents=True)
    (root / "ImageSets" / "2017" / "val.txt").write_text(f"{video.name}\n")
    shutil.copytree(video.frames_dir.parent, root / "JPEGImages" / "480p")
    shutil.copytree(video.annotations_dir.parent, root / "Annotations" / "480p")
    return root


def make_sav_gt(video: SyntheticVideo, root: Path) -> Path:
    """SA-V GT: <root>/<video>/<000|001>/<stem>.png binary, every 4th frame."""
    for index in range(0, NUM_FRAMES, SAV_STEP):
        label = read_label_map(video.annotation_path(index))
        for obj in OBJECT_IDS:
            path = root / video.name / f"{obj - 1:03d}" / f"{index:05d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((label == obj).astype(np.uint8) * 255).save(path)
    return root


def copy_annotations(video: SyntheticVideo, dest: Path) -> Path:
    shutil.copytree(video.annotations_dir, dest / video.name)
    return dest


def rewrite_labels(pred: Path, video: SyntheticVideo, edit: Callable[[LabelMap], LabelMap]) -> None:
    for index in range(NUM_FRAMES):
        path = pred / video.name / f"{index:05d}.png"
        write_label_map(path, edit(read_label_map(path)))
