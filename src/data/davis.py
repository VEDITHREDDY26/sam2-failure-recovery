"""DAVIS 2017 loader: official folder layout, objects taken from the frame-0 annotation.

<data_root>/<root>/ImageSets/<year>/<split>.txt            one video name per line
<data_root>/<root>/JPEGImages/<resolution>/<video>/*.jpg
<data_root>/<root>/Annotations/<resolution>/<video>/*.png  indexed PNG, every frame annotated
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.config import DataConfig
from src.common.layout import validate_name
from src.common.masks import BoolMask, label_map_to_masks, read_label_map
from src.data.video import VideoSample, sort_stems

DATASET = "davis"
FRAME_SUFFIX = ".jpg"
ANNOTATION_SUFFIX = ".png"


def _davis_root(cfg: DataConfig) -> Path:
    return cfg.resolve(cfg.davis.root)


def split_name(cfg: DataConfig) -> str:
    return f"{DATASET}_{cfg.davis.split}"


def list_davis_videos(cfg: DataConfig) -> tuple[str, ...]:
    path = _davis_root(cfg) / "ImageSets" / str(cfg.davis.year) / f"{cfg.davis.split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"DAVIS video list not found: {path}")
    names = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate video names")
    return names


def _stems(directory: Path, suffix: str) -> tuple[str, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(f"folder not found: {directory}")
    return sort_stems(p.stem for p in directory.iterdir() if p.suffix.lower() == suffix)


class DavisReader:
    """Reads one label map per frame; keeps only the most recent one in memory."""

    def __init__(self, annotations_dir: Path, stems: tuple[str, ...]) -> None:
        self._dir = annotations_dir
        self._stems = stems
        self._all = tuple(range(len(stems)))
        self._cached: tuple[int, dict[int, BoolMask]] | None = None

    def annotation_path(self, frame_index: int) -> Path:
        return self._dir / f"{self._stems[frame_index]}{ANNOTATION_SUFFIX}"

    def _masks(self, frame_index: int) -> dict[int, BoolMask]:
        if self._cached is None or self._cached[0] != frame_index:
            label = read_label_map(self.annotation_path(frame_index))
            self._cached = (frame_index, label_map_to_masks(label))
        return self._cached[1]

    def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
        return self._all

    def read_mask(self, obj_id: int, frame_index: int) -> BoolMask:
        masks = self._masks(frame_index)
        if obj_id in masks:
            return masks[obj_id]
        shape = next(iter(masks.values())).shape if masks else self._shape(frame_index)
        return _empty(shape)

    def _shape(self, frame_index: int) -> tuple[int, int]:
        height, width = read_label_map(self.annotation_path(frame_index)).shape
        return height, width


def _empty(shape: tuple[int, ...]) -> BoolMask:
    return np.zeros(shape, dtype=bool)


def load_davis_video(cfg: DataConfig, name: str) -> VideoSample:
    validate_name(name, "video")
    if name not in list_davis_videos(cfg):
        raise KeyError(f"video '{name}' is not in the DAVIS {cfg.davis.split} list")
    root = _davis_root(cfg)
    frames_dir = root / "JPEGImages" / cfg.davis.resolution / name
    annotations_dir = root / "Annotations" / cfg.davis.resolution / name
    stems = _stems(frames_dir, FRAME_SUFFIX)
    if not stems:
        raise FileNotFoundError(f"{frames_dir}: no {FRAME_SUFFIX} frames")
    annotated = set(_stems(annotations_dir, ANNOTATION_SUFFIX))
    missing = [s for s in stems if s not in annotated]
    if missing:
        raise FileNotFoundError(
            f"{annotations_dir}: missing annotation for {len(missing)} frame(s), "
            f"first '{missing[0]}'"
        )
    reader = DavisReader(annotations_dir, stems)
    ids = tuple(label_map_to_masks(read_label_map(reader.annotation_path(0))))
    if not ids:
        raise ValueError(f"{reader.annotation_path(0)}: frame 0 has no objects")
    return VideoSample(
        dataset=DATASET,
        split=split_name(cfg),
        name=name,
        frames_dir=frames_dir,
        frame_stems=stems,
        object_ids=ids,
        object_names={i: str(i) for i in ids},
        reader=reader,
        frame_suffix=FRAME_SUFFIX,
    )
