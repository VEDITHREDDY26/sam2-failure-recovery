"""SA-V val/test loader (24 fps frames, ground truth on every 4th frame, one PNG per object).

<data_root>/<root>/<split>/<split>.txt                        one video id per line
<data_root>/<root>/<split>/JPEGImages_24fps/<video>/*.jpg     every frame
<data_root>/<root>/<split>/Annotations_6fps/<video>/<obj>/*.png   binary mask, annotated frames

An (object, frame) pair is annotated iff its PNG exists. A missing PNG means "not annotated"
(gt_mask is None); an existing all-zero PNG means "annotated, object absent" (all-False).
Object folders are numeric ("000", "001", ...): object id = int(folder) + 1 and the folder name
is kept in `object_names`. Objects may overlap, so masks are never merged into one label map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.common.config import DataConfig
from src.common.layout import validate_name
from src.common.masks import MAX_OBJECT_ID, BoolMask
from src.data.video import VideoSample, sort_stems

DATASET = "sav"
FRAME_SUFFIX = ".jpg"
ANNOTATION_SUFFIX = ".png"
FRAMES_FOLDER = "JPEGImages_24fps"
ANNOTATIONS_FOLDER = "Annotations_6fps"
SPLITS = ("val", "test")


def split_folder(cfg: DataConfig, split: str) -> str:
    if split == "val":
        return cfg.sav.val_split
    if split == "test":
        return cfg.sav.test_split
    raise ValueError(f"invalid SA-V split '{split}': expected one of {SPLITS}")


def split_root(cfg: DataConfig, split: str) -> Path:
    return cfg.resolve(cfg.sav.root) / split_folder(cfg, split)


def list_file(cfg: DataConfig, split: str) -> Path:
    folder = split_folder(cfg, split)
    return split_root(cfg, split) / f"{folder}.txt"


def list_sav_videos(cfg: DataConfig, split: str) -> tuple[str, ...]:
    path = list_file(cfg, split)
    if not path.is_file():
        raise FileNotFoundError(f"SA-V video list not found: {path}")
    names = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate video names")
    return names


def object_id_from_folder(folder: str) -> int:
    if not folder.isdigit():
        raise ValueError(f"object folder '{folder}' is not numeric")
    obj_id = int(folder) + 1
    if obj_id > MAX_OBJECT_ID:
        raise ValueError(f"object folder '{folder}' gives id {obj_id} above {MAX_OBJECT_ID}")
    return obj_id


def _stems(directory: Path, suffix: str) -> tuple[str, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(f"folder not found: {directory}")
    return sort_stems(p.stem for p in directory.iterdir() if p.suffix.lower() == suffix)


class SavReader:
    """Per-object sparse masks; frames are indexed by position in the 24 fps frame list."""

    def __init__(self, paths: dict[int, dict[int, Path]]) -> None:
        self._paths = paths

    def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
        return tuple(sorted(self._paths[obj_id]))

    def annotation_path(self, obj_id: int, frame_index: int) -> Path:
        return self._paths[obj_id][frame_index]

    def read_mask(self, obj_id: int, frame_index: int) -> BoolMask:
        path = self._paths[obj_id][frame_index]
        with Image.open(path) as image:
            mask = np.array(image) > 0
        if mask.ndim != 2:
            raise ValueError(f"{path}: expected a single-channel mask, got shape {mask.shape}")
        return mask


def _object_paths(
    annotations_dir: Path, folder: str, frame_index_of: dict[str, int]
) -> dict[int, Path]:
    obj_dir = annotations_dir / folder
    paths: dict[int, Path] = {}
    for stem in _stems(obj_dir, ANNOTATION_SUFFIX):
        if stem not in frame_index_of:
            raise ValueError(f"{obj_dir}: annotation '{stem}' has no matching frame")
        paths[frame_index_of[stem]] = obj_dir / f"{stem}{ANNOTATION_SUFFIX}"
    if not paths:
        raise ValueError(f"{obj_dir}: no annotation PNGs")
    return paths


def load_sav_video(cfg: DataConfig, split: str, name: str) -> VideoSample:
    validate_name(name, "video")
    if name not in list_sav_videos(cfg, split):
        raise KeyError(f"video '{name}' is not in the SA-V {split} list")
    root = split_root(cfg, split)
    frames_dir = root / FRAMES_FOLDER / name
    annotations_dir = root / ANNOTATIONS_FOLDER / name
    stems = _stems(frames_dir, FRAME_SUFFIX)
    if not stems:
        raise FileNotFoundError(f"{frames_dir}: no {FRAME_SUFFIX} frames")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"folder not found: {annotations_dir}")
    index_of = {stem: i for i, stem in enumerate(stems)}
    folders = sorted((p.name for p in annotations_dir.iterdir() if p.is_dir()), key=_folder_key)
    ids = {folder: object_id_from_folder(folder) for folder in folders}
    if not ids:
        raise ValueError(f"{annotations_dir}: no object folders")
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"{annotations_dir}: object folders map to duplicate ids")
    paths = {ids[f]: _object_paths(annotations_dir, f, index_of) for f in folders}
    ordered = tuple(sorted(paths))
    return VideoSample(
        dataset=DATASET,
        split=f"{DATASET}_{split}",
        name=name,
        frames_dir=frames_dir,
        frame_stems=stems,
        object_ids=ordered,
        object_names={i: f for f, i in ids.items()},
        reader=SavReader(paths),
        frame_suffix=FRAME_SUFFIX,
    )


def _folder_key(folder: str) -> tuple[int, str]:
    return (int(folder), folder) if folder.isdigit() else (-1, folder)
