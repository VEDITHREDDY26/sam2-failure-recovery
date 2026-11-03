"""Mask storage: one indexed PNG (mode "P") per frame, pixel value = object id.

Id 0 is background, ids 1..255 are objects, the palette is the standard DAVIS colour map.
Overlaps are resolved before storage (see `masks_to_label_map`), so a label map never holds
two objects on one pixel.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

MAX_OBJECT_ID = 255

BoolMask = npt.NDArray[np.bool_]
LabelMap = npt.NDArray[np.uint8]


def davis_palette() -> list[int]:
    """Standard PASCAL VOC / DAVIS colour map as a flat list of 256*3 ints."""
    palette: list[int] = []
    for index in range(256):
        r = g = b = 0
        c = index
        for bit in range(8):
            r |= ((c >> 0) & 1) << (7 - bit)
            g |= ((c >> 1) & 1) << (7 - bit)
            b |= ((c >> 2) & 1) << (7 - bit)
            c >>= 3
        palette.extend((r, g, b))
    return palette


def write_label_map(path: str | Path, label: LabelMap) -> None:
    if label.ndim != 2 or label.dtype != np.uint8:
        raise ValueError(f"label map must be a 2-D uint8 array, got {label.dtype} {label.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(label)
    image.putpalette(davis_palette())
    image.save(path, format="PNG")


def read_label_map(path: str | Path) -> LabelMap:
    with Image.open(path) as image:
        if image.mode != "P":
            raise ValueError(f"{path}: expected an indexed (mode P) PNG, got mode {image.mode}")
        return np.array(image, dtype=np.uint8)


def label_map_to_masks(label: LabelMap) -> dict[int, BoolMask]:
    """One boolean mask per object id present, ascending; background excluded."""
    ids = [int(i) for i in np.unique(label) if i != 0]
    return {i: label == i for i in ids}


def masks_to_label_map(
    masks: Mapping[int, BoolMask],
    shape: tuple[int, int],
    scores: Mapping[int, npt.NDArray[np.floating]] | None = None,
) -> LabelMap:
    """Combine per-object masks into one label map.

    Overlap rule: a pixel claimed by several objects goes to the object with the highest
    per-pixel score (logit) if `scores` is given (ties go to the smaller id), otherwise to
    the smallest object id.
    """
    for obj_id, mask in masks.items():
        if not 1 <= obj_id <= MAX_OBJECT_ID:
            raise ValueError(f"object id {obj_id} outside 1..{MAX_OBJECT_ID}")
        if mask.shape != shape:
            raise ValueError(f"mask for object {obj_id} has shape {mask.shape}, expected {shape}")
        if scores is not None:
            if obj_id not in scores:
                raise ValueError(f"no scores for object {obj_id}")
            if scores[obj_id].shape != shape:
                raise ValueError(f"scores for object {obj_id} have shape {scores[obj_id].shape}")
    label = np.zeros(shape, dtype=np.uint8)
    best: npt.NDArray[np.float64] = np.full(shape, -np.inf)
    for obj_id in sorted(masks):
        mask = masks[obj_id]
        if scores is None:
            take = mask & (label == 0)
        else:
            take = mask & (scores[obj_id] > best)
            best = np.where(take, scores[obj_id], best).astype(np.float64)
        label[take] = obj_id
    return label
