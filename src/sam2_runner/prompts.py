"""First-frame box prompts, built outside this package (never from ground truth in here).

Single video (smoke): {"video": str, "frame_index": 0, "boxes": {"<obj_id>": [x0, y0, x1, y1]}}
Many videos (batch):
    {"split": "davis_val",
     "videos": {"<video>": {"frame_index": 0, "boxes": {"<obj_id>": [x0, y0, x1, y1]}}}}
Boxes use the project Box convention (right and bottom edges exclusive).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.boxes import Box
from src.common.layout import validate_name, validate_split


@dataclass(frozen=True)
class Prompts:
    video: str
    frame_index: int
    boxes: dict[int, Box]


@dataclass(frozen=True)
class BatchPrompts:
    split: str
    videos: dict[str, Prompts]


def parse_prompts(raw: Mapping[str, Any]) -> Prompts:
    return _parse_video(str(raw["video"]), raw)


def _parse_video(video: str, raw: Mapping[str, Any]) -> Prompts:
    boxes: dict[int, Box] = {}
    for key, coords in raw["boxes"].items():
        if len(coords) != 4:
            raise ValueError(f"box for object {key} needs 4 numbers, got {coords}")
        boxes[int(key)] = Box(*(int(v) for v in coords))
    if not boxes:
        raise ValueError(f"prompts for '{video}' contain no boxes")
    return Prompts(video, int(raw["frame_index"]), boxes)


def load_prompts(path: Path) -> Prompts:
    return parse_prompts(json.loads(path.read_text()))


def parse_batch_prompts(raw: Mapping[str, Any]) -> BatchPrompts:
    split = validate_split(str(raw["split"]))
    videos = {
        validate_name(name, "video"): _parse_video(name, entry)
        for name, entry in raw["videos"].items()
    }
    if not videos:
        raise ValueError("prompts contain no videos")
    return BatchPrompts(split, videos)


def load_batch_prompts(path: Path) -> BatchPrompts:
    return parse_batch_prompts(json.loads(path.read_text()))
