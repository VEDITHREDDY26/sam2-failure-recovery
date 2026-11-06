"""First-frame box prompts, built outside this package (never from ground truth in here).

Single video (smoke): {"video": str, "frame_index": 0, "boxes": {"<obj_id>": [x0, y0, x1, y1]}}
Many videos (batch):
    {"split": "davis_val",
     "videos": {"<video>": {"frame_index": 0, "boxes": {"<obj_id>": [x0, y0, x1, y1]}}}}
A box entry is either [x0, y0, x1, y1] (prompted at the video's "frame_index") or
{"frame_index": k, "box": [x0, y0, x1, y1]} (prompted at its own frame k). When every entry has
its own frame, the video-level "frame_index" may be left out.
Boxes use the project Box convention (right and bottom edges exclusive).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.boxes import Box
from src.common.layout import validate_name, validate_split


@dataclass(frozen=True)
class Prompts:
    """frame_index is the default prompt frame; object_frames holds per-object overrides."""

    video: str
    frame_index: int
    boxes: dict[int, Box]
    object_frames: dict[int, int] = field(default_factory=dict)

    def start_frame(self, obj_id: int) -> int:
        return self.object_frames.get(obj_id, self.frame_index)

    def groups(self) -> dict[int, list[int]]:
        """Object ids by prompt frame, ascending frame and ascending id."""
        found: dict[int, list[int]] = {}
        for obj_id in sorted(self.boxes):
            found.setdefault(self.start_frame(obj_id), []).append(obj_id)
        return dict(sorted(found.items()))


@dataclass(frozen=True)
class BatchPrompts:
    split: str
    videos: dict[str, Prompts]


def parse_prompts(raw: Mapping[str, Any]) -> Prompts:
    return _parse_video(str(raw["video"]), raw)


def _parse_box(key: str, coords: Any) -> Box:
    if len(coords) != 4:
        raise ValueError(f"box for object {key} needs 4 numbers, got {coords}")
    return Box(*(int(v) for v in coords))


def _parse_frame(key: str, value: Any) -> int:
    frame = int(value)
    if frame < 0:
        raise ValueError(f"object {key}: frame_index must be >= 0, got {frame}")
    return frame


def _parse_video(video: str, raw: Mapping[str, Any]) -> Prompts:
    boxes: dict[int, Box] = {}
    own_frames: dict[int, int] = {}
    for key, entry in raw["boxes"].items():
        if isinstance(entry, Mapping):
            if "box" not in entry or "frame_index" not in entry:
                raise ValueError(f"object {key} needs both 'frame_index' and 'box', got {entry}")
            boxes[int(key)] = _parse_box(key, entry["box"])
            own_frames[int(key)] = _parse_frame(key, entry["frame_index"])
        else:
            boxes[int(key)] = _parse_box(key, entry)
    if not boxes:
        raise ValueError(f"prompts for '{video}' contain no boxes")
    if "frame_index" in raw:
        default = _parse_frame("video", raw["frame_index"])
    elif len(own_frames) == len(boxes):
        default = min(own_frames.values())
    else:
        raise ValueError(f"prompts for '{video}': plain boxes need a video-level 'frame_index'")
    return Prompts(video, default, boxes, own_frames)


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
