"""Dataset-independent view of one video: frames, objects, sparse ground truth, first box.

Ground truth is read lazily through a `GtReader`. `gt_mask` returns None when a frame is not
annotated for an object and an all-False array when it is annotated but the object is absent.
`frame_index` is the 0-based position in `frame_stems`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from src.common.boxes import Box, box_from_mask
from src.common.masks import MAX_OBJECT_ID, BoolMask


class GtReader(Protocol):
    def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
        """Ascending frame indices with ground truth for this object."""
        ...

    def read_mask(self, obj_id: int, frame_index: int) -> BoolMask:
        """Mask of an annotated (object, frame) pair."""
        ...


@dataclasses.dataclass(frozen=True)
class PromptBox:
    frame_index: int
    box: Box


class NeverVisibleError(ValueError):
    """The object has no non-empty annotated frame, so there is nothing to prompt with."""


def sort_stems(names: Iterable[str]) -> tuple[str, ...]:
    """Numeric order of frame stems (so '2' comes before '10'); non-numeric stems are an error."""
    stems = list(names)
    for stem in stems:
        if not stem.isdigit():
            raise ValueError(f"frame name '{stem}' is not numeric")
    return tuple(sorted(stems, key=int))


@dataclasses.dataclass(frozen=True, eq=False)
class VideoSample:
    dataset: str
    split: str
    name: str
    frames_dir: Path
    frame_stems: tuple[str, ...]
    object_ids: tuple[int, ...]
    object_names: dict[int, str]
    reader: GtReader
    frame_suffix: str = ".jpg"

    def __post_init__(self) -> None:
        if not self.frame_stems:
            raise ValueError(f"video '{self.name}' has no frames")
        if list(self.object_ids) != sorted(set(self.object_ids)):
            raise ValueError(f"object ids must be unique and ascending: {self.object_ids}")
        for obj_id in self.object_ids:
            if not 1 <= obj_id <= MAX_OBJECT_ID:
                raise ValueError(f"object id {obj_id} outside 1..{MAX_OBJECT_ID}")
        if set(self.object_names) != set(self.object_ids):
            raise ValueError("object_names must have exactly one entry per object id")

    @property
    def num_frames(self) -> int:
        return len(self.frame_stems)

    def frame_path(self, frame_index: int) -> Path:
        self._check_frame(frame_index)
        return self.frames_dir / f"{self.frame_stems[frame_index]}{self.frame_suffix}"

    def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
        self._check_object(obj_id)
        return self.reader.annotated_frames(obj_id)

    def is_annotated(self, obj_id: int, frame_index: int) -> bool:
        self._check_frame(frame_index)
        return frame_index in self.annotated_frames(obj_id)

    def gt_mask(self, obj_id: int, frame_index: int) -> BoolMask | None:
        if not self.is_annotated(obj_id, frame_index):
            return None
        return self.reader.read_mask(obj_id, frame_index)

    def gt_masks(self, frame_index: int) -> dict[int, BoolMask]:
        """Masks of the objects annotated at this frame (unannotated objects are left out)."""
        masks: dict[int, BoolMask] = {}
        for obj_id in self.object_ids:
            mask = self.gt_mask(obj_id, frame_index)
            if mask is not None:
                masks[obj_id] = mask
        return masks

    def first_box(self, obj_id: int) -> PromptBox:
        """Tight box of the object's first annotated frame where it is visible."""
        for frame_index in self.annotated_frames(obj_id):
            box = box_from_mask(self.reader.read_mask(obj_id, frame_index))
            if box is not None:
                return PromptBox(frame_index, box)
        raise NeverVisibleError(
            f"{self.dataset}/{self.name}: object {obj_id} is never visible in an annotated frame"
        )

    def _check_frame(self, frame_index: int) -> None:
        if not 0 <= frame_index < self.num_frames:
            raise IndexError(f"frame {frame_index} outside 0..{self.num_frames - 1}")

    def _check_object(self, obj_id: int) -> None:
        if obj_id not in self.object_names:
            raise KeyError(f"{self.name}: unknown object id {obj_id}")
