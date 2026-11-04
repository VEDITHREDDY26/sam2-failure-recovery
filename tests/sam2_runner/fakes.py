"""Fake SAM 2 video predictor: replays ground-truth masks of the synthetic video as logits."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from src.common.masks import label_map_to_masks, read_label_map
from tests.fixtures.synthetic_video import SyntheticVideo

POSITIVE = 4.0
NEGATIVE = -4.0


class FakePredictor:
    """Yields (frame_idx, obj_ids, logits[n, 1, H, W]) like SAM2VideoPredictor."""

    def __init__(self, video: SyntheticVideo, overlap: bool = False) -> None:
        self.video = video
        self.overlap = overlap
        self.calls: list[tuple[int, int, tuple[float, ...]]] = []
        self.init_path: str | None = None
        self.obj_ids: list[int] = []

    def init_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> Any:
        self.init_path = video_path
        self.obj_ids = []
        return {"video_path": video_path}

    def add_new_points_or_box(
        self,
        inference_state: Any,
        frame_idx: int,
        obj_id: int,
        points: Any = None,
        labels: Any = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: Sequence[float] | None = None,
    ) -> tuple[int, list[int], Any]:
        assert box is not None
        self.calls.append((frame_idx, obj_id, tuple(float(v) for v in box)))
        self.obj_ids.append(obj_id)
        return frame_idx, list(self.obj_ids), np.zeros((len(self.obj_ids), 1, 1, 1))

    def frame_masks(self, index: int) -> dict[int, npt.NDArray[np.bool_]]:
        label = read_label_map(self.video.annotation_path(index))
        masks = label_map_to_masks(label)
        if self.overlap and 1 in masks and 2 in masks:
            masks[2] = masks[2] | masks[1]
        return {i: masks.get(i, np.zeros(label.shape, dtype=bool)) for i in self.obj_ids}

    def propagate_in_video(
        self,
        inference_state: Any,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], Any]]:
        for index in range(self.video.num_frames):
            masks = self.frame_masks(index)
            logits = np.stack([np.where(masks[i], POSITIVE, NEGATIVE) for i in self.obj_ids])
            yield index, list(self.obj_ids), logits[:, None].astype(np.float32)
