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

    def __init__(
        self,
        video: SyntheticVideo,
        overlap: bool = False,
        stored_scores: dict[tuple[int, int], Any] | None = None,
        positives: dict[int, float] | None = None,
    ) -> None:
        """stored_scores maps (obj_id, frame) to the value kept as object_score_logits.

        positives maps an object id to the logit of its foreground pixels (default POSITIVE).
        """
        self.video = video
        self.overlap = overlap
        self.stored_scores = stored_scores
        self.state: dict[str, Any] = {}
        self.calls: list[tuple[int, int, tuple[float, ...]]] = []
        self.init_path: str | None = None
        self.positives = positives or {}
        self.obj_ids: list[int] = []
        self.prompt_frames: dict[int, int] = {}
        self.starts: list[int | None] = []
        self.resets = 0
        self.offload_video: bool | None = None

    def init_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> Any:
        self.init_path = video_path
        self.offload_video = offload_video_to_cpu
        self.obj_ids = []
        self.prompt_frames = {}
        self.state = {
            "video_path": video_path,
            "obj_id_to_idx": {},
            "output_dict_per_obj": {},
        }
        return self.state

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
        self.prompt_frames[obj_id] = frame_idx
        idx = len(inference_state["obj_id_to_idx"])
        inference_state["obj_id_to_idx"][obj_id] = idx
        inference_state["output_dict_per_obj"][idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        return frame_idx, list(self.obj_ids), np.zeros((len(self.obj_ids), 1, 1, 1))

    def frame_masks(self, index: int) -> dict[int, npt.NDArray[np.bool_]]:
        label = read_label_map(self.video.annotation_path(index))
        masks = label_map_to_masks(label)
        if self.overlap and 1 in masks and 2 in masks:
            masks[2] = masks[2] | masks[1]
        return {i: masks.get(i, np.zeros(label.shape, dtype=bool)) for i in self.obj_ids}

    def reset_state(self, inference_state: Any) -> None:
        self.resets += 1
        self.obj_ids = []
        self.prompt_frames = {}
        inference_state["obj_id_to_idx"].clear()
        inference_state["output_dict_per_obj"].clear()

    def propagate_in_video(
        self,
        inference_state: Any,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], Any]]:
        self.starts.append(start_frame_idx)
        first = min(self.prompt_frames.values()) if start_frame_idx is None else start_frame_idx
        for index in range(first, self.video.num_frames):
            masks = self.frame_masks(index)
            self._store_scores(inference_state, index)
            logits = np.stack(
                [
                    np.where(masks[i], self.positives.get(i, POSITIVE), NEGATIVE)
                    for i in self.obj_ids
                ]
            )
            yield index, list(self.obj_ids), logits[:, None].astype(np.float32)

    def _store_scores(self, state: Any, index: int) -> None:
        if self.stored_scores is None:
            return
        for obj_id, idx in state["obj_id_to_idx"].items():
            key = (
                "cond_frame_outputs"
                if index == self.prompt_frames[obj_id]
                else "non_cond_frame_outputs"
            )
            value = self.stored_scores.get((obj_id, index))
            if isinstance(value, float):
                value = np.full((1, 1), value, dtype=np.float32)
            state["output_dict_per_obj"][idx][key][index] = {"object_score_logits": value}
