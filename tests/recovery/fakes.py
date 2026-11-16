"""Stateful fake SAM 2 predictor, scripted detector and scripted box source for recovery tests.

The fake's mask for object k at frame f is a square at the box of k's most recent prompt at or
before f, shifted right by f - prompt_frame pixels, so a re-prompt at t changes frames >= t only.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from src.common.boxes import Box, box_from_mask
from src.recovery.sources import Decision
from src.recovery.tracker import TrackResult

H, W = 64, 64
NUM_FRAMES = 30


def expected_mask(box: Box, prompt_frame: int, frame: int) -> npt.NDArray[np.bool_]:
    mask = np.zeros((H, W), dtype=bool)
    shift = frame - prompt_frame
    mask[box.y0 : box.y1, box.x0 + shift : min(box.x1 + shift, W)] = True
    return mask


class StatefulFake:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.state: dict[str, Any] = {}

    def init_state(self, video_path: str, **kwargs: Any) -> Any:
        self.calls.append(("init",))
        self.state = {"obj_id_to_idx": {}, "output_dict_per_obj": {}, "prompts": {}}
        return self.state

    def reset_state(self, inference_state: Any) -> None:
        self.calls.append(("reset",))
        inference_state["obj_id_to_idx"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["prompts"].clear()

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
        self.calls.append(("add", frame_idx, obj_id, tuple(int(v) for v in box)))
        inference_state["obj_id_to_idx"].setdefault(obj_id, len(inference_state["obj_id_to_idx"]))
        inference_state["prompts"].setdefault(obj_id, []).append(
            (frame_idx, Box(*(int(v) for v in box)))
        )
        return frame_idx, list(inference_state["obj_id_to_idx"]), None

    def propagate_in_video(
        self,
        inference_state: Any,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], Any]]:
        self.calls.append(("propagate", start_frame_idx, max_frame_num_to_track))
        assert start_frame_idx is not None
        end = NUM_FRAMES - 1
        if max_frame_num_to_track is not None:
            end = min(end, start_frame_idx + max_frame_num_to_track)
        for frame in range(start_frame_idx, end + 1):
            ids = list(inference_state["obj_id_to_idx"])
            masks = [self._mask(inference_state, obj, frame) for obj in ids]
            logits = np.where(np.stack(masks), 4.0, -4.0)[:, None].astype(np.float32)
            yield frame, ids, logits

    def _mask(self, state: Any, obj_id: int, frame: int) -> npt.NDArray[np.bool_]:
        before = [(f, b) for f, b in state["prompts"][obj_id] if f <= frame]
        if not before:
            return np.zeros((H, W), dtype=bool)
        prompt_frame, box = max(before, key=lambda item: item[0])
        return expected_mask(box, prompt_frame, frame)


class ScriptedDetector:
    def __init__(self, flags: dict[tuple[int, int], tuple[str, ...]] | None = None) -> None:
        self.flags = flags or {}
        self.updates: list[tuple[int, int]] = []
        self.observed: list[tuple[int, int, npt.NDArray[np.bool_]]] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def update(
        self,
        obj_id: int,
        frame_index: int,
        mask: npt.NDArray[np.bool_],
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> Decision:
        self.updates.append((obj_id, frame_index))
        fired = self.flags.get((obj_id, frame_index))
        return Decision(fired is not None, fired or ())

    def observe(
        self,
        obj_id: int,
        frame_index: int,
        mask: npt.NDArray[np.bool_],
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> None:
        self.observed.append((obj_id, frame_index, mask))


class ScriptedBoxes:
    """Returns the scripted track result for (object, frame); last reliable box from masks."""

    def __init__(self, results: dict[tuple[int, int], TrackResult] | None = None) -> None:
        self.results = results or {}
        self.proposed: list[tuple[int, int]] = []
        self.started: list[tuple[int, int]] = []
        self._last: dict[int, Box] = {}

    def reset(self) -> None:
        self._last.clear()

    def start(self, obj_id: int, frame_index: int, frame: Any, mask: Any) -> None:
        self.started.append((obj_id, frame_index))
        self._remember(obj_id, mask)

    def observe(self, obj_id: int, frame_index: int, frame: Any, mask: Any) -> None:
        self._remember(obj_id, mask)

    def propose(self, obj_id: int, frame_index: int, frame: Any) -> TrackResult:
        self.proposed.append((obj_id, frame_index))
        return self.results[(obj_id, frame_index)]

    def last_reliable_box(self, obj_id: int) -> Box | None:
        return self._last.get(obj_id)

    def _remember(self, obj_id: int, mask: Any) -> None:
        box = box_from_mask(mask)
        if box is not None:
            self._last[obj_id] = box


def ok(box: Box) -> TrackResult:
    return TrackResult("ok", box, box)
