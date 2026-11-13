"""Streaming recovery loop for one video.

SAM 2 is propagated frame by frame. At each frame the detector looks at every object; a flagged
object gets a box from the box source, the box passes the sanity check, and SAM 2 is re-prompted
at that frame on a fresh inference state (docs/interfaces.md section 6): the initial prompts and
earlier recovery boxes are added again, frames before t are re-propagated and their outputs are
discarded (they were already emitted and stay as they were), the new boxes are added at t and
propagation continues from t. Objects accepted at the same frame share one restart.

A frame goes to the sink as soon as it is final; this module never writes files. Tracker loss,
rejected boxes and cooldown are recorded as events, never replaced by another box.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from src.common.boxes import Box
from src.recovery.box_check import BoxCheck
from src.recovery.events import Event
from src.recovery.settings import RecoverySettings
from src.recovery.sources import BoxSource, Detector, Frame
from src.sam2_runner.predictor import VideoPredictor, to_numpy
from src.sam2_runner.runner import sam2_box
from src.sam2_runner.scores import object_score

Logits = npt.NDArray[np.floating[Any]]
FrameReader = Callable[[int], Frame]
Sink = Callable[[int, dict[int, Logits], dict[int, float]], None]
Output = tuple[int, list[int], Any]


class RecoveryRun:
    def __init__(
        self,
        predictor: VideoPredictor,
        detector: Detector,
        boxes: BoxSource,
        box_check: BoxCheck,
        settings: RecoverySettings,
    ) -> None:
        self.predictor = predictor
        self.detector = detector
        self.boxes = boxes
        self.box_check = box_check
        self.settings = settings

    def run(
        self,
        video: str,
        frames_dir: str,
        num_frames: int,
        read_frame: FrameReader,
        prompts: Mapping[int, Box],
        start_frame: int,
        sink: Sink,
        offload_video: bool = False,
    ) -> list[Event]:
        """Emit frames start_frame..num_frames-1 to the sink; return the events in order."""
        if not prompts or not 0 <= start_frame < num_frames:
            raise ValueError(f"'{video}': need prompts and a start frame inside the video")
        self.detector.reset()
        self.boxes.reset()
        self._video, self._start, self._prompts = video, start_frame, dict(prompts)
        self._recoveries: list[tuple[int, int, Box]] = []
        self._last_reprompt: dict[int, int] = {}
        events: list[Event] = []
        state = self.predictor.init_state(frames_dir, offload_video_to_cpu=offload_video)
        self._add_prompts(state)
        stream = self.predictor.propagate_in_video(state, start_frame_idx=start_frame)
        for expected in range(start_frame, num_frames):
            item = next(stream, None)
            if item is None or item[0] != expected:
                raise RuntimeError(f"'{video}': expected frame {expected} from SAM 2")
            arrays, scores = self._unpack(state, item)
            frame = read_frame(expected)
            if expected == start_frame:
                self._start_objects(frame, expected, arrays, scores)
            else:
                accepted = self._decide(expected, frame, arrays, scores, events)
                if accepted:
                    stream = self._restart(state, expected, accepted)
                    arrays, scores = self._unpack(state, self._next(stream, expected))
                    self._after_recovery(frame, expected, arrays, scores, accepted)
            sink(expected, arrays, scores)
        return events

    def _unpack(self, state: Any, item: Output) -> tuple[dict[int, Logits], dict[int, float]]:
        frame_idx, obj_ids, logits = item
        arrays = {obj: to_numpy(logits[row, 0]) for row, obj in enumerate(obj_ids)}
        scores = {obj: object_score(state, obj, frame_idx, a)[0] for obj, a in arrays.items()}
        return arrays, scores

    def _next(self, stream: Iterator[Output], expected: int) -> Output:
        item = next(stream, None)
        if item is None or item[0] != expected:
            raise RuntimeError(f"'{self._video}': restart did not return frame {expected}")
        return item

    def _add_prompts(self, state: Any) -> None:
        for obj_id, box in self._prompts.items():
            self.predictor.add_new_points_or_box(state, self._start, obj_id, box=sam2_box(box))

    def _start_objects(
        self, frame: Frame, index: int, arrays: dict[int, Logits], scores: dict[int, float]
    ) -> None:
        for obj_id, logits in arrays.items():
            mask = logits > 0
            shape = (mask.shape[0], mask.shape[1])
            self.detector.observe(obj_id, index, mask, scores[obj_id], shape)
            self.boxes.start(obj_id, index, frame, mask)

    def _decide(
        self,
        index: int,
        frame: Frame,
        arrays: dict[int, Logits],
        scores: dict[int, float],
        events: list[Event],
    ) -> dict[int, Box]:
        accepted: dict[int, Box] = {}
        for obj_id, logits in arrays.items():
            mask = logits > 0
            shape = (mask.shape[0], mask.shape[1])
            decision = self.detector.update(obj_id, index, mask, scores[obj_id], shape)
            if not decision.flag:
                self.boxes.observe(obj_id, index, frame, mask)
                continue
            fired = decision.fired
            last = self._last_reprompt.get(obj_id)
            if last is not None and index - last < self.settings.cooldown_frames:
                events.append(Event(self._video, obj_id, index, fired, "cooldown"))
                continue
            result = self.boxes.propose(obj_id, index, frame)
            if result.status != "ok" or result.box is None:
                events.append(
                    Event(
                        self._video,
                        obj_id,
                        index,
                        fired,
                        "tracker_lost",
                        result.status,
                        result.raw_box,
                    )
                )
                continue
            check = self.box_check.check(
                result.raw_box, shape, self.boxes.last_reliable_box(obj_id)
            )
            if not check.ok:
                events.append(
                    Event(
                        self._video,
                        obj_id,
                        index,
                        fired,
                        "rejected",
                        check.reason or "",
                        result.raw_box,
                    )
                )
                continue
            accepted[obj_id] = result.box
            events.append(
                Event(
                    self._video,
                    obj_id,
                    index,
                    fired,
                    "reprompted",
                    "",
                    result.raw_box,
                    index - self._start,
                )
            )
        return accepted

    def _restart(self, state: Any, index: int, accepted: dict[int, Box]) -> Iterator[Output]:
        self.predictor.reset_state(state)
        self._add_prompts(state)
        for frame_idx, obj_id, box in self._recoveries:
            self.predictor.add_new_points_or_box(state, frame_idx, obj_id, box=sam2_box(box))
        replayed = self.predictor.propagate_in_video(
            state, start_frame_idx=self._start, max_frame_num_to_track=index - 1 - self._start
        )
        count = sum(1 for _ in replayed)
        if count != index - self._start:
            raise RuntimeError(
                f"'{self._video}': replay gave {count} frames, not {index - self._start}"
            )
        for obj_id, box in accepted.items():
            self.predictor.add_new_points_or_box(state, index, obj_id, box=sam2_box(box))
            self._recoveries.append((index, obj_id, box))
            self._last_reprompt[obj_id] = index
        return self.predictor.propagate_in_video(state, start_frame_idx=index)

    def _after_recovery(
        self,
        frame: Frame,
        index: int,
        arrays: dict[int, Logits],
        scores: dict[int, float],
        accepted: dict[int, Box],
    ) -> None:
        for obj_id in accepted:
            mask = arrays[obj_id] > 0
            shape = (mask.shape[0], mask.shape[1])
            self.detector.observe(obj_id, index, mask, scores[obj_id], shape)
            self.boxes.start(obj_id, index, frame, mask)
