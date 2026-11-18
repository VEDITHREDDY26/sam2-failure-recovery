"""Streaming recovery loop for one video.

SAM 2 is propagated frame by frame. At each frame the detector looks at every object; a flagged
object gets a box from the box source, the box passes the sanity check, and SAM 2 is re-prompted
at that frame on a fresh inference state (the re-prompt contract): the initial prompts and
earlier recovery boxes are added again, frames before t are re-propagated and their outputs are
discarded (they were already emitted and stay as they were), the new boxes are added at t and
propagation continues from t. Objects accepted at the same frame share one restart.

A frame goes to the sink as soon as it is final; this module never writes files. Tracker loss,
rejected boxes and cooldown are recorded as events, never replaced by another box.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from src.common.boxes import Box, box_from_mask
from src.recovery.box_check import BoxCheck
from src.recovery.events import Event
from src.recovery.gate import BoxGate
from src.recovery.settings import RecoverySettings
from src.recovery.sources import BoxSource, Detector, Frame
from src.recovery.verify import CandidateRow, VerifyThresholds, candidate_features
from src.sam2_runner.predictor import VideoPredictor, to_numpy
from src.sam2_runner.runner import sam2_box
from src.sam2_runner.scores import object_score

Logits = npt.NDArray[np.floating[Any]]
FrameReader = Callable[[int], Frame]
Sink = Callable[[int, dict[int, Logits], dict[int, float]], None]
Output = tuple[int, list[int], Any]


class RecoveryRun:
    """Streaming recovery loop for one video at a time (see the module docstring for the contract).

    Attributes:
        predictor: SAM 2 video predictor (possibly a timing wrapper around it).
        detector: Failure detector fed SAM 2's own outputs.
        boxes: Source of prompt boxes for flagged frames.
        box_check: Sanity check applied to every proposed box.
        settings: Cooldown and tracker settings.
        verifier: Frozen thresholds for verified recovery, or None.
        log_candidates: If True, every re-prompt candidate is logged in `candidates`.
        gate: Box gate, or None when it is absent or both of its tests are off.
        candidates: Candidate rows logged during the current run.
    """

    def __init__(
        self,
        predictor: VideoPredictor,
        detector: Detector,
        boxes: BoxSource,
        box_check: BoxCheck,
        settings: RecoverySettings,
        verifier: VerifyThresholds | None = None,
        log_candidates: bool = False,
        gate: BoxGate | None = None,
    ) -> None:
        """Store the parts of the loop.

        Args:
            predictor: SAM 2 video predictor.
            detector: Failure detector.
            boxes: Box source.
            box_check: Box sanity check.
            settings: Recovery settings.
            verifier: Thresholds that accept or reject a re-prompt candidate, or None to accept all
                boxes that pass the check.
            log_candidates: Log candidate features even without a verifier.
            gate: Optional gate applied before the box check; an inactive gate is dropped.
        """
        self.gate = gate if gate is not None and gate.active else None
        self.verifier = verifier
        self.log_candidates = log_candidates
        self.candidates: list[CandidateRow] = []
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
        # Fresh detector and box-source state for every video.
        self.detector.reset()
        self.boxes.reset()
        self._video, self._start, self._prompts = video, start_frame, dict(prompts)
        self._recoveries: list[tuple[int, int, Box]] = []
        self._last_reprompt: dict[int, int] = {}
        self._reliable_area: dict[int, int] = {}
        self.candidates = []
        events: list[Event] = []
        # Initial state: add the prompts and start propagating from the prompt frame.
        state = self.predictor.init_state(frames_dir, offload_video_to_cpu=offload_video)
        self._add_prompts(state)
        stream = self.predictor.propagate_in_video(state, start_frame_idx=start_frame)
        # One pass over the frames; each frame is sent to the sink as soon as it is final.
        for expected in range(start_frame, num_frames):
            item = next(stream, None)
            if item is None or item[0] != expected:
                raise RuntimeError(f"'{video}': expected frame {expected} from SAM 2")
            arrays, scores = self._unpack(state, item)
            frame = read_frame(expected)
            # The prompt frame only seeds the detector and tracker; later frames are checked for
            # failure.
            if expected == start_frame:
                self._start_objects(frame, expected, arrays, scores)
            else:
                accepted = self._decide(expected, frame, arrays, scores, events)
                # Verified or logged re-prompts: restart, judge the candidate, and restart again if
                # some are rejected.
                if accepted and (self.verifier is not None or self.log_candidates):
                    stream, arrays, scores, accepted = self._restart_checked(
                        state, expected, accepted, arrays, scores, events
                    )
                    self._after_recovery(frame, expected, arrays, scores, accepted)
                # Plain re-prompt: restart at this frame and use the new output for it.
                elif accepted:
                    stream = self._restart(state, expected, accepted)
                    self._commit(expected, accepted)
                    arrays, scores = self._unpack(state, self._next(stream, expected))
                    self._after_recovery(frame, expected, arrays, scores, accepted)
            sink(expected, arrays, scores)
        return events

    def _unpack(self, state: Any, item: Output) -> tuple[dict[int, Logits], dict[int, float]]:
        """Turn one SAM 2 output into per-object logits and object scores.

        Args:
            state: SAM 2 inference state.
            item: (frame index, object ids, logits) as yielded by propagation.

        Returns:
            (logits per object id as HxW float arrays, object score per object id).
        """
        frame_idx, obj_ids, logits = item
        arrays = {obj: to_numpy(logits[row, 0]) for row, obj in enumerate(obj_ids)}
        scores = {obj: object_score(state, obj, frame_idx, a)[0] for obj, a in arrays.items()}
        return arrays, scores

    def _next(self, stream: Iterator[Output], expected: int) -> Output:
        """Take the next item of a restarted stream and check that it is the expected frame.

        Args:
            stream: Stream returned by a restart.
            expected: 0-based index of the frame that the restart must give first.

        Returns:
            The stream item for that frame.

        Raises:
            RuntimeError: If the stream is empty or gives another frame.
        """
        item = next(stream, None)
        if item is None or item[0] != expected:
            raise RuntimeError(f"'{self._video}': restart did not return frame {expected}")
        return item

    def _add_prompts(self, state: Any) -> None:
        """Add the first-frame box prompts of every object to a fresh inference state."""
        for obj_id, box in self._prompts.items():
            self.predictor.add_new_points_or_box(state, self._start, obj_id, box=sam2_box(box))

    def _start_objects(
        self, frame: Frame, index: int, arrays: dict[int, Logits], scores: dict[int, float]
    ) -> None:
        """Start the detector history and box source from the prompt frame's masks.

        Args:
            frame: BGR image of the prompt frame.
            index: 0-based index of the prompt frame.
            arrays: Logits per object id.
            scores: Object score per object id.
        """
        for obj_id, logits in arrays.items():
            mask = logits > 0
            shape = (mask.shape[0], mask.shape[1])
            self.detector.observe(obj_id, index, mask, scores[obj_id], shape)
            self.boxes.start(obj_id, index, frame, mask)
            self._reliable_area[obj_id] = int(np.count_nonzero(mask))

    def _decide(
        self,
        index: int,
        frame: Frame,
        arrays: dict[int, Logits],
        scores: dict[int, float],
        events: list[Event],
    ) -> dict[int, Box]:
        """Run the detector on every object at a frame and collect the boxes to re-prompt.

        Each flagged object ends in exactly one event: cooldown, tracker_lost, gated, rejected or
        reprompted. Nothing is ever replaced by another box.

        Args:
            index: 0-based frame index.
            frame: BGR image of this frame.
            arrays: Logits per object id.
            scores: Object score per object id.
            events: List that receives the events (appended in order).

        Returns:
            Accepted box (clipped, xyxy pixels) per object id to re-prompt at this frame.
        """
        accepted: dict[int, Box] = {}
        for obj_id, logits in arrays.items():
            # SAM 2 logits above zero are the foreground mask.
            mask = logits > 0
            shape = (mask.shape[0], mask.shape[1])
            decision = self.detector.update(obj_id, index, mask, scores[obj_id], shape)
            # Unflagged frames are reliable: the box source and reliable area follow them.
            if not decision.flag:
                self.boxes.observe(obj_id, index, frame, mask)
                self._reliable_area[obj_id] = int(np.count_nonzero(mask))
                continue
            fired = decision.fired
            # Cooldown: no new re-prompt within cooldown_frames of the last one for this object.
            last = self._last_reprompt.get(obj_id)
            if last is not None and index - last < self.settings.cooldown_frames:
                events.append(Event(self._video, obj_id, index, fired, "cooldown"))
                self.boxes.skip(obj_id, index, frame)
                continue
            # Ask the box source; a lost tracker is recorded, never replaced by another box.
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
            # Gate first (tracker score, agreement with SAM 2), then the box sanity check.
            if self.gate is not None:
                why = self.gate.reason(result, box_from_mask(mask))
                if why is not None:
                    events.append(
                        Event(self._video, obj_id, index, fired, "gated", why, result.raw_box)
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
        """Rebuild the SAM 2 state, replay the frames before this one, prompt it and restart there.

        The initial prompts and earlier recovery boxes are added again so frames before the index
        are re-propagated exactly as before (their outputs are discarded, never written).

        Args:
            state: SAM 2 inference state, reset in place.
            index: 0-based frame where the new boxes are added.
            accepted: New box per object id to add at that frame.

        Returns:
            Stream of propagation from that frame on.

        Raises:
            RuntimeError: If the replay does not give every earlier frame.
        """
        self.predictor.reset_state(state)
        self._add_prompts(state)
        for frame_idx, obj_id, box in self._recoveries:
            self.predictor.add_new_points_or_box(state, frame_idx, obj_id, box=sam2_box(box))
        # Replay frames before the index to rebuild SAM 2's memory; outputs are counted and dropped.
        replayed = self.predictor.propagate_in_video(
            state, start_frame_idx=self._start, max_frame_num_to_track=index - 1 - self._start
        )
        count = sum(1 for _ in replayed)
        if count != index - self._start:
            raise RuntimeError(
                f"'{self._video}': replay gave {count} frames, not {index - self._start}"
            )
        # Now add the recovery boxes at the index and propagate from there.
        for obj_id, box in accepted.items():
            self.predictor.add_new_points_or_box(state, index, obj_id, box=sam2_box(box))
        return self.predictor.propagate_in_video(state, start_frame_idx=index)

    def _commit(self, index: int, accepted: dict[int, Box]) -> None:
        """Remember accepted re-prompts: keep boxes for later replays and start the cooldown."""
        for obj_id, box in accepted.items():
            self._recoveries.append((index, obj_id, box))
            self._last_reprompt[obj_id] = index

    def _restart_checked(
        self,
        state: Any,
        index: int,
        candidates: dict[int, Box],
        flagged: dict[int, Logits],
        flagged_scores: dict[int, float],
        events: list[Event],
    ) -> tuple[Iterator[Output], dict[int, Logits], dict[int, float], dict[int, Box]]:
        """Restart with every candidate box, judge each, restart again if some are rejected."""
        stream = self._restart(state, index, candidates)
        arrays, scores = self._unpack(state, self._next(stream, index))
        # Keep the candidates that pass; rejected ones still start the cooldown.
        passed: dict[int, Box] = {}
        for obj_id, box in candidates.items():
            failed = self._judge(obj_id, index, box, flagged, flagged_scores, arrays, scores)
            if failed:
                self._last_reprompt[obj_id] = index
                self._mark_rejected(events, obj_id, index, ",".join(failed))
            else:
                passed[obj_id] = box
        self._commit(index, passed)
        if len(passed) == len(candidates):
            return stream, arrays, scores, passed
        # Some candidates failed: restart again with only the passed boxes.
        stream = self._restart(state, index, passed)
        arrays, scores = self._unpack(state, self._next(stream, index))
        return stream, arrays, scores, passed

    def _judge(
        self,
        obj_id: int,
        index: int,
        box: Box,
        flagged: dict[int, Logits],
        flagged_scores: dict[int, float],
        arrays: dict[int, Logits],
        scores: dict[int, float],
    ) -> list[str]:
        """Judge one candidate with the verification features and log it.

        Args:
            obj_id: Object id.
            index: 0-based frame index of the re-prompt.
            box: Prompt box that produced the candidate.
            flagged: Logits per object id of the original (flagged) output.
            flagged_scores: Object score per object id of the original output.
            arrays: Logits per object id of the restarted output.
            scores: Object score per object id of the restarted output.

        Returns:
            Names of the failed checks; empty when accepted or when there is no verifier.
        """
        ratio, overlap, same = candidate_features(
            arrays[obj_id] > 0, flagged[obj_id] > 0, box, self._reliable_area.get(obj_id, 0)
        )
        # Features come from SAM 2 outputs at this frame only; without a verifier nothing fails.
        failed = self.verifier.failed(scores[obj_id], ratio, overlap) if self.verifier else []
        self.candidates.append(
            CandidateRow(
                self._video,
                obj_id,
                index,
                scores[obj_id],
                flagged_scores[obj_id],
                ratio,
                overlap,
                same,
                not failed,
                ",".join(failed),
            )
        )
        return failed

    @staticmethod
    def _mark_rejected(events: list[Event], obj_id: int, index: int, reason: str) -> None:
        """Change the object's "reprompted" event at this frame into "verify_rejected".

        Args:
            events: Event list of the run.
            obj_id: Object id.
            index: 0-based frame index.
            reason: Comma-separated names of the failed checks.

        Raises:
            RuntimeError: If there is no matching re-prompt event.
        """
        # Search from the newest event backwards.
        for position in range(len(events) - 1, -1, -1):
            event = events[position]
            if (event.object_id, event.frame_index, event.action) == (obj_id, index, "reprompted"):
                events[position] = dataclasses.replace(
                    event, action="verify_rejected", reason=reason
                )
                return
        raise RuntimeError(f"no re-prompt event for object {obj_id} at frame {index}")

    def _after_recovery(
        self,
        frame: Frame,
        index: int,
        arrays: dict[int, Logits],
        scores: dict[int, float],
        accepted: dict[int, Box],
    ) -> None:
        """Reset the detector history and box source on the recovered frame's masks.

        Args:
            frame: BGR image of the recovered frame.
            index: 0-based frame index.
            arrays: Logits per object id of the final output at this frame.
            scores: Object score per object id.
            accepted: Boxes that were accepted at this frame (only these objects are reset).
        """
        for obj_id in accepted:
            mask = arrays[obj_id] > 0
            shape = (mask.shape[0], mask.shape[1])
            self.detector.observe(obj_id, index, mask, scores[obj_id], shape)
            self.boxes.start(obj_id, index, frame, mask)
            self._reliable_area[obj_id] = int(np.count_nonzero(mask))
