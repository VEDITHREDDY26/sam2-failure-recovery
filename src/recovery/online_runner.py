"""Online runner: SAM 2 + detector + recovery on whole videos, resume-safe.

Same inputs as the SAM 2 alone runner (prompts file, frames folder) and the same output layout
under <output_root>/<method>/<split>/{masks,signals,events,timing}. A frame's label map is written
as soon as the recovery loop emits it; the timing, events and signal files are written last and
the signal file is the final, atomic write, so a killed job leaves the video incomplete and the
next run redoes only that video. SAM 2 alone and baseline method folders are refused, so cached
masks can never be written here.

compute_s is the loop time without frame decoding, PNG writing and SAM 2's init_state (the same
span the SAM 2 alone timing covers) but including the detector, the tracker and the restarts.
restart_s is the time inside restarts (reset and replay of the frames before the recovered frame).
"""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.common.config import DataConfig
from src.common.layout import events_path, mask_path, signals_path, validate_method
from src.common.masks import masks_to_label_map, write_label_map
from src.common.signals import SIGNAL_COLUMNS, read_signals, write_signals
from src.recovery.box_check import BoxCheck
from src.recovery.events import EVENT_COLUMNS, Event
from src.recovery.gate import BoxGate
from src.recovery.recovery import Logits, RecoveryRun
from src.recovery.settings import RecoverySettings
from src.recovery.sources import BoxSource, Detector, Frame
from src.recovery.verify import VerifyThresholds, candidate_path, write_candidates
from src.sam2_runner.predictor import VideoPredictor
from src.sam2_runner.prompts import BatchPrompts, Prompts
from src.sam2_runner.run import frames_dir_for
from src.sam2_runner.runner import (
    FRAME_SUFFIXES,
    TEMP_SUFFIX,
    clear_video,
    signal_row,
)
from src.sam2_runner.timing import TIMING_COLUMNS, timing_path, timing_row

REFUSED_PREFIXES = ("sam2", "baseline")
RESTART_COLUMNS = ("restarts", "restart_s")
ONLINE_TIMING_COLUMNS = TIMING_COLUMNS + RESTART_COLUMNS

PredictorFactory = Callable[[], VideoPredictor]
DetectorFactory = Callable[[str], Detector]
BoxSourceFactory = Callable[[str], BoxSource]


def check_online_method(method: str) -> str:
    """Refuse method names that belong to SAM 2 alone or baseline results.

    Args:
        method: Method (output folder) name.

    Returns:
        The same name.

    Raises:
        ValueError: If the name is invalid or starts with "sam2" or "baseline", so cached masks
            can never be overwritten.
    """
    validate_method(method)
    if method.startswith(REFUSED_PREFIXES):
        raise ValueError(
            f"method '{method}' is a SAM 2 alone or baseline folder; the online runner never "
            "writes there"
        )
    return method


def frame_paths(frames_dir: Path) -> list[Path]:
    """List the JPEG frames of a video folder in frame order.

    Args:
        frames_dir: Folder of frames named by their integer index.

    Returns:
        Frame paths sorted by the integer value of the file stem.

    Raises:
        FileNotFoundError: If the folder holds no JPEG frames.
    """
    paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in FRAME_SUFFIXES]
    if not paths:
        raise FileNotFoundError(f"no JPEG frames in {frames_dir}")
    return sorted(paths, key=lambda p: int(p.stem))


def _read_bgr(path: Path) -> Frame:
    """Read an image as a BGR uint8 array (OpenCV's order, requested explicitly).

    Raises:
        OSError: If the file cannot be read.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read frame {path}")
    return np.asarray(image, dtype=np.uint8)


def _synchronise() -> None:
    """Wait for queued GPU work to finish, so timings are real; does nothing without CUDA."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimedPredictor:
    """Passes every call through and times init_state, reset_state and the replay passes."""

    def __init__(self, inner: VideoPredictor) -> None:
        """Wrap a predictor and start the timers at zero."""
        self.inner = inner
        self.init_s = 0.0
        self.restart_s = 0.0
        self.restarts = 0

    def init_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> Any:
        """Time and forward init_state (frame loading), which is kept out of the compute time.

        Args:
            video_path: Folder of frames.
            offload_video_to_cpu: Keep frames on the CPU.
            offload_state_to_cpu: Keep the inference state on the CPU.
            async_loading_frames: Load frames in a background thread.

        Returns:
            The SAM 2 inference state.
        """
        tick = time.perf_counter()
        state = self.inner.init_state(
            video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        self.init_s += time.perf_counter() - tick
        return state

    def add_new_points_or_box(self, *args: Any, **kwargs: Any) -> Any:
        """Forward the prompt call unchanged."""
        return self.inner.add_new_points_or_box(*args, **kwargs)

    def reset_state(self, inference_state: Any) -> None:
        """Time and forward a state reset, and count it as one restart."""
        tick = time.perf_counter()
        self.inner.reset_state(inference_state)
        self.restart_s += time.perf_counter() - tick
        self.restarts += 1

    def propagate_in_video(
        self,
        inference_state: Any,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[Any]:
        """Forward propagation; passes bounded replays through the restart timer.

        Args:
            inference_state: SAM 2 inference state.
            start_frame_idx: 0-based frame to start from.
            max_frame_num_to_track: Number of frames to track; set only for the replay of frames
                before a recovered frame, which is therefore timed.
            reverse: Propagate backwards.

        Returns:
            Stream of (frame index, object ids, logits).
        """
        stream = self.inner.propagate_in_video(
            inference_state, start_frame_idx, max_frame_num_to_track, reverse
        )
        if max_frame_num_to_track is None:
            return stream
        return self._timed(stream)

    def _timed(self, stream: Iterator[Any]) -> Iterator[Any]:
        """Yield items of a replay stream, adding the time spent producing them to restart_s."""
        while True:
            # Only the time to produce each item counts; the consumer's time between items does not.
            tick = time.perf_counter()
            item = next(stream, None)
            if item is None:
                _synchronise()
            self.restart_s += time.perf_counter() - tick
            if item is None:
                return
            yield item


class _OnlineSink:
    """Frame sink of the loop: writes each final frame's label map and collects its signal rows.

    Attributes:
        seconds: Total time spent in the sink (PNG writing and signal rows).
    """

    def __init__(
        self,
        output_root: Path,
        method: str,
        split: str,
        video: str,
        stems: Sequence[str],
        obj_ids: Sequence[int],
        start_frame: int,
    ) -> None:
        """Prepare the sink for one video.

        Args:
            output_root: Root folder for run outputs.
            method: Method name.
            split: Split name.
            video: Video name.
            stems: File stem of every frame, in frame order.
            obj_ids: Object ids of the video.
            start_frame: 0-based prompt frame; earlier frames are blanked at the end.
        """
        self._target = (output_root, method, split, video)
        self._stems = stems
        self._video = video
        self._rows: list[Any] = []
        self._start = start_frame
        self._obj_ids = list(obj_ids)
        self._shape: tuple[int, int] | None = None
        self.seconds = 0.0

    def __call__(self, index: int, arrays: dict[int, Logits], scores: dict[int, float]) -> None:
        """Write one frame's label map and add its signal rows.

        Args:
            index: 0-based frame index.
            arrays: Logits per object id.
            scores: Object score per object id.
        """
        tick = time.perf_counter()
        first = next(iter(arrays.values()))
        shape = (int(first.shape[0]), int(first.shape[1]))
        self._shape = shape
        # Overlaps between objects go to the higher logit.
        label = masks_to_label_map({i: a > 0 for i, a in arrays.items()}, shape, scores=arrays)
        output_root, method, split, video = self._target
        write_label_map(mask_path(output_root, method, split, video, self._stems[index]), label)
        for obj_id, logits in arrays.items():
            self._rows.append(signal_row(self._video, obj_id, index, logits > 0, scores[obj_id]))
        self.seconds += time.perf_counter() - tick

    def blank_before_start(self) -> None:
        """Frames before the prompt frame: empty label map and NaN-score rows."""
        if self._shape is None:
            raise RuntimeError("SAM 2 returned no frames")
        output_root, method, split, video = self._target
        nan = float("nan")
        # Frames before the prompt frame get empty masks and NaN-score rows, so every frame has an
        # output.
        for index in range(self._start):
            blank = np.zeros(self._shape, dtype=np.uint8)
            write_label_map(mask_path(output_root, method, split, video, self._stems[index]), blank)
            for obj_id in self._obj_ids:
                self._rows.append((self._video, obj_id, index, nan, 0, nan, nan, True))

    def table(self) -> pd.DataFrame:
        """Signal rows of the whole video as a table sorted by object and frame."""
        table = pd.DataFrame(self._rows, columns=list(SIGNAL_COLUMNS))
        return table.sort_values(["object_id", "frame_index"], ignore_index=True)


@dataclass(frozen=True)
class OnlineResult:
    """What run_video returns.

    Attributes:
        table: Signal table of the video (one row per object and frame).
        events: Recovery events in the order they happened.
        restarts: Number of SAM 2 state resets done.
    """

    table: pd.DataFrame
    events: list[Event]
    restarts: int


def is_online_complete(
    output_root: Path,
    method: str,
    split: str,
    video: str,
    stems: Sequence[str],
    obj_ids: Sequence[int],
    with_candidates: bool = False,
) -> bool:
    """Valid signals (one row per object and frame), events, timing and every PNG exist."""
    csv_path = signals_path(output_root, method, split, video)
    if not (csv_path.is_file() and events_path(output_root, method, split, video).is_file()):
        return False
    if with_candidates and not candidate_path(output_root, method, split, video).is_file():
        return False
    if not timing_path(output_root, method, split, video).is_file():
        return False
    try:
        table = read_signals(csv_path)
    except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    # Complete means exactly one signal row per object and frame, plus a PNG for every frame.
    expected = {(i, f) for i in obj_ids for f in range(len(stems))}
    keys = set(zip(table["object_id"].tolist(), table["frame_index"].tolist(), strict=True))
    if len(table) != len(expected) or keys != expected or set(table["video"]) != {video}:
        return False
    return all(mask_path(output_root, method, split, video, s).is_file() for s in stems)


def _clear(output_root: Path, method: str, split: str, video: str) -> None:
    """Delete the earlier outputs of one video (masks, signals, timing, events, candidates)."""
    clear_video(output_root, method, split, video)
    events_path(output_root, method, split, video).unlink(missing_ok=True)
    candidate_path(output_root, method, split, video).unlink(missing_ok=True)


def _write_events(path: Path, events: Sequence[Event]) -> None:
    """Write the events of one video as CSV with the EVENT_COLUMNS header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([e.as_row() for e in events], columns=list(EVENT_COLUMNS)).to_csv(
        path, index=False
    )


def _write_timing(path: Path, row: dict[str, Any]) -> None:
    """Write the single timing row of one video as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=list(ONLINE_TIMING_COLUMNS)).to_csv(path, index=False)


def run_video(
    predictor: VideoPredictor,
    detector: Detector,
    boxes: BoxSource,
    box_check: BoxCheck,
    settings: RecoverySettings,
    frames_dir: Path,
    prompts: Prompts,
    output_root: Path,
    method: str,
    split: str,
    video: str,
    offload_video: bool = False,
    verifier: VerifyThresholds | None = None,
    log_candidates: bool = False,
    gate: BoxGate | None = None,
) -> OnlineResult:
    """Run one video; PNGs as frames arrive, then timing, events and (last) the signal CSV."""
    check_online_method(method)
    # Recovery works with one prompt frame per video.
    groups = prompts.groups()
    if len(groups) != 1:
        raise ValueError(f"'{video}': recovery needs one prompt frame, got {sorted(groups)}")
    start = next(iter(groups))
    wall_start = time.perf_counter()
    paths = frame_paths(frames_dir)
    stems = [p.stem for p in paths]
    if start >= len(stems):
        raise ValueError(f"'{video}': prompt frame {start} but only {len(stems)} frames")
    # Start clean: earlier partial outputs of this video are removed.
    _clear(output_root, method, split, video)
    timed = TimedPredictor(predictor)
    sink = _OnlineSink(output_root, method, split, video, stems, list(prompts.boxes), start)
    # `reading` accumulates time spent decoding frames so it can be left out of compute_s.
    reading = [0.0]

    def read_frame(index: int) -> Frame:
        """Read one frame as BGR and add the decoding time to the `reading` counter."""
        tick = time.perf_counter()
        frame = _read_bgr(paths[index])
        reading[0] += time.perf_counter() - tick
        return frame

    # The loop writes PNGs through the sink as frames become final.
    run = RecoveryRun(timed, detector, boxes, box_check, settings, verifier, log_candidates, gate)
    loop_start = time.perf_counter()
    events = run.run(
        video, str(frames_dir), len(stems), read_frame, prompts.boxes, start, sink, offload_video
    )
    loop_s = time.perf_counter() - loop_start
    # Fill in frames before the prompt frame, then build the signal table.
    sink.blank_before_start()
    table = sink.table()
    # Compute time excludes PNG writing, frame decoding and init_state, like SAM 2 alone.
    compute_s = loop_s - sink.seconds - reading[0] - timed.init_s
    wall_s = time.perf_counter() - wall_start
    row = timing_row(video, len(stems), len(prompts.boxes), compute_s, wall_s, offload_video)
    row.update(restarts=timed.restarts, restart_s=timed.restart_s)
    _write_timing(timing_path(output_root, method, split, video), row)
    _write_events(events_path(output_root, method, split, video), events)
    if verifier is not None or log_candidates:
        write_candidates(candidate_path(output_root, method, split, video), run.candidates)
    # Signal CSV last and atomically (temp file then rename): it marks the video complete.
    signals = signals_path(output_root, method, split, video)
    temp = signals.with_name(signals.name + TEMP_SUFFIX)
    write_signals(temp, table)
    os.replace(temp, signals)
    return OnlineResult(table, events, timed.restarts)


def run_batch(
    prompts: BatchPrompts,
    cfg: DataConfig,
    method: str,
    make_predictor: PredictorFactory,
    make_detector: DetectorFactory,
    make_boxes: BoxSourceFactory,
    box_check: BoxCheck,
    settings: RecoverySettings,
    videos: Sequence[str] | None = None,
    offload_video: bool = False,
    verifier: VerifyThresholds | None = None,
    log_candidates: bool = False,
    gate: BoxGate | None = None,
) -> tuple[int, int]:
    """Run every selected video that is not complete; return (run, skipped)."""
    check_online_method(method)
    names = list(videos) if videos else list(prompts.videos)
    unknown = [v for v in names if v not in prompts.videos]
    if unknown:
        raise ValueError(f"no prompts for videos {unknown}")
    output_root = cfg.data_root / cfg.output_root
    predictor: VideoPredictor | None = None
    ran = skipped = 0
    # Per video: skip if complete, else build fresh detector and box source and run.
    for video in names:
        entry = prompts.videos[video]
        frames_dir = frames_dir_for(cfg, prompts.split, video)
        stems = [p.stem for p in frame_paths(frames_dir)]
        if is_online_complete(
            output_root,
            method,
            prompts.split,
            video,
            stems,
            sorted(entry.boxes),
            verifier is not None or log_candidates,
        ):
            print(f"{video}: skipped (complete)")
            skipped += 1
            continue
        # The SAM 2 model is built once, on the first video that needs it.
        if predictor is None:
            predictor = make_predictor()
        result = run_video(
            predictor,
            make_detector(video),
            make_boxes(video),
            box_check,
            settings,
            frames_dir,
            entry,
            output_root,
            method,
            prompts.split,
            video,
            offload_video,
            verifier,
            log_candidates,
            gate,
        )
        recovered = sum(1 for e in result.events if e.action == "reprompted")
        print(
            f"{video}: done, {len(stems)} frames, {len(result.events)} events, "
            f"{recovered} re-prompts, {result.restarts} restarts"
        )
        ran += 1
    print(f"{ran} run, {skipped} skipped")
    return ran, skipped


__all__ = [
    "OnlineResult",
    "check_online_method",
    "is_online_complete",
    "run_batch",
    "run_video",
]
