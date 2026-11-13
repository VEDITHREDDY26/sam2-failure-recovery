"""E9: what the fix costs. Time per component of the real system on a seeded sample of videos.

    python -m experiments.e9.time_components --prompts prompts/davis_val.json \
        [--models tiny large] [--repeats 3] [--device cuda] [--out results/e9.csv]

The real system (rule detector + aligned CSRT + recovery) runs through RecoveryRun with thin
timing wrappers around the predictor, the detector, the box source and the box check. The
accelerator is synchronised before every start and stop of a timer when CUDA is there. One
warm-up pass over the sample is discarded, then `repeats` timed passes follow. Nothing is
written to the cached mask folders: the sink is empty and frames are only read.

Components: sam2_propagate (the main stream, one call per frame), sam2_prompt
(add_new_points_or_box), reprompt_restart (reset_state and the replay of the frames before
the recovered frame), detector,
csrt (box source), box_check, other (the rest of the loop: conversions, scores), init_state.
total_loop is the loop without init_state, frame reading and sink, as the E2 compute time; its
fps is frames / total_loop. Spread is the standard deviation over the timed passes.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import random
import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.common.config import DATA_ROOT_ENV, DataConfig
from src.recovery.box_check import BoxCheck, load_box_check_config
from src.recovery.config import load_recovery_config
from src.recovery.online_runner import check_online_method, frame_paths
from src.recovery.recovery import RecoveryRun
from src.recovery.settings import CONFIG_PATH, RecoverySettings, load_recovery_settings
from src.recovery.sources import BoxSource, CSRTBoxSource, Detector, Frame
from src.sam2_runner.predictor import MODELS, VideoPredictor, build_predictor
from src.sam2_runner.prompts import BatchPrompts, Prompts, load_batch_prompts
from src.sam2_runner.run import frames_dir_for
from src.sam2_runner.smoke import load_data_config
from src.sam2_runner.timing import device_fields

SEED = 5415
SAMPLE_SIZE = 3
DEFAULT_REPEATS = 3
DEFAULT_OUT = "results/e9.csv"
CONFIGS = {m: f"configs/recovery/rules_csrt_{m}.yaml" for m in MODELS}
COMPONENTS = (
    "sam2_propagate",
    "sam2_prompt",
    "reprompt_restart",
    "detector",
    "csrt",
    "box_check",
    "other",
    "init_state",
    "total_loop",
)
COLUMNS = (
    "model",
    "component",
    "split",
    "videos",
    "frames",
    "repeats",
    "calls",
    "ms_per_frame_mean",
    "ms_per_frame_std",
    "ms_per_call_mean",
    "ms_per_call_std",
    "fps_mean",
    "fps_std",
    "reprompts",
    "gpu_name",
    "torch_version",
    "cuda_version",
)


def synchronise() -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class Clock:
    """Seconds and call counts per component; every timer start and stop is synchronised."""

    seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    def timed(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        synchronise()
        tick = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            synchronise()
            self.add(name, time.perf_counter() - tick)

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds
        self.calls[name] = self.calls.get(name, 0) + 1


class TimingPredictor:
    def __init__(self, inner: VideoPredictor, clock: Clock) -> None:
        self.inner, self.clock = inner, clock

    def init_state(self, video_path: str, **kwargs: Any) -> Any:
        return self.clock.timed("init_state", self.inner.init_state, video_path, **kwargs)

    def add_new_points_or_box(self, *args: Any, **kwargs: Any) -> Any:
        return self.clock.timed("sam2_prompt", self.inner.add_new_points_or_box, *args, **kwargs)

    def reset_state(self, inference_state: Any) -> None:
        self.clock.timed("reprompt_restart", self.inner.reset_state, inference_state)

    def propagate_in_video(
        self,
        inference_state: Any,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[Any]:
        stream = self.inner.propagate_in_video(
            inference_state, start_frame_idx, max_frame_num_to_track, reverse
        )
        replay = max_frame_num_to_track is not None
        return self._stream(stream, "reprompt_restart" if replay else "sam2_propagate", replay)

    def _stream(self, stream: Iterator[Any], name: str, replay: bool) -> Iterator[Any]:
        """Replay frames add seconds only: the restart is counted once, in reset_state."""
        while True:
            synchronise()
            tick = time.perf_counter()
            item = next(stream, None)
            synchronise()
            spent = time.perf_counter() - tick
            if replay:
                self.clock.seconds[name] = self.clock.seconds.get(name, 0.0) + spent
            elif item is not None:
                self.clock.add(name, spent)
            if item is None:
                return
            yield item


class TimingDetector:
    def __init__(self, inner: Detector, clock: Clock) -> None:
        self.inner, self.clock = inner, clock

    def reset(self) -> None:
        self.inner.reset()

    def update(self, *args: Any, **kwargs: Any) -> Any:
        return self.clock.timed("detector", self.inner.update, *args, **kwargs)

    def observe(self, *args: Any, **kwargs: Any) -> None:
        self.clock.timed("detector", self.inner.observe, *args, **kwargs)


class TimingBoxes:
    def __init__(self, inner: BoxSource, clock: Clock) -> None:
        self.inner, self.clock = inner, clock

    def reset(self) -> None:
        self.inner.reset()

    def start(self, *args: Any, **kwargs: Any) -> None:
        self.clock.timed("csrt", self.inner.start, *args, **kwargs)

    def observe(self, *args: Any, **kwargs: Any) -> None:
        self.clock.timed("csrt", self.inner.observe, *args, **kwargs)

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        return self.clock.timed("csrt", self.inner.propose, *args, **kwargs)

    def last_reliable_box(self, obj_id: int) -> Any:
        return self.inner.last_reliable_box(obj_id)


class TimingBoxCheck:
    def __init__(self, inner: BoxCheck, clock: Clock) -> None:
        self.inner, self.clock = inner, clock

    def check(self, *args: Any, **kwargs: Any) -> Any:
        return self.clock.timed("box_check", self.inner.check, *args, **kwargs)


@dataclass(frozen=True)
class VideoTiming:
    frames: int
    reprompts: int
    seconds: dict[str, float]
    calls: dict[str, int]


def time_video(
    predictor: VideoPredictor,
    detector: Detector,
    boxes: BoxSource,
    box_check: BoxCheck,
    settings: RecoverySettings,
    frames_dir: Path,
    prompts: Prompts,
) -> VideoTiming:
    """One timed pass over a video; frames are decoded up front and never counted."""
    groups = prompts.groups()
    if len(groups) != 1:
        raise ValueError(f"{frames_dir.name}: recovery needs one prompt frame")
    start = next(iter(groups))
    frames: list[Frame] = []
    for path in frame_paths(frames_dir):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read frame {path}")
        frames.append(np.asarray(image, dtype=np.uint8))
    clock = Clock()
    run = RecoveryRun(
        TimingPredictor(predictor, clock),
        TimingDetector(detector, clock),
        TimingBoxes(boxes, clock),
        TimingBoxCheck(box_check, clock),  # type: ignore[arg-type]
        settings,
    )
    synchronise()
    tick = time.perf_counter()
    events = run.run(
        frames_dir.name,
        str(frames_dir),
        len(frames),
        lambda index: frames[index],
        prompts.boxes,
        start,
        lambda index, arrays, scores: None,
    )
    synchronise()
    loop = time.perf_counter() - tick - clock.seconds.get("init_state", 0.0)
    parts = sum(clock.seconds.get(c, 0.0) for c in COMPONENTS[:6])
    seconds = {c: clock.seconds.get(c, 0.0) for c in COMPONENTS[:6] + ("init_state",)}
    seconds["other"] = loop - parts
    seconds["total_loop"] = loop
    calls = {c: clock.calls.get(c, 0) for c in COMPONENTS[:6] + ("init_state",)}
    calls["other"] = calls["total_loop"] = len(frames) - start
    return VideoTiming(
        len(frames) - start, sum(e.action == "reprompted" for e in events), seconds, calls
    )


def sample_videos(names: Sequence[str], size: int = SAMPLE_SIZE) -> list[str]:
    pool = sorted(names)
    if size > len(pool):
        raise ValueError(f"cannot sample {size} of {len(pool)} videos")
    return sorted(random.Random(SEED).sample(pool, size))


def _mean(values: Sequence[float]) -> float:
    return float("nan") if any(math.isnan(v) for v in values) else statistics.mean(values)


def _spread(values: Sequence[float]) -> float:
    if any(math.isnan(v) for v in values):
        return float("nan")
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarise(
    model: str, split: str, videos: Sequence[str], passes: Sequence[Sequence[VideoTiming]]
) -> list[dict[str, Any]]:
    """One row per component: per-pass totals over the sampled videos, mean and std over passes."""
    device = device_fields()
    frames = [sum(v.frames for v in p) for p in passes]
    rows: list[dict[str, Any]] = []
    for component in COMPONENTS:
        secs = [sum(v.seconds[component] for v in p) for p in passes]
        calls = [sum(v.calls[component] for v in p) for p in passes]
        per_frame = [1000 * s / f for s, f in zip(secs, frames, strict=True)]
        per_call = [1000 * s / c if c else float("nan") for s, c in zip(secs, calls, strict=True)]
        timed_fps = component == "total_loop"
        fps = [f / s for f, s in zip(frames, secs, strict=True)] if timed_fps else []
        rows.append(
            {
                "model": model,
                "component": component,
                "split": split,
                "videos": ";".join(videos),
                "frames": frames[0],
                "repeats": len(passes),
                "calls": statistics.mean(calls),
                "ms_per_frame_mean": statistics.mean(per_frame),
                "ms_per_frame_std": _spread(per_frame),
                "ms_per_call_mean": _mean(per_call),
                "ms_per_call_std": _spread(per_call),
                "fps_mean": statistics.mean(fps) if fps else float("nan"),
                "fps_std": _spread(fps) if fps else float("nan"),
                "reprompts": statistics.mean(sum(v.reprompts for v in p) for p in passes),
                "gpu_name": device["gpu_name"],
                "torch_version": device["torch_version"],
                "cuda_version": device["cuda_version"],
            }
        )
    return rows


def measure_model(
    model: str,
    prompts: BatchPrompts,
    cfg: DataConfig,
    make_predictor: Callable[[], VideoPredictor],
    make_detector: Callable[[], Detector],
    make_boxes: Callable[[], BoxSource],
    box_check: BoxCheck,
    settings: RecoverySettings,
    videos: Sequence[str],
    repeats: int,
) -> list[dict[str, Any]]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    predictor = make_predictor()

    def one_pass() -> list[VideoTiming]:
        return [
            time_video(
                predictor,
                make_detector(),
                make_boxes(),
                box_check,
                settings,
                frames_dir_for(cfg, prompts.split, video),
                prompts.videos[video],
            )
            for video in videos
        ]

    one_pass()
    passes = [one_pass() for _ in range(repeats)]
    return summarise(model, prompts.split, videos, passes)


def run_model(
    model: str,
    args: argparse.Namespace,
    data: DataConfig,
    prompts: BatchPrompts,
    videos: Sequence[str],
    settings: RecoverySettings,
    box_check: BoxCheck,
) -> list[dict[str, Any]]:
    from src.detection.online import OnlineRuleDetector
    from src.detection.rules import DEFAULT_RULES_PATH

    run_cfg = load_recovery_config(Path(CONFIGS[model]))
    check_online_method(run_cfg.method)
    return measure_model(
        model,
        prompts,
        data,
        lambda: build_predictor(model, data.data_root, device=args.device),
        lambda: OnlineRuleDetector.from_config(model, DEFAULT_RULES_PATH, run_cfg.enabled_rules),
        lambda: CSRTBoxSource(settings.tracker_mode),
        box_check,
        settings,
        videos,
        args.repeats,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    args = parser.parse_args(argv)
    from src.detection.rules import DEFAULT_RULES_PATH

    if not DEFAULT_RULES_PATH.is_file():
        raise SystemExit(f"detector not frozen yet: {DEFAULT_RULES_PATH} does not exist")
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    data = load_data_config()
    prompts = load_batch_prompts(args.prompts)
    videos = sample_videos(list(prompts.videos))
    settings = load_recovery_settings(CONFIG_PATH)
    box_check = BoxCheck(load_box_check_config())
    rows: list[dict[str, Any]] = []
    for model in args.models:
        print(f"{model}: videos {videos}, {args.repeats} repeats after one warm-up pass")
        rows += run_model(model, args, data, prompts, videos, settings, box_check)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(COLUMNS)).to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
