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
    """Wait for pending accelerator work so a timer reads the true time (no-op without CUDA)."""
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
        """Call a function and add its wall time to a component.

        Args:
            name: Component to charge.
            fn: Function to call.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            What fn returns. The time is recorded also when fn raises.
        """
        synchronise()
        tick = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            synchronise()
            self.add(name, time.perf_counter() - tick)

    def add(self, name: str, seconds: float) -> None:
        """Add seconds and one call to a component."""
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds
        self.calls[name] = self.calls.get(name, 0) + 1


class TimingPredictor:
    """Wrapper around the SAM 2 predictor that times the calls the recovery loop makes.

    It offers the same methods as the predictor and passes every call through unchanged.
    """

    def __init__(self, inner: VideoPredictor, clock: Clock) -> None:
        """Keep the wrapped predictor and the clock to charge."""
        self.inner, self.clock = inner, clock

    def init_state(self, video_path: str, **kwargs: Any) -> Any:
        """Load the video into the predictor; timed as init_state."""
        return self.clock.timed("init_state", self.inner.init_state, video_path, **kwargs)

    def add_new_points_or_box(self, *args: Any, **kwargs: Any) -> Any:
        """Give SAM 2 a prompt; timed as sam2_prompt."""
        return self.clock.timed("sam2_prompt", self.inner.add_new_points_or_box, *args, **kwargs)

    def reset_state(self, inference_state: Any) -> None:
        """Restart the predictor state; timed as reprompt_restart."""
        self.clock.timed("reprompt_restart", self.inner.reset_state, inference_state)

    def propagate_in_video(
        self,
        inference_state: Any,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[Any]:
        """Propagate masks and time the stream.

        A call with a frame limit is the replay of the frames before the recovered frame and is
        charged to reprompt_restart; any other call is the main stream, charged to sam2_propagate.

        Args:
            inference_state: Predictor state.
            start_frame_idx: First frame to propagate from.
            max_frame_num_to_track: Frame limit; its presence marks a replay.
            reverse: Whether to go backwards.

        Returns:
            An iterator of the predictor outputs.
        """
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
            # Time only the work done to produce the next item, with the GPU
            # synchronised before and after.
            item = next(stream, None)
            synchronise()
            spent = time.perf_counter() - tick
            # During a replay the time is added directly; otherwise only real items are counted.
            if replay:
                self.clock.seconds[name] = self.clock.seconds.get(name, 0.0) + spent
            elif item is not None:
                self.clock.add(name, spent)
            if item is None:
                return
            yield item


class TimingDetector:
    """Wrapper around the failure detector that times update and observe as detector."""

    def __init__(self, inner: Detector, clock: Clock) -> None:
        """Keep the wrapped detector and the clock to charge."""
        self.inner, self.clock = inner, clock

    def reset(self) -> None:
        """Reset the detector (not timed)."""
        self.inner.reset()

    def update(self, *args: Any, **kwargs: Any) -> Any:
        """Run the detector update; timed as detector."""
        return self.clock.timed("detector", self.inner.update, *args, **kwargs)

    def observe(self, *args: Any, **kwargs: Any) -> None:
        """Show the detector a frame; timed as detector."""
        self.clock.timed("detector", self.inner.observe, *args, **kwargs)


class TimingBoxes:
    """Wrapper around the box source (CSRT) that times its calls as csrt."""

    def __init__(self, inner: BoxSource, clock: Clock) -> None:
        """Keep the wrapped box source and the clock to charge."""
        self.inner, self.clock = inner, clock

    def reset(self) -> None:
        """Reset the box source (not timed)."""
        self.inner.reset()

    def start(self, *args: Any, **kwargs: Any) -> None:
        """Start tracking an object; timed as csrt."""
        self.clock.timed("csrt", self.inner.start, *args, **kwargs)

    def observe(self, *args: Any, **kwargs: Any) -> None:
        """Pass a frame and mask to the tracker; timed as csrt."""
        self.clock.timed("csrt", self.inner.observe, *args, **kwargs)

    def skip(self, *args: Any, **kwargs: Any) -> None:
        """Skip a frame in the tracker; timed as csrt."""
        self.clock.timed("csrt", self.inner.skip, *args, **kwargs)

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        """Ask the tracker for a box; timed as csrt."""
        return self.clock.timed("csrt", self.inner.propose, *args, **kwargs)

    def last_reliable_box(self, obj_id: int) -> Any:
        """Return the last reliable box of an object (not timed)."""
        return self.inner.last_reliable_box(obj_id)


class TimingBoxCheck:
    """Wrapper around the box check that times its check calls as box_check."""

    def __init__(self, inner: BoxCheck, clock: Clock) -> None:
        """Keep the wrapped box check and the clock to charge."""
        self.inner, self.clock = inner, clock

    def check(self, *args: Any, **kwargs: Any) -> Any:
        """Run the box check; timed as box_check."""
        return self.clock.timed("box_check", self.inner.check, *args, **kwargs)


@dataclass(frozen=True)
class VideoTiming:
    """Timing result of one pass over one video.

    Attributes:
        frames: Frames processed, counted from the prompt frame to the end of the video.
        reprompts: Number of re-prompts made.
        seconds: Seconds per component, including other and total_loop.
        calls: Number of calls per component.
    """

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
    # Recovery starts from a single prompt frame.
    groups = prompts.groups()
    if len(groups) != 1:
        raise ValueError(f"{frames_dir.name}: recovery needs one prompt frame")
    start = next(iter(groups))
    # Decode all frames first so image reading is not timed.
    frames: list[Frame] = []
    for path in frame_paths(frames_dir):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read frame {path}")
        frames.append(np.asarray(image, dtype=np.uint8))
    # Build the real recovery loop with every part wrapped by a timer.
    clock = Clock()
    run = RecoveryRun(
        TimingPredictor(predictor, clock),
        TimingDetector(detector, clock),
        TimingBoxes(boxes, clock),
        TimingBoxCheck(box_check, clock),  # type: ignore[arg-type]
        settings,
    )
    # Time the whole loop; the sink is a no-op because nothing is written.
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
    # The loop time excludes init_state; other is what the timed parts do not cover.
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
    """Pick a seeded sample of videos, sorted.

    Args:
        names: Video names to choose from.
        size: Number of videos.

    Returns:
        The sample.

    Raises:
        ValueError: If size is larger than the number of videos.
    """
    pool = sorted(names)
    if size > len(pool):
        raise ValueError(f"cannot sample {size} of {len(pool)} videos")
    return sorted(random.Random(SEED).sample(pool, size))


def _mean(values: Sequence[float]) -> float:
    """Mean of the values, or NaN if any value is NaN."""
    return float("nan") if any(math.isnan(v) for v in values) else statistics.mean(values)


def _spread(values: Sequence[float]) -> float:
    """Standard deviation of the values: 0 for a single value, NaN if any value is NaN."""
    if any(math.isnan(v) for v in values):
        return float("nan")
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarise(
    model: str, split: str, videos: Sequence[str], passes: Sequence[Sequence[VideoTiming]]
) -> list[dict[str, Any]]:
    """One row per component: per-pass totals over the sampled videos, mean and std over passes."""
    device = device_fields()
    # Frames per pass, from the sum over the sampled videos.
    frames = [sum(v.frames for v in p) for p in passes]
    rows: list[dict[str, Any]] = []
    # One row per component: time per frame and per call, and fps for total_loop.
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
    """Time the real system for one model: one warm-up pass, then timed passes.

    Args:
        model: SAM 2 size.
        prompts: Prompts for the videos.
        cfg: Data configuration.
        make_predictor: Builds the predictor once.
        make_detector: Builds a fresh detector for each video pass.
        make_boxes: Builds a fresh box source for each video pass.
        box_check: The box check.
        settings: Recovery settings.
        videos: Sampled video names.
        repeats: Number of timed passes.

    Returns:
        The summary rows of the model.

    Raises:
        ValueError: If repeats is below 1.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    predictor = make_predictor()

    def one_pass() -> list[VideoTiming]:
        """Time every sampled video once and return the per-video timings."""
        return [
            # Time every video once with fresh detector and box-source objects.
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

    # Warm-up pass, discarded.
    one_pass()
    # The timed passes.
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
    """Build the real system for a model and time it.

    Args:
        model: SAM 2 size.
        args: Parsed arguments (device and repeats are used).
        data: Data configuration.
        prompts: Prompts for the videos.
        videos: Sampled video names.
        settings: Recovery settings.
        box_check: The box check.

    Returns:
        The summary rows of the model.
    """
    from src.detection.online import OnlineRuleDetector
    from src.detection.rules import DEFAULT_RULES_PATH

    run_cfg = load_recovery_config(Path(CONFIGS[model]))
    check_online_method(run_cfg.method)
    # Factories build fresh detectors and trackers; the predictor is built once.
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
    """Command line entry point: time the components of the real system.

    Reads the prompts file, the frozen detector rules and the recovery settings; frames and
    weights come from the data root in the environment. Writes the timing table to --out and
    prints progress lines. Exits with the usage error code 2 if the data root variable is not set.

    Args:
        argv: Command line arguments; None means sys.argv.

    Raises:
        SystemExit: If the detector rules file does not exist yet (the detector is not frozen).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    args = parser.parse_args(argv)
    from src.detection.rules import DEFAULT_RULES_PATH

    # Timing needs the frozen rule detector.
    if not DEFAULT_RULES_PATH.is_file():
        raise SystemExit(f"detector not frozen yet: {DEFAULT_RULES_PATH} does not exist")
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    data = load_data_config()
    prompts = load_batch_prompts(args.prompts)
    videos = sample_videos(list(prompts.videos))
    settings = load_recovery_settings(CONFIG_PATH)
    box_check = BoxCheck(load_box_check_config())
    # Time each model in turn and collect its rows.
    rows: list[dict[str, Any]] = []
    for model in args.models:
        print(f"{model}: videos {videos}, {args.repeats} repeats after one warm-up pass")
        rows += run_model(model, args, data, prompts, videos, settings, box_check)
    # Write the combined timing table.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(COLUMNS)).to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
