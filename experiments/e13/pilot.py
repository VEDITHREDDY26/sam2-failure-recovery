"""E13 pilot: how good is each box source at the frames the frozen detector flags (SA-V val only).

    python -m experiments.e13.pilot --model tiny [--videos a b] [--shard K --nshards N] \
        [--out-dir D]
    python -m experiments.e13.pilot --model tiny --merge

Replays the cached SAM 2-alone outputs (no SAM 2 call). The frozen rule detector runs on every
frame; each object starts at its prompt frame and every arm's box source is driven in the online
loop's order: start, observe on unflagged frames, skip on flagged frames inside the cooldown, and
propose at a candidate (a flag outside the cooldown). A candidate starts the cooldown whatever any
gate would decide, and nothing is restarted after it, so every arm sees the same candidates. Reads
ground truth, so it is an evaluation script. Part files (one per video, written atomically) go to
<out-dir>/<model>/; a finished video is skipped on a rerun. --merge writes the two CSVs under
results/ and only when every SA-V val video has its part files.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.common.boxes import Box, box_from_mask
from src.common.config import DATA_ROOT_ENV, DataConfig
from src.common.layout import mask_path, signals_path
from src.common.masks import LabelMap, read_label_map
from src.common.signals import read_signals
from src.data.video import VideoSample
from src.detection.labels import FAILURE_IOU, mask_iou
from src.detection.online import OnlineRuleDetector
from src.detection.rules import DEFAULT_RULES_PATH
from src.eval.check_complete import load_video, prompts_file, split_videos
from src.eval.score import load_data_config
from src.recovery.box_check import BoxCheck
from src.recovery.settings import CONFIG_PATH, load_recovery_settings
from src.recovery.sources import (
    BoxSource,
    CSRTBoxSource,
    Detector,
    Frame,
    MotionBoxSource,
    ViTBoxSource,
)
from src.recovery.verify import box_iou
from src.recovery.vit_tracker import ViTFactory, load_vit_config, make_vit_factory
from src.sam2_runner.predictor import REPO_ROOT
from src.sam2_runner.prompts import Prompts, load_batch_prompts

SPLIT = "sav_val"
MODELS = ("tiny", "large")
ARMS = (
    "csrt_reinit",
    "csrt_continuous",
    "vit_reinit",
    "vit_continuous",
    "vit_anchored",
    "motion",
)
CANDIDATE_COLUMNS = (
    "video",
    "object_id",
    "frame_index",
    "arm",
    "fired",
    "status",
    "raw_x0",
    "raw_y0",
    "raw_x1",
    "raw_y1",
    "x0",
    "y0",
    "x1",
    "y1",
    "score",
    "iou_sam2",
    "box_check_ok",
    "box_check_reason",
    "scored_frame",
    "gt_visible",
    "iou_gt",
    "sam2_iou",
    "cls",
    "first_of_object",
    "sam2_mask_empty",
)
TIMING_COLUMNS = ("video", "arm", "calls", "seconds")
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_NAME = "e13_pilot"

Mask = np.ndarray[Any, np.dtype[np.bool_]]


@dataclass
class VideoInputs:
    """Everything the replay needs to know about one video.

    Attributes:
        video: The video sample (frames and ground truth).
        prompts: The prompts of the video; one prompt frame is expected.
        labels: Function from frame index to the cached SAM 2 alone label map.
        scores: SAM 2 object score per object id and frame index.
        read_frame: Function from frame index to the BGR image of that frame.
    """

    video: VideoSample
    prompts: Prompts
    labels: Callable[[int], LabelMap]
    scores: dict[int, dict[int, float]]
    read_frame: Callable[[int], Frame]


@dataclass(frozen=True)
class Truth:
    """What the ground truth says about one candidate (the same for every arm)."""

    scored_frame: int | None
    visible: bool | None
    gt_box: Box | None
    sam2_iou: float | None
    cls: str


def check_split(split: str) -> str:
    """Refuse any split other than SA-V val.

    Args:
        split: Split name given on the command line.

    Returns:
        The split name, unchanged.

    Raises:
        SystemExit: If the split is not "sav_val".
    """
    if split != SPLIT:
        raise SystemExit(f"split '{split}' refused: the E13 pilot reads SA-V val only")
    return split


def part_paths(out_dir: Path, model: str, video: str) -> tuple[Path, Path]:
    """Paths of the per-video part files of the pilot.

    Args:
        out_dir: Folder for the part files.
        model: "tiny" or "large".
        video: Video name.

    Returns:
        The candidates CSV path and the timing CSV path, in that order.
    """
    base = out_dir / model
    return base / "candidates" / f"{video}.csv", base / "timing" / f"{video}.csv"


def make_source(arm: str, factory: ViTFactory | None, anchor_iou: float) -> BoxSource:
    """Create the box source for one arm.

    Args:
        arm: One of ARMS ("csrt_*", "motion" or "vit_*").
        factory: Creates the ViT tracker; None when it is not loaded.
        anchor_iou: IoU threshold used by the anchored ViT mode.

    Returns:
        A new box source for the arm.

    Raises:
        ValueError: If a ViT arm is asked for and no ViT factory is given.
    """
    if arm.startswith("csrt_"):
        return CSRTBoxSource(arm.removeprefix("csrt_"))
    if arm == "motion":
        return MotionBoxSource()
    if factory is None:
        raise ValueError(f"arm '{arm}' needs the ViT tracker")
    return ViTBoxSource(factory, arm.removeprefix("vit_"), anchor_iou)


def classify(video: VideoSample, obj_id: int, t: int, labels: Callable[[int], LabelMap]) -> Truth:
    """Scored frame = t when annotated, else the first annotated frame after it."""
    later = [f for f in video.annotated_frames(obj_id) if f >= t]
    gt = video.gt_mask(obj_id, later[0]) if later else None
    if gt is None:
        return Truth(None, None, None, None, "unscored")
    frame = later[0]
    visible = bool(gt.any())
    iou = mask_iou(labels(frame) == obj_id, gt)
    if not visible:
        return Truth(frame, False, None, iou, "not_visible")
    cls = "real_failure" if iou < FAILURE_IOU else "false_alarm"
    return Truth(frame, True, box_from_mask(gt), iou, cls)


class _Clock:
    """Adds up the time and number of calls spent in each arm's box source."""

    def __init__(self) -> None:
        """Start with zero seconds and zero calls for every arm."""
        self.seconds = dict.fromkeys(ARMS, 0.0)
        self.calls = dict.fromkeys(ARMS, 0)

    def timed(self, arm: str, call: Callable[..., Any], *args: Any) -> Any:
        """Call a function, add its wall time to the arm's total and count the call.

        Args:
            arm: Arm name the time is charged to.
            call: Function to run.
            *args: Arguments passed to the function.

        Returns:
            Whatever the function returns.
        """
        tick = time.perf_counter()
        out = call(*args)
        self.seconds[arm] += time.perf_counter() - tick
        self.calls[arm] += 1
        return out


def _row(
    video: str,
    obj_id: int,
    t: int,
    arm: str,
    fired: tuple[str, ...],
    result: Any,
    sam2_box: Box | None,
    check: Any,
    truth: Truth,
    first: bool,
    empty: bool,
) -> tuple[object, ...]:
    """Build one candidate row (one arm at one candidate frame) in CANDIDATE_COLUMNS order.

    Boxes are xyxy. Missing values are written as empty strings. IoU against SAM 2 and against
    the ground truth is only filled when the box source returned a usable box.

    Args:
        video: Video name.
        obj_id: Object id.
        t: Frame index of the candidate.
        arm: Arm name.
        fired: Names of the detector rules that fired.
        result: The box source's proposal (status, raw box, box, score).
        sam2_box: Box of SAM 2's current mask, or None if the mask is empty.
        check: Box-check verdict, or None if no check was made.
        truth: Ground-truth facts about the candidate.
        first: True if this is the first candidate of the object.
        empty: True if SAM 2's mask is empty at this frame.

    Returns:
        The row values.
    """
    raw, box = result.raw_box, result.box
    raw_c = (raw.x0, raw.y0, raw.x1, raw.y1) if raw else ("",) * 4
    box_c = (box.x0, box.y0, box.x1, box.y1) if box else ("",) * 4
    ok = result.status == "ok" and box is not None
    iou_sam2 = box_iou(box, sam2_box) if ok else ""
    iou_gt = box_iou(box, truth.gt_box) if ok and truth.gt_box is not None else ""
    return (
        video,
        obj_id,
        t,
        arm,
        "+".join(fired),
        result.status,
        *raw_c,
        *box_c,
        "" if result.score is None else result.score,
        iou_sam2,
        "" if check is None else check.ok,
        "" if check is None else (check.reason or ""),
        "" if truth.scored_frame is None else truth.scored_frame,
        "" if truth.visible is None else truth.visible,
        iou_gt,
        "" if truth.sam2_iou is None else truth.sam2_iou,
        truth.cls,
        first,
        empty,
    )


def replay_video(
    inputs: VideoInputs,
    detector: Detector,
    sources: dict[str, BoxSource],
    box_check: BoxCheck,
    cooldown: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate rows (candidate x arm) and the tracker time per arm for one video."""
    video, prompts = inputs.video, inputs.prompts
    groups = prompts.groups()
    if len(groups) != 1:
        raise ValueError(f"'{video.name}': need one prompt frame, got {sorted(groups)}")
    start, objects = next(iter(groups.items()))
    # Reset all stateful parts so videos do not leak into each other.
    detector.reset()
    for source in sources.values():
        source.reset()
    clock = _Clock()
    # Latest candidate frame per object, used for the cooldown.
    last: dict[int, int] = {}
    rows: list[tuple[object, ...]] = []
    # Walk the frames in the same order as the online loop.
    for t in range(start, video.num_frames):
        label, frame = inputs.labels(t), inputs.read_frame(t)
        shape = (label.shape[0], label.shape[1])
        for obj in objects:
            mask = label == obj
            score = inputs.scores[obj][t]
            # The prompt frame only initialises the detector and every box source.
            if t == start:
                detector.observe(obj, t, mask, score, shape)
                for arm, source in sources.items():
                    clock.timed(arm, source.start, obj, t, frame, mask)
                continue
            decision = detector.update(obj, t, mask, score, shape)
            # No flag: sources just observe the frame and the SAM 2 mask.
            if not decision.flag:
                for arm, source in sources.items():
                    clock.timed(arm, source.observe, obj, t, frame, mask)
            # Flag inside the cooldown: no proposal; sources skip the frame.
            elif obj in last and t - last[obj] < cooldown:
                for arm, source in sources.items():
                    clock.timed(arm, source.skip, obj, t, frame)
            else:
                # A real candidate: score it once, then ask every arm for a box on the same frame.
                truth = classify(video, obj, t, inputs.labels)
                sam2_box = box_from_mask(mask)
                for arm, source in sources.items():
                    result = clock.timed(arm, source.propose, obj, t, frame)
                    ok = result.status == "ok" and result.box is not None
                    # The box check only runs on boxes the source actually returned.
                    check = (
                        box_check.check(result.raw_box, shape, source.last_reliable_box(obj))
                        if ok
                        else None
                    )
                    rows.append(
                        _row(
                            video.name,
                            obj,
                            t,
                            arm,
                            decision.fired,
                            result,
                            sam2_box,
                            check,
                            truth,
                            obj not in last,
                            sam2_box is None,
                        )
                    )
                # Any candidate starts the cooldown, whatever a gate would decide.
                last[obj] = t
    timing = [(video.name, arm, clock.calls[arm], clock.seconds[arm]) for arm in sources]
    return (
        pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS)),
        pd.DataFrame(timing, columns=list(TIMING_COLUMNS)),
    )


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a temporary file so a stopped run leaves no half-written file.

    Args:
        frame: Table to save.
        path: Final file path; its folder is created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def run_part(
    inputs: VideoInputs,
    out_dir: Path,
    model: str,
    detector: Detector,
    factory: ViTFactory | None,
    anchor_iou: float,
    box_check: BoxCheck,
    cooldown: int,
) -> None:
    """Replay one video with every arm and save its part files.

    Args:
        inputs: Data of the video.
        out_dir: Folder for the part files.
        model: "tiny" or "large".
        detector: Frozen rule detector.
        factory: Creates the ViT tracker, or None.
        anchor_iou: IoU threshold for the anchored ViT mode.
        box_check: Box check applied to each proposed box.
        cooldown: Frames after a candidate in which new flags are skipped.
    """
    candidates_path, timing_path = part_paths(out_dir, model, inputs.video.name)
    sources = {arm: make_source(arm, factory, anchor_iou) for arm in ARMS}
    candidates, timing = replay_video(inputs, detector, sources, box_check, cooldown)
    # Timing is written first: a video counts as done only when both files exist.
    _write_atomic(timing, timing_path)
    _write_atomic(candidates, candidates_path)


def is_done(out_dir: Path, model: str, video: str) -> bool:
    """Tell whether both part files of a video already exist.

    Args:
        out_dir: Folder for the part files.
        model: "tiny" or "large".
        video: Video name.

    Returns:
        True if the candidates file and the timing file are both present.
    """
    return all(p.is_file() for p in part_paths(out_dir, model, video))


def select_videos(
    names: Sequence[str], requested: Sequence[str] | None, shard: int, nshards: int
) -> list[str]:
    """Sorted list, optionally narrowed to explicit names, then to every nshards-th video."""
    if nshards < 1 or not 0 <= shard < nshards:
        raise ValueError(f"shard {shard} of {nshards} is not valid")
    ordered = sorted(names)
    if requested:
        unknown = [v for v in requested if v not in ordered]
        if unknown:
            raise ValueError(f"not SA-V val videos: {unknown}")
        ordered = sorted(requested)
    return ordered[shard::nshards]


def merge_parts(
    out_dir: Path, model: str, videos: Sequence[str], results_dir: Path
) -> tuple[Path, Path] | list[str]:
    """Write the merged CSVs, or return the videos that have no part files yet."""
    # Merge only when every video has both part files.
    missing = [v for v in videos if not is_done(out_dir, model, v)]
    if missing:
        return missing
    # Concatenate the per-video files in the given video order.
    cand = pd.concat(
        [pd.read_csv(part_paths(out_dir, model, v)[0], dtype={"video": str}) for v in videos],
        ignore_index=True,
    )
    timing = pd.concat(
        [pd.read_csv(part_paths(out_dir, model, v)[1], dtype={"video": str}) for v in videos],
        ignore_index=True,
    )
    cand_out = results_dir / f"e13_pilot_candidates_{model}_{SPLIT}.csv"
    timing_out = results_dir / f"e13_pilot_timing_{model}_{SPLIT}.csv"
    cand_out.parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(cand_out, index=False)
    timing.to_csv(timing_out, index=False)
    return cand_out, timing_out


def load_inputs(cfg: DataConfig, model: str, name: str) -> VideoInputs:
    """Cached SAM 2 alone outputs and frames for one SA-V val video."""
    # Cached outputs of SAM 2 alone (masks and per-frame object scores) are the replay input.
    method, output_root = f"sam2_{model}", cfg.data_root / cfg.output_root
    prompts = load_batch_prompts(prompts_file(cfg, SPLIT)).videos[name]
    video = load_video(cfg, SPLIT, name)
    table = read_signals(signals_path(output_root, method, SPLIT, name))
    # Index object_score by object id and frame index.
    scores = {}
    for obj in table["object_id"].unique().tolist():
        rows = table[table["object_id"] == obj]
        scores[int(obj)] = dict(
            zip(rows["frame_index"].tolist(), rows["object_score"].tolist(), strict=True)
        )

    def labels(index: int) -> LabelMap:
        """Read the cached SAM 2 alone label map of one frame."""
        return read_label_map(mask_path(output_root, method, SPLIT, name, video.frame_stems[index]))

    def read_frame(index: int) -> Frame:
        """Read one frame as a BGR uint8 image.

        Raises:
            OSError: If the image file cannot be read.
        """
        image = cv2.imread(str(video.frame_path(index)), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read frame {video.frame_path(index)}")
        return np.asarray(image, dtype=np.uint8)

    return VideoInputs(video, prompts, labels, scores, read_frame)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pilot for some videos, or merge the part files (command line).

    Reads cached SAM 2 outputs, frames and ground truth under the data root (environment
    variable). Without --merge it writes per-video part files under <out-dir>/<model>/ and
    skips videos that are finished. With --merge it writes the merged candidates and timing CSVs
    into --results-dir once every SA-V val video has its part files.

    Args:
        argv: Command-line arguments; None means sys.argv.

    Returns:
        0 on success; 1 if --merge finds videos without part files.

    Raises:
        SystemExit: If the split is not sav_val.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(argv)
    check_split(args.split)
    # Resolve the output folder and the full list of SA-V val videos.
    root = os.environ.get(DATA_ROOT_ENV)
    if not root:
        parser.error(f"{DATA_ROOT_ENV} is not set")
    out_dir = args.out_dir or Path(root) / DEFAULT_OUT_NAME
    cfg = load_data_config()
    everything = sorted(split_videos(cfg, SPLIT))
    if args.merge:
        merged = merge_parts(out_dir, args.model, everything, args.results_dir)
        if isinstance(merged, list):
            sys.stdout.write(f"{len(merged)} video(s) have no part files: {merged}\n")
            return 1
        sys.stdout.write(f"wrote {merged[0]} and {merged[1]}\n")
        return 0
    videos = select_videos(everything, args.videos, args.shard, args.nshards)
    # Load the ViT tracker, recovery settings and box check once for all videos.
    vit = load_vit_config()
    factory = make_vit_factory(vit)
    settings = load_recovery_settings(CONFIG_PATH)
    box_check = BoxCheck()
    ran = 0
    # Replay each video not done yet; a fresh detector is made per video.
    for name in videos:
        if is_done(out_dir, args.model, name):
            sys.stdout.write(f"{name}: skipped (done)\n")
            continue
        detector = OnlineRuleDetector.from_config(args.model, DEFAULT_RULES_PATH)
        inputs = load_inputs(cfg, args.model, name)
        run_part(
            inputs,
            out_dir,
            args.model,
            detector,
            factory,
            vit.anchor_iou,
            box_check,
            settings.cooldown_frames,
        )
        ran += 1
        sys.stdout.write(f"{name}: done\n")
    sys.stdout.write(f"{ran} run, {len(videos) - ran} skipped\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
