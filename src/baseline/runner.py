"""CSRT + GrabCut baseline on one video.

Every object starts at its own first annotated frame with the tight box of its ground-truth mask
(the standard semi-supervised prompt); that is the only ground truth read. From then on the CSRT
tracker follows the box and GrabCut cuts the object out of it. When the tracker reports failure
or its box is unusable the object's mask is empty for that frame and the loss is recorded; the
tracker keeps being updated and is never re-initialised. If GrabCut finds no foreground inside a
valid box, the mask falls back to the filled box and the frame is flagged `grabcut_fallback`.

Outputs per video: one indexed PNG per frame, then a timing CSV, then the per-frame status CSV
(written last and atomically; its presence together with all PNGs marks a finished video).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.baseline.config import BaselineConfig
from src.baseline.hardware import hardware_fields
from src.baseline.opencv_ops import (
    BgrImage,
    BoxTracker,
    CsrtTracker,
    RgbImage,
    grabcut_in_box,
    pad_to_min_side,
    rgb_to_bgr,
)
from src.common.boxes import Box
from src.common.layout import mask_path, masks_dir, run_dir
from src.common.masks import BoolMask, masks_to_label_map, write_label_map
from src.data.video import VideoSample

STATUS_COLUMNS = (
    "video",
    "object_id",
    "frame_index",
    "tracker_ok",
    "x0",
    "y0",
    "x1",
    "y1",
    "grabcut_fallback",
    "area",
)
TIMING_COLUMNS = (
    "video",
    "frames",
    "objects",
    "compute_s",
    "wall_s",
    "fps",
    "cpu_model",
    "cpu_count",
    "cv2_version",
    "cv2_threads",
)
TrackerFactory = Callable[[], BoxTracker]


def tracker_path(output_root: Path, method: str, split: str, video: str) -> Path:
    return run_dir(output_root, method, split) / "tracker" / f"{video}.csv"


def timing_path(output_root: Path, method: str, split: str, video: str) -> Path:
    return run_dir(output_root, method, split) / "timing" / f"{video}.csv"


def default_tracker_factory(cfg: BaselineConfig) -> TrackerFactory:
    return lambda: CsrtTracker(cfg.min_box_side)


@dataclass
class _ObjectState:
    start: int
    start_box: Box
    tracker: BoxTracker | None = None


@dataclass(frozen=True)
class VideoRun:
    status: pd.DataFrame
    timing: dict[str, object]


def _row(
    video: str, obj: int, frame: int, ok: bool | None, box: Box | None, fallback: bool, area: int
) -> dict[str, object]:
    coords = (box.x0, box.y0, box.x1, box.y1) if box else (None, None, None, None)
    return {
        "video": video,
        "object_id": obj,
        "frame_index": frame,
        "tracker_ok": None if ok is None else int(ok),
        "x0": coords[0],
        "y0": coords[1],
        "x1": coords[2],
        "y1": coords[3],
        "grabcut_fallback": int(fallback),
        "area": area,
    }


def _cut(frame_bgr: BgrImage, box: Box, cfg: BaselineConfig) -> tuple[BoolMask, bool]:
    mask = grabcut_in_box(frame_bgr, box, cfg.grabcut_iterations, cfg.roi_margin_frac)
    if mask.any():
        return mask, False
    fallback = np.zeros(mask.shape, dtype=np.bool_)
    fallback[box.y0 : box.y1, box.x0 : box.x1] = True
    return fallback, True


def _read_rgb(path: Path) -> RgbImage:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _start_states(video: VideoSample) -> dict[int, _ObjectState]:
    states = {}
    for obj in video.object_ids:
        prompt = video.first_box(obj)
        states[obj] = _ObjectState(prompt.frame_index, prompt.box)
    return states


def _step_object(
    state: _ObjectState,
    frame_index: int,
    frame_bgr: BgrImage,
    cfg: BaselineConfig,
    make_tracker: TrackerFactory,
    shape: tuple[int, int],
) -> tuple[BoolMask, bool | None, Box | None, bool]:
    """Returns (mask, tracker_ok, box, grabcut_fallback) for one object on one frame."""
    empty = np.zeros(shape, dtype=np.bool_)
    if frame_index < state.start:
        return empty, None, None, False
    if frame_index == state.start:
        box = pad_to_min_side(state.start_box, cfg.min_box_side, *shape)
        state.tracker = make_tracker()
        state.tracker.init(frame_bgr, box)
        mask, fallback = _cut(frame_bgr, box, cfg)
        return mask, True, box, fallback
    assert state.tracker is not None
    result = state.tracker.update(frame_bgr)
    if result.box is None:
        return empty, result.ok, None, False
    mask, fallback = _cut(frame_bgr, result.box, cfg)
    return mask, result.ok, result.box, fallback


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    table.to_csv(temp, index=False)
    os.replace(temp, path)


def run_video(
    video: VideoSample,
    cfg: BaselineConfig,
    output_root: Path,
    method: str,
    split: str,
    make_tracker: TrackerFactory | None = None,
) -> VideoRun:
    make_tracker = make_tracker or default_tracker_factory(cfg)
    states = _start_states(video)
    rows: list[dict[str, object]] = []
    compute = 0.0
    wall_start = time.perf_counter()
    for frame_index in range(video.num_frames):
        tick = time.perf_counter()
        rgb = _read_rgb(video.frame_path(frame_index))
        shape = (rgb.shape[0], rgb.shape[1])
        bgr = rgb_to_bgr(rgb)
        masks: dict[int, BoolMask] = {}
        for obj, state in states.items():
            mask, ok, box, fallback = _step_object(
                state, frame_index, bgr, cfg, make_tracker, shape
            )
            masks[obj] = mask
            rows.append(_row(video.name, obj, frame_index, ok, box, fallback, int(mask.sum())))
        compute += time.perf_counter() - tick
        label = masks_to_label_map(masks, shape)
        write_label_map(
            mask_path(output_root, method, split, video.name, video.frame_stems[frame_index]), label
        )
    wall = time.perf_counter() - wall_start
    status = pd.DataFrame(rows, columns=list(STATUS_COLUMNS))
    timing: dict[str, object] = {
        "video": video.name,
        "frames": video.num_frames,
        "objects": len(video.object_ids),
        "compute_s": compute,
        "wall_s": wall,
        "fps": video.num_frames / compute if compute > 0 else float("nan"),
        **hardware_fields(),
    }
    _write_csv(timing_path(output_root, method, split, video.name), pd.DataFrame([timing]))
    _write_csv(tracker_path(output_root, method, split, video.name), status)
    return VideoRun(status, timing)


def is_complete(
    output_root: Path,
    method: str,
    split: str,
    video: str,
    stems: Sequence[str],
    obj_ids: Sequence[int],
) -> bool:
    status_file = tracker_path(output_root, method, split, video)
    if not status_file.is_file() or not timing_path(output_root, method, split, video).is_file():
        return False
    try:
        table = pd.read_csv(status_file)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    if list(table.columns) != list(STATUS_COLUMNS):
        return False
    expected = {(o, f) for o in obj_ids for f in range(len(stems))}
    keys = set(zip(table["object_id"].tolist(), table["frame_index"].tolist(), strict=True))
    if len(table) != len(expected) or keys != expected:
        return False
    return all(mask_path(output_root, method, split, video, s).is_file() for s in stems)


def clear_video(output_root: Path, method: str, split: str, video: str) -> None:
    """Remove this video's own outputs before a redo of an incomplete video."""
    folder = masks_dir(output_root, method, split, video)
    if folder.is_dir():
        for png in folder.glob("*.png"):
            png.unlink()
    for path in (
        tracker_path(output_root, method, split, video),
        timing_path(output_root, method, split, video),
    ):
        path.unlink(missing_ok=True)
