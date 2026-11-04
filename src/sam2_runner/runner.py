"""Run SAM 2 on one video: box prompts, propagation, indexed PNGs and the signal CSV.

The signal CSV is written last and atomically, so its presence (and validity) marks a
finished video. A video whose output is incomplete is wiped (only its own files) and redone.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.common.boxes import Box
from src.common.layout import mask_path, masks_dir, signals_path
from src.common.masks import masks_to_label_map, write_label_map
from src.common.signals import read_signals, signals_from_masks, write_signals
from src.sam2_runner.predictor import VideoPredictor, to_numpy
from src.sam2_runner.prompts import Prompts
from src.sam2_runner.scores import object_score

FRAME_SUFFIXES = (".jpg", ".jpeg")
TEMP_SUFFIX = ".tmp"


@dataclass(frozen=True)
class RunResult:
    table: pd.DataFrame
    stored_rows: int

    @property
    def source_summary(self) -> str:
        total = len(self.table)
        if self.stored_rows == total:
            return f"object_score source: object_score_logits for {total}/{total} rows"
        return (
            f"object_score source: object_score_logits for {self.stored_rows}/{total} rows, "
            f"mean-mask-logit fallback for {total - self.stored_rows}"
        )


def sam2_box(box: Box) -> list[float]:
    """Box as SAM 2 expects it: [x0, y0, x1, y1] corners in pixels of the original frame.

    add_new_points_or_box reshapes it to two points and divides by the video size. Our right
    and bottom edges are exclusive, i.e. the continuous edge of the last pixel, so they pass
    through unchanged.
    """
    return [float(box.x0), float(box.y0), float(box.x1), float(box.y1)]


def frame_stems(frames_dir: Path) -> list[str]:
    """Frame file stems in SAM 2's order (numeric stem)."""
    stems = [p.stem for p in frames_dir.iterdir() if p.suffix.lower() in FRAME_SUFFIXES]
    if not stems:
        raise FileNotFoundError(f"no JPEG frames in {frames_dir}")
    return sorted(stems, key=int)


def is_complete(
    output_root: Path,
    method: str,
    split: str,
    video: str,
    stems: Sequence[str],
    obj_ids: Sequence[int],
) -> bool:
    """True only if the signal CSV is valid with one row per (object, frame) and all PNGs exist."""
    csv_path = signals_path(output_root, method, split, video)
    if not csv_path.is_file():
        return False
    try:
        table = read_signals(csv_path)
    except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    expected = {(i, f) for i in obj_ids for f in range(len(stems))}
    keys = set(zip(table["object_id"].tolist(), table["frame_index"].tolist(), strict=True))
    if len(table) != len(expected) or keys != expected:
        return False
    if set(table["video"]) != {video}:
        return False
    return all(mask_path(output_root, method, split, video, s).is_file() for s in stems)


def clear_video(output_root: Path, method: str, split: str, video: str) -> None:
    """Delete this video's masks and signal file (and a leftover temp file) under the run dir."""
    shutil.rmtree(masks_dir(output_root, method, split, video), ignore_errors=True)
    csv_path = signals_path(output_root, method, split, video)
    csv_path.unlink(missing_ok=True)
    csv_path.with_name(csv_path.name + TEMP_SUFFIX).unlink(missing_ok=True)


def run_video(
    predictor: VideoPredictor,
    frames_dir: Path,
    prompts: Prompts,
    output_root: Path,
    method: str,
    split: str,
    video: str,
) -> RunResult:
    """Run one video; write one indexed PNG per frame, then the signal CSV; return the signals."""
    stems = frame_stems(frames_dir)
    clear_video(output_root, method, split, video)
    state = predictor.init_state(str(frames_dir))
    for obj_id, box in prompts.boxes.items():
        predictor.add_new_points_or_box(state, prompts.frame_index, obj_id, box=sam2_box(box))
    per_object: dict[int, list[npt.NDArray[np.bool_]]] = {i: [] for i in prompts.boxes}
    scores: dict[int, list[float]] = {i: [] for i in prompts.boxes}
    stored_rows = 0
    for count, (frame_idx, obj_ids, logits) in enumerate(predictor.propagate_in_video(state)):
        if frame_idx != count or count >= len(stems):
            raise RuntimeError(f"unexpected frame index {frame_idx} at position {count}")
        _store_frame(output_root, method, split, video, stems[frame_idx], obj_ids, logits)
        for row, obj_id in enumerate(obj_ids):
            obj_logits = to_numpy(logits[row, 0])
            per_object[obj_id].append(obj_logits > 0)
            score, stored = object_score(state, obj_id, frame_idx, obj_logits)
            scores[obj_id].append(score)
            stored_rows += stored
    if len(per_object[next(iter(per_object))]) != len(stems):
        raise RuntimeError(f"expected {len(stems)} frames, SAM 2 returned fewer")
    table = pd.concat(
        [signals_from_masks(video, i, per_object[i], scores[i]) for i in sorted(per_object)],
        ignore_index=True,
    )
    _write_signals_atomic(signals_path(output_root, method, split, video), table)
    return RunResult(table, stored_rows)


def _write_signals_atomic(path: Path, table: pd.DataFrame) -> None:
    temp = path.with_name(path.name + TEMP_SUFFIX)
    write_signals(temp, table)
    os.replace(temp, path)


def _store_frame(
    output_root: Path,
    method: str,
    split: str,
    video: str,
    stem: str,
    obj_ids: Sequence[int],
    logits: Any,
) -> None:
    arrays = {obj_id: to_numpy(logits[row, 0]) for row, obj_id in enumerate(obj_ids)}
    first = next(iter(arrays.values()))
    shape = (int(first.shape[0]), int(first.shape[1]))
    label = masks_to_label_map({i: a > 0 for i, a in arrays.items()}, shape, scores=arrays)
    write_label_map(mask_path(output_root, method, split, video, stem), label)
