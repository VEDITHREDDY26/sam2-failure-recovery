"""Run SAM 2 on one video: box prompts, propagation, indexed PNGs and the signal CSV.

The signal CSV is written last and atomically, so its presence (and validity) marks a
finished video. A video whose output is incomplete is wiped (only its own files) and redone.
"""

from __future__ import annotations

import os
import shutil
import tempfile
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
from src.common.signals import SIGNAL_COLUMNS, read_signals, write_signals
from src.sam2_runner.predictor import VideoPredictor, to_numpy
from src.sam2_runner.prompts import Prompts
from src.sam2_runner.scores import object_score

FRAME_SUFFIXES = (".jpg", ".jpeg")
TEMP_SUFFIX = ".tmp"


@dataclass(frozen=True)
class RunResult:
    table: pd.DataFrame
    stored_rows: int
    unavailable_rows: int = 0

    @property
    def source_summary(self) -> str:
        total = len(self.table) - self.unavailable_rows
        if self.stored_rows == total:
            text = f"object_score source: object_score_logits for {total}/{total} rows"
        else:
            text = (
                f"object_score source: object_score_logits for {self.stored_rows}/{total} rows, "
                f"mean-mask-logit fallback for {total - self.stored_rows}"
            )
        if self.unavailable_rows:
            text += f"; NaN before the prompt frame for {self.unavailable_rows} rows"
        return text


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
    offload_video: bool = False,
) -> RunResult:
    """Run one video; write one indexed PNG per frame, then the signal CSV; return the signals.

    Objects are prompted at their own start frame; objects sharing a start frame share one SAM 2
    pass, passes run in ascending start order. Before its start frame an object has an empty
    mask and a NaN object score (SAM 2 produced no output).
    """
    stems = frame_stems(frames_dir)
    clear_video(output_root, method, split, video)
    groups = prompts.groups()
    if max(groups) >= len(stems):
        raise ValueError(f"'{video}': prompt frame {max(groups)} but only {len(stems)} frames")
    state = predictor.init_state(str(frames_dir), offload_video_to_cpu=offload_video)
    rows: dict[int, list[SignalRow]] = {i: [] for i in prompts.boxes}
    counts = _Counts()
    with _FrameSink(output_root, method, split, video, stems, len(groups) > 1) as sink:
        for position, (start, obj_ids) in enumerate(groups.items()):
            if position:
                predictor.reset_state(state)
            for obj_id in obj_ids:
                box = sam2_box(prompts.boxes[obj_id])
                predictor.add_new_points_or_box(state, start, obj_id, box=box)
            _propagate_group(predictor, state, start, len(stems), sink, rows, counts, video)
        sink.finish()
    table = _signal_table(video, rows, prompts, len(stems))
    _write_signals_atomic(signals_path(output_root, method, split, video), table)
    unavailable = sum(start * len(ids) for start, ids in groups.items())
    return RunResult(table, counts.stored, unavailable)


@dataclass
class _Counts:
    stored: int = 0


SignalRow = tuple[str, int, int, float, int, float, float, bool]


def signal_row(
    video: str, obj_id: int, frame_idx: int, mask: npt.NDArray[np.bool_], score: float
) -> SignalRow:
    """One signal row from one frame's mask; same arithmetic as signals_from_masks."""
    ys, xs = np.nonzero(mask)
    area = int(ys.size)
    cx = float(xs.mean()) if area else float("nan")
    cy = float(ys.mean()) if area else float("nan")
    return (video, obj_id, frame_idx, float(score), area, cx, cy, area == 0)


def _propagate_group(
    predictor: VideoPredictor,
    state: Any,
    start: int,
    num_frames: int,
    sink: _FrameSink,
    rows: dict[int, list[SignalRow]],
    counts: _Counts,
    video: str,
) -> None:
    seen = 0
    # to_numpy moves every output to the CPU, which waits for the GPU, so timings taken around
    # this loop include the finished computation without an explicit synchronise.
    for frame_idx, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=start):
        if frame_idx != start + seen or frame_idx >= num_frames:
            raise RuntimeError(f"unexpected frame index {frame_idx} at position {start + seen}")
        ids = list(obj_ids)
        arrays = {obj_id: to_numpy(logits[row, 0]) for row, obj_id in enumerate(ids)}
        sink.add(frame_idx, arrays)
        for obj_id, obj_logits in arrays.items():
            score, stored = object_score(state, obj_id, frame_idx, obj_logits)
            rows[obj_id].append(signal_row(video, obj_id, frame_idx, obj_logits > 0, score))
            counts.stored += stored
        seen += 1
    if start + seen != num_frames:
        raise RuntimeError(f"expected {num_frames - start} frames from {start}, got {seen}")


def _signal_table(
    video: str, rows: dict[int, list[SignalRow]], prompts: Prompts, num_frames: int
) -> pd.DataFrame:
    nan = float("nan")
    parts = []
    for obj_id in sorted(rows):
        before = [
            (video, obj_id, f, nan, 0, nan, nan, True) for f in range(prompts.start_frame(obj_id))
        ]
        parts.append(pd.DataFrame(before + rows[obj_id], columns=list(SIGNAL_COLUMNS)))
    table = pd.concat(parts, ignore_index=True)
    if len(table) != len(rows) * num_frames:
        raise RuntimeError(f"expected {len(rows) * num_frames} signal rows, got {len(table)}")
    return table


class _FrameSink:
    """Writes one label-map PNG per frame without keeping frames in memory.

    With one SAM 2 pass the PNG is written as frames arrive. With several passes, each frame's
    label map and winning logit go to a scratch folder; a later pass merges into them (highest
    logit wins, ties go to the smaller id, as inside one pass) and `finish` writes the PNGs.
    """

    def __init__(
        self,
        output_root: Path,
        method: str,
        split: str,
        video: str,
        stems: Sequence[str],
        merge: bool,
    ) -> None:
        self._target = (output_root, method, split, video)
        self._stems = stems
        self._merge = merge
        self._shape: tuple[int, int] | None = None
        self._scratch = tempfile.TemporaryDirectory(prefix="sam2fr_") if merge else None
        self._written: set[int] = set()

    def __enter__(self) -> _FrameSink:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._scratch is not None:
            self._scratch.cleanup()

    def add(self, frame_idx: int, arrays: dict[int, npt.NDArray[Any]]) -> None:
        first = next(iter(arrays.values()))
        shape = (int(first.shape[0]), int(first.shape[1]))
        self._shape = shape
        label = masks_to_label_map({i: a > 0 for i, a in arrays.items()}, shape, scores=arrays)
        if not self._merge:
            self._write_png(frame_idx, label)
            return
        best = np.full(shape, -np.inf, dtype=np.float32)
        for obj_id, a in arrays.items():
            won = label == obj_id
            best[won] = a[won]
        self._merge_frame(frame_idx, label, best)

    def finish(self) -> None:
        if self._shape is None:
            raise RuntimeError("SAM 2 returned no frames")
        for index in range(len(self._stems)):
            if self._merge and self._scratch_file(index, "label").is_file():
                self._write_png(index, np.load(self._scratch_file(index, "label")))
            elif index not in self._written:
                self._write_png(index, np.zeros(self._shape, dtype=np.uint8))

    def _scratch_file(self, index: int, kind: str) -> Path:
        assert self._scratch is not None
        return Path(self._scratch.name) / f"{index}.{kind}.npy"

    def _merge_frame(
        self, index: int, label: npt.NDArray[np.uint8], best: npt.NDArray[np.float32]
    ) -> None:
        label_file, best_file = self._scratch_file(index, "label"), self._scratch_file(
            index, "best"
        )
        if label_file.is_file():
            old_label, old_best = np.load(label_file), np.load(best_file)
            take = (label > 0) & (
                (best > old_best) | ((best == old_best) & ((old_label == 0) | (label < old_label)))
            )
            label = np.where(take, label, old_label).astype(np.uint8)
            best = np.where(take, best, old_best).astype(np.float32)
        np.save(label_file, label)
        np.save(best_file, best)

    def _write_png(self, index: int, label: npt.NDArray[np.uint8]) -> None:
        output_root, method, split, video = self._target
        write_label_map(mask_path(output_root, method, split, video, self._stems[index]), label)
        self._written.add(index)


def _write_signals_atomic(path: Path, table: pd.DataFrame) -> None:
    temp = path.with_name(path.name + TEMP_SUFFIX)
    write_signals(temp, table)
    os.replace(temp, path)
