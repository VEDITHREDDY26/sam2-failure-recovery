"""Smoke run: prompt SAM 2 with first-frame boxes, propagate, store masks and signals.

    python -m src.sam2_runner.smoke --model tiny --video bear --prompts prompts.json

prompts.json: {"video": str, "frame_index": 0, "boxes": {"<obj_id>": [x0, y0, x1, y1]}} with
boxes in the project Box convention (right and bottom edges exclusive).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.common.boxes import Box
from src.common.config import DATA_ROOT_ENV, ConfigError, DataConfig, load_config
from src.common.layout import mask_path, signals_path
from src.common.masks import masks_to_label_map, write_label_map
from src.common.signals import signals_from_masks, write_signals
from src.sam2_runner.predictor import REPO_ROOT, VideoPredictor, build_predictor, to_numpy
from src.sam2_runner.prompts import Prompts, load_prompts, parse_prompts
from src.sam2_runner.scores import object_score

DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"
SPLIT = "davis_val"
__all__ = ["Prompts", "load_prompts", "parse_prompts"]
FRAME_SUFFIXES = (".jpg", ".jpeg")


@dataclass(frozen=True)
class SmokeResult:
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


def run_smoke(
    predictor: VideoPredictor,
    frames_dir: Path,
    prompts: Prompts,
    output_root: Path,
    method: str,
    video: str | None = None,
) -> SmokeResult:
    """Run one video; write one indexed PNG per frame and the signal CSV; return the signals."""
    video = video or prompts.video
    stems = frame_stems(frames_dir)
    state = predictor.init_state(str(frames_dir))
    for obj_id, box in prompts.boxes.items():
        predictor.add_new_points_or_box(state, prompts.frame_index, obj_id, box=sam2_box(box))
    per_object: dict[int, list[npt.NDArray[np.bool_]]] = {i: [] for i in prompts.boxes}
    scores: dict[int, list[float]] = {i: [] for i in prompts.boxes}
    stored_rows = 0
    for count, (frame_idx, obj_ids, logits) in enumerate(predictor.propagate_in_video(state)):
        if frame_idx != count or count >= len(stems):
            raise RuntimeError(f"unexpected frame index {frame_idx} at position {count}")
        _store_frame(output_root, method, video, stems[frame_idx], obj_ids, logits)
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
    write_signals(signals_path(output_root, method, SPLIT, video), table)
    return SmokeResult(table, stored_rows)


def _store_frame(
    output_root: Path, method: str, video: str, stem: str, obj_ids: Sequence[int], logits: Any
) -> None:
    arrays = {obj_id: to_numpy(logits[row, 0]) for row, obj_id in enumerate(obj_ids)}
    first = next(iter(arrays.values()))
    shape = (int(first.shape[0]), int(first.shape[1]))
    label = masks_to_label_map({i: a > 0 for i, a in arrays.items()}, shape, scores=arrays)
    write_label_map(mask_path(output_root, method, SPLIT, video, stem), label)


def load_data_config() -> DataConfig:
    cfg = load_config(DATA_CONFIG)
    if not isinstance(cfg, DataConfig):
        raise ConfigError(f"{DATA_CONFIG}: not a data config")
    return cfg


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", choices=["tiny", "large"], required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default=None)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    cfg = load_data_config()
    prompts = load_prompts(args.prompts)
    if prompts.video != args.video:
        parser.error(f"prompts are for video '{prompts.video}', not '{args.video}'")
    frames_dir = cfg.data_root / cfg.davis.root / "JPEGImages" / cfg.davis.resolution / args.video
    method = args.method or f"sam2_{args.model}_smoke"
    predictor = build_predictor(args.model, cfg.data_root, device=args.device)
    result = run_smoke(predictor, frames_dir, prompts, cfg.data_root / cfg.output_root, method)
    table = result.table
    print(
        f"{method}: {args.video} {table['frame_index'].nunique()} frames, "
        f"objects {sorted(table['object_id'].unique().tolist())}"
    )
    print(result.source_summary)


if __name__ == "__main__":
    main()
