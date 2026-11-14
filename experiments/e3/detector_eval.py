"""E3 detector side: recall of the frozen rule detector with each rule removed in turn.

    python -m experiments.e3.detector_eval [--model tiny] [--splits davis_val sav_test]
        [--subset-file f] [--results-dir results]

Runs next to the SAM 2 alone masks and the ground truth. Thresholds come only from the frozen
configs/detector/rules.yaml (the run is refused if it is missing; nothing is fitted here). Only
the test sets are accepted: SA-V val is the tuning data. Per split it builds the per-frame table
(signals CSV joined with boxes derived from the cached masks, one mask read per frame), the
failure labels from src.detection.labels, and counts windows with the same definition as the
threshold tuning (experiments.tune_rules.tune.build_windows / window_any). On DAVIS every frame
is annotated, so each window is a single frame. Writes results/e3_detector_<split>.csv with one
row per removed rule ("none" = the full detector).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.tune_rules.export_sav_val import BOX_COLUMNS
from experiments.tune_rules.tune import build_windows, window_any
from src.common.boxes import box_from_mask
from src.common.config import DATA_ROOT_ENV
from src.common.layout import mask_path, signals_path, validate_method
from src.common.masks import read_label_map
from src.common.signals import read_signals
from src.data.video import NeverVisibleError, VideoSample
from src.detection.labels import video_labels
from src.detection.rules import (
    ALL_RULES,
    DEFAULT_RULES_PATH,
    RuleThresholds,
    batch_flags,
    load_rules_config,
)
from src.detection.signals import count_overlapped, join_signals_boxes
from src.eval.check_complete import MODELS, SUBSET_FILE, check_run, load_video, split_videos
from src.eval.score import load_data_config
from src.sam2_runner.predictor import REPO_ROOT

TEST_SPLITS = ("davis_val", "sav_test")
NONE = "none"
REMOVED = (NONE, *ALL_RULES)
RESULT_COLUMNS = (
    "model",
    "split",
    "removed_rule",
    "windows",
    "positives",
    "flagged",
    "tp",
    "precision",
    "recall",
)
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


class DetectorEvalError(ValueError):
    """An input is missing, incomplete or not allowed for this evaluation."""


def detector_path(results_dir: Path, split: str) -> Path:
    return results_dir / f"e3_detector_{split}.csv"


def video_boxes(video: VideoSample, output_root: Path, method: str, split: str) -> pd.DataFrame:
    """Tight predicted box per (object, frame) from the object's prompt frame on."""
    starts: dict[int, int] = {}
    for obj_id in video.object_ids:
        try:
            starts[obj_id] = video.first_box(obj_id).frame_index
        except NeverVisibleError:
            continue
    rows: list[tuple[object, ...]] = []
    if starts:
        for frame_index in range(min(starts.values()), video.num_frames):
            stem = video.frame_stems[frame_index]
            label = read_label_map(mask_path(output_root, method, split, video.name, stem))
            height, width = label.shape
            for obj_id, start in starts.items():
                if frame_index < start:
                    continue
                box = box_from_mask(label == obj_id)
                corners = (box.x0, box.y0, box.x1, box.y1) if box else (None,) * 4
                rows.append((video.name, obj_id, frame_index, height, width, *corners))
    frame = pd.DataFrame(rows, columns=list(BOX_COLUMNS))
    return frame.astype({c: "Int64" for c in BOX_COLUMNS[3:]})


def evaluate_table(
    table: pd.DataFrame, labels: pd.DataFrame, thresholds: RuleThresholds
) -> list[dict[str, object]]:
    """One row per removed rule: window counts, precision and recall (no model / split yet)."""
    windows = build_windows(table, labels)
    positives = int(windows.truth.sum())
    rows: list[dict[str, object]] = []
    for removed in REMOVED:
        enabled = [r for r in ALL_RULES if r != removed]
        flags = batch_flags(table, thresholds, enabled)["flag"].to_numpy(dtype=bool)
        hit = window_any(flags, windows)
        flagged, tp = int(hit.sum()), int((hit & windows.truth).sum())
        rows.append(
            {
                "removed_rule": removed,
                "windows": len(windows.truth),
                "positives": positives,
                "flagged": flagged,
                "tp": tp,
                "precision": tp / flagged if flagged else 0.0,
                "recall": tp / positives if positives else 0.0,
            }
        )
    return rows


def split_inputs(model: str, split: str, subset_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joined per-frame table and failure labels of the SAM 2 alone run on one split."""
    cfg = load_data_config()
    method = validate_method(f"sam2_{model}")
    check = check_run(cfg, method, split, subset_file)
    if check.missing:
        raise DetectorEvalError("\n".join(check.lines()) + f"\n{method}/{split}: run incomplete")
    output_root = cfg.data_root / cfg.output_root
    tables, labels, boxes = [], [], []
    for name in split_videos(cfg, split, subset_file):
        video = load_video(cfg, split, name)
        boxes.append(video_boxes(video, output_root, method, split))
        labels.append(video_labels(video, output_root, method, split))
        tables.append(read_signals(signals_path(output_root, method, split, name)))
    # Same opt-in as the SA-V val tuning: an object fully covered by another object has a SAM 2
    # mask (area > 0 in the signal file) but no pixels in the stored label map, so its box is
    # unknown and the disappearance rule cannot fire from that frame.
    table = join_signals_boxes(
        pd.concat(tables, ignore_index=True),
        pd.concat(boxes, ignore_index=True),
        allow_overlapped=True,
    )
    sys.stdout.write(
        f"{method}/{split}: overlapped rows (area > 0, no box) = {count_overlapped(table)}\n"
    )
    return table, pd.concat(labels, ignore_index=True)


def check_splits(splits: Sequence[str]) -> None:
    for split in splits:
        if split not in TEST_SPLITS:
            raise DetectorEvalError(
                f"refused: {split} is not a test set (SA-V val is the tuning data); "
                f"allowed: {list(TEST_SPLITS)}"
            )


def write_rows(path: Path, model: str, split: str, rows: Sequence[dict[str, object]]) -> None:
    frame = pd.DataFrame([{"model": model, "split": split, **r} for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame[list(RESULT_COLUMNS)].to_csv(temp, index=False)
    os.replace(temp, path)


def run(
    model: str,
    splits: Sequence[str],
    results_dir: Path,
    subset_file: Path,
    rules_path: Path = DEFAULT_RULES_PATH,
) -> list[Path]:
    check_splits(splits)
    if not rules_path.is_file():
        raise DetectorEvalError(f"detector not frozen yet: {rules_path} does not exist")
    thresholds = load_rules_config(model, rules_path)
    written = []
    for split in splits:
        table, labels = split_inputs(model, split, subset_file)
        path = detector_path(results_dir, split)
        write_rows(path, model, split, evaluate_table(table, labels, thresholds))
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", choices=MODELS, default="tiny")
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS))
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    try:
        written = run(args.model, args.splits, args.results_dir, args.subset_file, args.rules)
    except (DetectorEvalError, OSError) as error:
        sys.stdout.write(f"e3 detector_eval: {error}\n")
        return 1
    for path in written:
        sys.stdout.write(f"wrote {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
