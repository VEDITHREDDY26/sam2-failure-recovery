"""E8 helper: which re-prompts produced a wrong mask (needs ground truth, so Newton only).

    python -m experiments.e8.wrong_reprompts --config configs/recovery/rules_csrt_tiny.yaml \
        --splits davis_val sav_test [--results-dir results]

A re-prompt is wrong when the method's own mask at the re-prompted frame has IoU < 0.5 with the
ground truth (the same threshold as a failure label). When the frame has no annotation (SA-V),
the first annotated frame at or after it is used and recorded in gt_frame_index; with none left
the re-prompt is unscored (blank iou_after and wrong). Writes
results/<method>_<split>_reprompts.csv with one row per re-prompt event. sav_val is refused.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.common.config import DATA_ROOT_ENV, DataConfig
from src.common.layout import events_path, mask_path
from src.common.masks import read_label_map
from src.data.video import VideoSample
from src.detection.labels import FAILURE_IOU, mask_iou
from src.eval.check_complete import SUBSET_FILE, load_video, split_videos
from src.eval.score import load_data_config
from src.recovery.config import load_recovery_config
from src.recovery.events import EVENT_COLUMNS
from src.recovery.online_runner import check_online_method
from src.sam2_runner.predictor import REPO_ROOT

E8_SPLITS = ("davis_val", "sav_test")
REPROMPT_COLUMNS = ("video", "object_id", "frame_index", "gt_frame_index", "iou_after", "wrong")
RESULTS_DIR = REPO_ROOT / "results"


def reprompts_name(method: str, split: str) -> str:
    return f"{check_online_method(method)}_{split}_reprompts.csv"


def judge_event(
    video: VideoSample, output_root: Path, method: str, split: str, obj_id: int, frame_index: int
) -> tuple[int | str, float | str, bool | str]:
    """(gt frame used, iou, wrong), blanks if no annotated frame at or after the re-prompt."""
    later = [f for f in video.annotated_frames(obj_id) if f >= frame_index]
    if not later:
        return "", "", ""
    gt_frame = later[0]
    gt = video.gt_mask(obj_id, gt_frame)
    if gt is None:
        return "", "", ""
    stem = video.frame_stems[gt_frame]
    label = read_label_map(mask_path(output_root, method, split, video.name, stem))
    iou = mask_iou(label == obj_id, gt)
    return gt_frame, iou, iou < FAILURE_IOU


def reprompt_table(cfg: DataConfig, method: str, split: str, videos: Sequence[str]) -> pd.DataFrame:
    output_root = cfg.data_root / cfg.output_root
    rows: list[tuple[object, ...]] = []
    for name in videos:
        events = pd.read_csv(events_path(output_root, method, split, name), dtype={"video": str})
        if list(events.columns) != list(EVENT_COLUMNS):
            raise ValueError(f"unusable events file for {name}")
        events = events[events["action"] == "reprompted"]
        if events.empty:
            continue
        video = load_video(cfg, split, name)
        for event in events.itertuples():
            obj_id, frame = int(event.object_id), int(event.frame_index)  # type: ignore[call-overload]
            judged = judge_event(video, output_root, method, split, obj_id, frame)
            rows.append((name, obj_id, frame, *judged))
    return pd.DataFrame(rows, columns=list(REPROMPT_COLUMNS))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(E8_SPLITS))
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    for split in args.splits:
        if split not in E8_SPLITS:
            raise SystemExit(f"split '{split}' refused: E8 uses only {E8_SPLITS}")
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    method = check_online_method(load_recovery_config(args.config).method)
    cfg = load_data_config()
    for split in args.splits:
        videos = split_videos(cfg, split, args.subset_file)
        table = reprompt_table(cfg, method, split, videos)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.results_dir / reprompts_name(method, split), index=False)
        wrong = int((table["wrong"] == True).sum())  # noqa: E712
        sys.stdout.write(f"{method}/{split}: {len(table)} re-prompts, {wrong} wrong\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
