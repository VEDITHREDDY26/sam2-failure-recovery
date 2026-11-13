"""E10 failure types, step 1: candidates, the seeded sample of 40, clips and a labelling template.

    python -m experiments.e10_types.sample --clips-dir <dir outside the repo> \
        [--results-dir results] [--sample-out results/e10_types_sample.csv]

A flagged failure is one trigger row (reprompted, rejected or tracker_lost; cooldown excluded) of
results/events_fix_rules_csrt_tiny_<split>.csv whose SAM 2-alone mask (sam2_tiny) has IoU < 0.5 at
that frame, or at the next annotated frame for SA-V. 40 are drawn with random.Random(5415) from the
sorted list. Each gets a PNG strip (frames t-12..t+12) and a row in labels_template.csv (written to
the clips dir). Runs next to the masks and ground truth (Newton, CPU).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from src.common.config import DATA_ROOT_ENV
from src.common.layout import mask_path
from src.common.masks import read_label_map
from src.data.video import VideoSample
from src.detection.labels import is_failure, mask_iou
from src.recovery.events import EVENT_COLUMNS

SEED = 5415
SAMPLE_SIZE = 40
SPLITS = ("davis_val", "sav_test")
ALONE_METHOD = "sam2_tiny"
FIX_METHOD = "fix_rules_csrt_tiny"
TRIGGER_ACTIONS = ("reprompted", "rejected", "tracker_lost")
TYPES = ("occlusion", "fast_motion", "distractor", "leaves_frame", "other")
CANDIDATE_KEY = ["split", "video", "object_id", "frame_index"]
SAMPLE_COLUMNS = (
    "sample_id",
    "split",
    "video",
    "object_id",
    "frame_index",
    "action",
    "fired",
    "gt_frame_index",
    "iou",
)
TEMPLATE_COLUMNS = ("sample_id", "type", "notes")
CLIP_RADIUS = 12
CLIP_STEP = 2
STRIP_COLUMNS = 5
THUMB_WIDTH = 320
LoadVideo = Callable[[str, str], VideoSample]
Image = npt.NDArray[np.uint8]
REPO_ROOT = Path(__file__).resolve().parents[2]


def check_outside_repo(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise SystemExit(f"{path} is inside the repository; write the clips outside it")
    return path


def read_events(results_dir: Path, split: str) -> pd.DataFrame:
    path = results_dir / f"events_{FIX_METHOD}_{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing events file: {path}")
    table = pd.read_csv(path, dtype={"video": str}, keep_default_na=False)
    if list(table.columns) != list(EVENT_COLUMNS):
        raise ValueError(f"{path}: columns are {list(table.columns)}")
    return table


def triggers(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (video, object, frame) among the trigger actions (cooldown rows dropped)."""
    kept = events[events["action"].isin(TRIGGER_ACTIONS)]
    return kept.drop_duplicates(["video", "object_id", "frame_index"]).reset_index(drop=True)


def label_frame(video: VideoSample, obj_id: int, frame_index: int) -> int | None:
    """The trigger frame if annotated, else the next annotated frame of the object, else None."""
    later = [f for f in video.annotated_frames(obj_id) if f >= frame_index]
    return later[0] if later else None


def build_candidates(
    results_dir: Path, output_root: Path, load_video: LoadVideo, splits: Sequence[str] = SPLITS
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in splits:
        loaded: dict[str, VideoSample] = {}
        for event in triggers(read_events(results_dir, split)).itertuples():
            name, obj_id, frame = str(event.video), int(event.object_id), int(event.frame_index)  # type: ignore[call-overload]
            video = loaded.setdefault(name, load_video(split, name))
            gt_frame = label_frame(video, obj_id, frame)
            if gt_frame is None:
                continue
            gt = video.gt_mask(obj_id, gt_frame)
            if gt is None:
                continue
            stem = video.frame_stems[gt_frame]
            alone = read_label_map(mask_path(output_root, ALONE_METHOD, split, name, stem))
            iou = mask_iou(alone == obj_id, gt)
            if is_failure(iou):
                rows.append(
                    {
                        "split": split,
                        "video": name,
                        "object_id": obj_id,
                        "frame_index": frame,
                        "action": event.action,
                        "fired": event.fired,
                        "gt_frame_index": gt_frame,
                        "iou": iou,
                    }
                )
    columns = [*CANDIDATE_KEY, "action", "fired", "gt_frame_index", "iou"]
    return pd.DataFrame(rows, columns=columns).sort_values(CANDIDATE_KEY, ignore_index=True)


def draw_sample(candidates: pd.DataFrame, size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Seeded sample from the sorted candidates; all of them when there are fewer than size."""
    ordered = candidates.sort_values(CANDIDATE_KEY, ignore_index=True)
    picks = sorted(random.Random(SEED).sample(range(len(ordered)), min(size, len(ordered))))
    sample = ordered.iloc[picks].reset_index(drop=True)
    sample.insert(0, "sample_id", [f"s{i + 1:02d}" for i in range(len(sample))])
    return sample[list(SAMPLE_COLUMNS)]


def clip_indices(frame_index: int, num_frames: int) -> list[int]:
    wanted = range(frame_index - CLIP_RADIUS, frame_index + CLIP_RADIUS + 1, CLIP_STEP)
    return [i for i in wanted if 0 <= i < num_frames]


def _thumb(video: VideoSample, alone_dir: Path, row: pd.Series, index: int) -> Image:
    image = cv2.imread(str(video.frame_path(index)), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read frame {video.frame_path(index)}")
    frame = np.asarray(image, dtype=np.uint8)
    obj_id = int(row["object_id"])
    stem = video.frame_stems[index]
    label = read_label_map(mask_path(alone_dir, ALONE_METHOD, str(row["split"]), video.name, stem))
    found, _ = cv2.findContours(
        (label == obj_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(frame, found, -1, (0, 255, 0), 2)
    box = [row.get(k, "") for k in ("box_x0", "box_y0", "box_x1", "box_y1")]
    if all(str(v) != "" and not pd.isna(v) for v in box) and index == int(row["frame_index"]):
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 128, 0), 2)
    scale = THUMB_WIDTH / frame.shape[1]
    frame = cv2.resize(frame, (THUMB_WIDTH, max(1, int(frame.shape[0] * scale))))
    colour = (0, 0, 255) if index == int(row["frame_index"]) else (255, 255, 255)
    cv2.putText(frame, f"frame {index}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    if index == int(row["frame_index"]):
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 3)
    return frame


def render_strip(row: pd.Series, video: VideoSample, output_root: Path, path: Path) -> Path:
    thumbs = [
        _thumb(video, output_root, row, i)
        for i in clip_indices(int(row["frame_index"]), video.num_frames)
    ]
    height, width = thumbs[0].shape[:2]
    rows = []
    for start in range(0, len(thumbs), STRIP_COLUMNS):
        chunk = thumbs[start : start + STRIP_COLUMNS]
        chunk += [np.zeros((height, width, 3), dtype=np.uint8)] * (STRIP_COLUMNS - len(chunk))
        rows.append(np.hstack(chunk))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def event_boxes(results_dir: Path, sample: pd.DataFrame) -> pd.DataFrame:
    events = pd.concat(
        [triggers(read_events(results_dir, s)).assign(split=s) for s in SPLITS], ignore_index=True
    )
    keep = ["split", "video", "object_id", "frame_index", "box_x0", "box_y0", "box_x1", "box_y1"]
    return sample.merge(events[keep], on=["split", "video", "object_id", "frame_index"], how="left")


def build_sample(
    results_dir: Path,
    output_root: Path,
    load_video: LoadVideo,
    clips_dir: Path,
    sample_out: Path,
) -> pd.DataFrame:
    check_outside_repo(clips_dir)
    sample = draw_sample(build_candidates(results_dir, output_root, load_video))
    write_csv(sample_out, SAMPLE_COLUMNS, sample.itertuples(index=False))
    write_csv(
        clips_dir / "labels_template.csv",
        TEMPLATE_COLUMNS,
        [(s, "", "") for s in sample["sample_id"]],
    )
    with_boxes = event_boxes(results_dir, sample)
    for _, row in with_boxes.iterrows():
        video = load_video(str(row["split"]), str(row["video"]))
        render_strip(row, video, output_root, clips_dir / f"{row['sample_id']}.png")
    return sample


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--sample-out", type=Path, default=Path("results/e10_types_sample.csv"))
    args = parser.parse_args(argv)
    check_outside_repo(args.clips_dir)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    from experiments.e10_overlays.overlays import default_loader
    from src.sam2_runner.smoke import load_data_config

    cfg = load_data_config()
    sample = build_sample(
        args.results_dir,
        cfg.data_root / cfg.output_root,
        default_loader(cfg),
        args.clips_dir,
        args.sample_out,
    )
    print(f"{len(sample)} samples -> {args.sample_out}, clips and template in {args.clips_dir}")
    if len(sample) < SAMPLE_SIZE:
        print(f"only {len(sample)} candidates exist; all taken")


if __name__ == "__main__":
    main()
