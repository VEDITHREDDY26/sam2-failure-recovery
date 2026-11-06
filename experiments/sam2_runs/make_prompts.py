"""Build the batch prompts file for one split from the dataset loaders.

    python -m experiments.sam2_runs.make_prompts --split sav_val [--out PATH]

Every object is prompted with the tight box of its first visible annotated frame, at that
frame. Objects that are never visible are left out and listed under "never_visible"; a video
with no promptable object is an error. davis_val = all DAVIS val videos, sav_val = all SA-V val
videos, sav_test = exactly the videos of the seeded 60-video subset file.
Default output: <data_root>/<output_root>/prompts/<split>.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.common.config import DATA_ROOT_ENV, DataConfig
from src.common.layout import validate_split
from src.data.davis import list_davis_videos, load_davis_video
from src.data.make_subset import load_subset
from src.data.sav import list_sav_videos, load_sav_video
from src.data.video import NeverVisibleError, VideoSample
from src.sam2_runner.predictor import REPO_ROOT
from src.sam2_runner.smoke import load_data_config

SUBSET_FILE = REPO_ROOT / "configs" / "splits" / "sav_test_subset60.txt"


def split_videos(cfg: DataConfig, split: str, subset_file: Path = SUBSET_FILE) -> list[str]:
    """The videos a run of this split covers (shared with the completeness check)."""
    validate_split(split)
    if split == "davis_val":
        return list(list_davis_videos(cfg))
    if split == "sav_val":
        return list(list_sav_videos(cfg, "val"))
    wanted = list(load_subset(subset_file))
    available = set(list_sav_videos(cfg, "test"))
    missing = [v for v in wanted if v not in available]
    if missing:
        raise ValueError(
            f"{len(missing)} subset video(s) not in the SA-V test list: {missing[:10]}"
        )
    return wanted


def load_video(cfg: DataConfig, split: str, name: str) -> VideoSample:
    if split == "davis_val":
        return load_davis_video(cfg, name)
    return load_sav_video(cfg, "val" if split == "sav_val" else "test", name)


def video_prompts(video: VideoSample) -> tuple[dict[str, Any], list[int]]:
    boxes: dict[str, Any] = {}
    never: list[int] = []
    for obj_id in video.object_ids:
        try:
            first = video.first_box(obj_id)
        except NeverVisibleError:
            never.append(obj_id)
            continue
        box = first.box
        boxes[str(obj_id)] = {
            "frame_index": first.frame_index,
            "box": [box.x0, box.y0, box.x1, box.y1],
        }
    if not boxes:
        raise ValueError(
            f"video '{video.name}' has no object that is ever visible: nothing to prompt"
        )
    return {"boxes": boxes}, never


def build_prompts(cfg: DataConfig, split: str, subset_file: Path = SUBSET_FILE) -> dict[str, Any]:
    videos: dict[str, Any] = {}
    never_visible: dict[str, list[int]] = {}
    for name in split_videos(cfg, split, subset_file):
        entry, never = video_prompts(load_video(cfg, split, name))
        videos[name] = entry
        if never:
            never_visible[name] = never
    return {"split": split, "videos": videos, "never_visible": never_visible}


def default_output(cfg: DataConfig, split: str) -> Path:
    return cfg.data_root / cfg.output_root / "prompts" / f"{split}.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--split", choices=["davis_val", "sav_val", "sav_test"], required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    cfg = load_data_config()
    data = build_prompts(cfg, args.split, args.subset_file)
    out = args.out or default_output(cfg, args.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    objects = sum(len(v["boxes"]) for v in data["videos"].values())
    later = sum(
        1 for v in data["videos"].values() for b in v["boxes"].values() if b["frame_index"] > 0
    )
    print(
        f"{args.split}: {len(data['videos'])} videos, {objects} objects, "
        f"{later} start after frame 0 -> {out}"
    )
    for name, ids in data["never_visible"].items():
        print(f"never visible, left out: {name} objects {ids}")


if __name__ == "__main__":
    main()
