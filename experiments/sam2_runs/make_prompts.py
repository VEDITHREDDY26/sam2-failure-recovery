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
from src.data.video import NeverVisibleError, VideoSample
from src.eval.check_complete import SUBSET_FILE, load_video, split_videos
from src.sam2_runner.smoke import load_data_config


def video_prompts(video: VideoSample) -> tuple[dict[str, Any], list[int]]:
    """Build the prompt entry of one video.

    Args:
        video: Video with ground truth.

    Returns:
        ({"boxes": {object id: {"frame_index": first visible frame (0-based), "box": [x0, y0, x1,
        y1] with right and bottom exclusive}}}, ids of objects that are never visible).

    Raises:
        ValueError: If no object of the video is ever visible.
    """
    boxes: dict[str, Any] = {}
    never: list[int] = []
    # Prompt each object at the first annotated frame where it is visible.
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
    # A video where nothing is ever visible cannot be run at all.
    if not boxes:
        raise ValueError(
            f"video '{video.name}' has no object that is ever visible: nothing to prompt"
        )
    return {"boxes": boxes}, never


def build_prompts(cfg: DataConfig, split: str, subset_file: Path = SUBSET_FILE) -> dict[str, Any]:
    """Build the prompts of every video of a split.

    Args:
        cfg: Data config.
        split: "davis_val", "sav_val" or "sav_test".
        subset_file: List of the seeded SA-V test subset (used only for "sav_test").

    Returns:
        Dict with "split", "videos" (name -> entry) and "never_visible" (name -> object ids).
    """
    videos: dict[str, Any] = {}
    never_visible: dict[str, list[int]] = {}
    # Loop over the split's video list in its order.
    for name in split_videos(cfg, split, subset_file):
        entry, never = video_prompts(load_video(cfg, split, name))
        videos[name] = entry
        if never:
            never_visible[name] = never
    return {"split": split, "videos": videos, "never_visible": never_visible}


def default_output(cfg: DataConfig, split: str) -> Path:
    """Default path of a split's prompts file: <data_root>/<output_root>/prompts/<split>.json.

    Args:
        cfg: Data config.
        split: Split name.
    """
    return cfg.data_root / cfg.output_root / "prompts" / f"{split}.json"


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: write the prompts JSON for one split.

    Writes the file to --out (default: the path from default_output) and prints the number of
    videos, objects and objects that start after frame 0, then the objects left out.

    Args:
        argv: Command-line arguments; None means sys.argv.

    Raises:
        SystemExit: Status 2 from argparse on bad arguments or when the data-root environment
            variable is not set.
    """
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
    # Objects whose first visible frame is after frame 0 start later in the video.
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
