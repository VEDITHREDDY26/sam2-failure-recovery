"""Dataset sanity report: `python -m src.data.check --dataset davis|sav --split val|test [--full]`.

Default mode reads only the frame-0 annotation, the first box and the first frame header of each
video (the data lives on a slow drive). `--full` also reads every ground-truth mask.
Exit code 0 = video count matches and no problems, 1 = otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from src.common.config import DataConfig, load_config
from src.data.davis import list_davis_videos, load_davis_video
from src.data.sav import list_sav_videos, load_sav_video
from src.data.video import VideoSample

# Dataset facts (project plan, "Data"): number of videos per split.
EXPECTED_VIDEOS = {("davis", "val"): 30, ("sav", "val"): 155, ("sav", "test"): 150}
# SAM 2 README (sav_dataset/README.md): number of masklets (objects) per split.
EXPECTED_MASKLETS = {("sav", "val"): 293, ("sav", "test"): 278}
DEFAULT_CONFIG = Path("configs/data.yaml")
LOAD_ERRORS = (FileNotFoundError, ValueError, KeyError, IndexError)


@dataclasses.dataclass
class Report:
    """Counters and problem messages collected while checking one dataset split.

    Attributes:
        dataset: "davis" or "sav".
        split: Split name, "val" or "test".
        videos: Number of videos found in the split.
        objects: Total number of objects (masklets) over all videos.
        frames: Total number of frames over all videos.
        late_objects: Objects whose first annotated frame is not frame 0.
        annotated_frames: Total number of annotated (object, frame) pairs.
        problems: One message per problem found (load errors, size mismatches).
    """

    dataset: str
    split: str
    videos: int = 0
    objects: int = 0
    frames: int = 0
    late_objects: int = 0
    annotated_frames: int = 0
    problems: list[str] = dataclasses.field(default_factory=list)


def _frame_size(video: VideoSample) -> tuple[int, int]:
    """Return the (height, width) of the video's first frame, reading only the image header."""
    with Image.open(video.frame_path(0)) as image:
        width, height = image.size
    return height, width


def _check_video(video: VideoSample, report: Report, full: bool) -> None:
    """Update the report with one video's counts and check its annotation size.

    Args:
        video: The loaded video.
        report: Report to update in place.
        full: If True, also read every annotated mask of every object.
    """
    # Annotation masks must have the same height and width as the frames.
    report.frames += video.num_frames
    report.objects += len(video.object_ids)
    size = _frame_size(video)
    for obj_id in video.object_ids:
        report.annotated_frames += len(video.annotated_frames(obj_id))
        # The first box also proves the object is visible in at least one annotated frame.
        first = video.first_box(obj_id)
        if first.frame_index > 0:
            report.late_objects += 1
        mask = video.gt_mask(obj_id, first.frame_index)
        if mask is not None and mask.shape != size:
            report.problems.append(
                f"{video.name}: frame size {size} differs from annotation size {mask.shape}"
            )
        if full:
            _read_all(video, obj_id, size, report)


def _read_all(video: VideoSample, obj_id: int, size: tuple[int, int], report: Report) -> None:
    """Read every annotated mask of one object and record the first size mismatch.

    Args:
        video: The loaded video.
        obj_id: Object id to check.
        size: Expected mask shape (height, width) in pixels, taken from the first frame.
        report: Report to add a problem to; stops after the first problem for this object.
    """
    # Only annotated frames can be read; stop at the first bad one to keep the output short.
    for frame_index in video.annotated_frames(obj_id):
        mask = video.gt_mask(obj_id, frame_index)
        if mask is None or mask.shape != size:
            report.problems.append(
                f"{video.name}: object {obj_id} frame {frame_index} annotation size mismatch"
            )
            return


def check_davis(cfg: DataConfig, full: bool) -> Report:
    """Check every video of the configured DAVIS split.

    Args:
        cfg: Data configuration.
        full: If True, read every ground-truth mask.

    Returns:
        The filled report; load errors are recorded as problems instead of raised.
    """
    report = Report("davis", cfg.davis.split)
    names = list_davis_videos(cfg)
    report.videos = len(names)
    for name in names:
        try:
            _check_video(load_davis_video(cfg, name), report, full)
        except LOAD_ERRORS as error:
            report.problems.append(f"{name}: {error}")
    return report


def check_sav(cfg: DataConfig, split: str, full: bool) -> Report:
    """Check every video of an SA-V split.

    Args:
        cfg: Data configuration.
        split: "val" or "test".
        full: If True, read every annotated mask.

    Returns:
        The filled report; load errors are recorded as problems instead of raised.
    """
    report = Report("sav", split)
    names = list_sav_videos(cfg, split)
    report.videos = len(names)
    for name in names:
        try:
            _check_video(load_sav_video(cfg, split, name), report, full)
        except LOAD_ERRORS as error:
            report.problems.append(f"{name}: {error}")
    return report


def _print_report(report: Report, expected: int) -> None:
    """Print the report as plain text, with warnings for unexpected counts.

    Args:
        report: The finished report.
        expected: Number of videos the split should contain.
    """
    print(f"dataset: {report.dataset}")
    print(f"split: {report.split}")
    print(f"videos: {report.videos} (expected {expected})")
    print(f"objects: {report.objects}")
    # Only SA-V has a published object count to compare against.
    masklets = EXPECTED_MASKLETS.get((report.dataset, report.split))
    if masklets is not None and report.objects != masklets:
        print(f"WARN: expected {masklets} masklets, found {report.objects}")
    print(f"frames: {report.frames}")
    if report.dataset == "sav":
        print(f"annotated (object, frame) pairs: {report.annotated_frames}")
    print(f"objects first visible after frame 0: {report.late_objects}")
    print(f"problems: {len(report.problems)}")
    for line in report.problems:
        print(f"  {line}")
    if report.videos != expected:
        print(f"FAIL: expected {expected} videos, found {report.videos}")


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments of the dataset check."""
    parser = argparse.ArgumentParser(prog="python -m src.data.check")
    parser.add_argument("--dataset", choices=["davis", "sav"], required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--full", action="store_true", help="read every ground-truth mask")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-videos", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: check a dataset split and print a summary.

    Reads the data configuration (--config) and the dataset it points to, and prints counts and
    problems to standard output; nothing is written.

    Args:
        argv: Command-line arguments; sys.argv is used when None.

    Returns:
        0 if the video count equals the expected count and no problems were found, otherwise 1
        (also 1 if the requested DAVIS split differs from the configured one).
    """
    args = _parse(argv)
    cfg: DataConfig = load_config(args.config)
    # Fall back to the known video count of the split unless one is given.
    expected = args.expected_videos or EXPECTED_VIDEOS.get((args.dataset, args.split), 0)
    if args.dataset == "sav":
        report = check_sav(cfg, args.split, args.full)
    elif args.split != cfg.davis.split:
        print(f"FAIL: config davis split is '{cfg.davis.split}', asked for '{args.split}'")
        return 1
    else:
        report = check_davis(cfg, args.full)
    _print_report(report, expected)
    return 0 if report.videos == expected and not report.problems else 1


if __name__ == "__main__":
    sys.exit(main())
