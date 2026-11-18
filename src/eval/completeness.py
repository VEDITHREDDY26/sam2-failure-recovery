"""Completeness check run before scoring: every GT-annotated frame needs a usable prediction.

DAVIS GT: <gt_root>/<video>/<stem>.png (indexed label maps).
SA-V GT:  <gt_root>/<video>/<object folder>/<stem>.png (binary, one object per folder).
Predictions are label maps: <pred_root>/<video>/<stem>.png, mode "P", same size as the GT.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

MAX_LISTED = 20


class IncompleteError(ValueError):
    """Predictions are missing or malformed; the message lists the problems."""

    def __init__(self, problems: Sequence[str]) -> None:
        """Build the error message from the problems.

        Args:
            problems: One text per problem; at most MAX_LISTED are shown in the message and the rest
                are counted. All of them stay in the `problems` attribute.
        """
        self.problems = list(problems)
        shown = self.problems[:MAX_LISTED]
        more = len(self.problems) - len(shown)
        tail = f"\n... and {more} more" if more > 0 else ""
        super().__init__(
            f"{len(self.problems)} problem(s) in the predictions:\n" + "\n".join(shown) + tail
        )


def _pngs(folder: Path) -> list[Path]:
    """PNG files directly inside a folder, sorted by name."""
    return sorted(folder.glob("*.png"))


def davis_gt_frames(gt_root: Path, video: str) -> list[str]:
    """Frame stems that have DAVIS ground truth for a video.

    Args:
        gt_root: Folder with one subfolder of label-map PNGs per video.
        video: Video name.
    """
    return [p.stem for p in _pngs(gt_root / video)]


def sav_object_folders(gt_root: Path, video: str) -> list[str]:
    """Names of the per-object subfolders of an SA-V video's ground truth, sorted.

    Args:
        gt_root: Folder with one subfolder per video.
        video: Video name.
    """
    return sorted(p.name for p in (gt_root / video).iterdir() if p.is_dir())


def sav_gt_frames(gt_root: Path, video: str) -> list[str]:
    """Frame stems that have ground truth for at least one object of an SA-V video, sorted.

    Args:
        gt_root: Folder with one subfolder per video.
        video: Video name.
    """
    stems: set[str] = set()
    for folder in sav_object_folders(gt_root, video):
        stems.update(p.stem for p in _pngs(gt_root / video / folder))
    return sorted(stems)


def _image_size(path: Path) -> tuple[int, int]:
    """Return the (width, height) of an image in pixels."""
    with Image.open(path) as image:
        width, height = image.size
    return width, height


def _gt_reference(gt_root: Path, video: str, sav: bool) -> Path:
    """First ground-truth PNG of a video, used as the reference for the expected image size.

    Args:
        gt_root: Ground-truth folder.
        video: Video name.
        sav: True for the SA-V layout (looks in the first object folder).
    """
    # SA-V keeps one subfolder per object; the first one gives the image size.
    folder = gt_root / video
    if sav:
        folder = folder / sav_object_folders(gt_root, video)[0]
    return _pngs(folder)[0]


def _check_frame(path: Path, size: tuple[int, int], max_id: int | None) -> str | None:
    """Check one predicted PNG and describe the first problem found.

    Args:
        path: Predicted PNG.
        size: Expected (width, height) from the ground truth.
        max_id: Highest allowed object id (DAVIS), or None for no check (SA-V).

    Returns:
        A message if the file is not an indexed (mode P) image, has the wrong size, holds an
        id above max_id or cannot be read; None if it is fine.
    """
    # Opening the file catches unreadable headers; for DAVIS, reading the pixels for the id check
    # also catches truncated data.
    try:
        with Image.open(path) as image:
            if image.mode != "P":
                return f"{path}: not an indexed (mode P) PNG, mode is {image.mode}"
            if image.size != size:
                return f"{path}: size {image.size} differs from the ground truth {size}"
            if max_id is not None and int(np.max(np.array(image))) > max_id:
                return f"{path}: object id above the {max_id} objects in the ground truth"
    except OSError as error:
        return f"{path}: unreadable ({error})"
    return None


def _davis_object_count(gt_root: Path, video: str) -> int:
    """Number of objects of a DAVIS video: the highest label in its first ground-truth frame.

    The value 255 is ignored.

    Args:
        gt_root: Ground-truth folder.
        video: Video name.
    """
    first = _pngs(gt_root / video)[0]
    with Image.open(first) as image:
        values = np.array(image)
    return int(np.max(values[values != 255], initial=0))


def _check_video(gt_root: Path, pred_root: Path, video: str, sav: bool) -> list[str]:
    """List the problems of one video's predictions.

    Args:
        gt_root: Ground-truth folder.
        pred_root: Folder with one subfolder of predicted label-map PNGs per video.
        video: Video name.
        sav: True for the SA-V layout, False for DAVIS.

    Returns:
        One message per problem: missing folders, missing frames, or frames that fail the checks.
    """
    if not (gt_root / video).is_dir():
        return [f"{video}: no ground truth at {gt_root / video}"]
    if not (pred_root / video).is_dir():
        return [f"{video}: missing prediction folder {pred_root / video}"]
    # Every frame that has ground truth must have a prediction of the same size.
    frames = sav_gt_frames(gt_root, video) if sav else davis_gt_frames(gt_root, video)
    size = _image_size(_gt_reference(gt_root, video, sav))
    max_id = None if sav else _davis_object_count(gt_root, video)
    problems: list[str] = []
    # Collect all problems instead of stopping at the first.
    for stem in frames:
        path = pred_root / video / f"{stem}.png"
        if not path.is_file():
            problems.append(f"{video}: missing frame {path}")
            continue
        problem = _check_frame(path, size, max_id)
        if problem:
            problems.append(problem)
    return problems


def find_problems(gt_root: Path, pred_root: Path, videos: Sequence[str], sav: bool) -> list[str]:
    """List the problems of the predictions of several videos.

    Args:
        gt_root: Ground-truth folder.
        pred_root: Folder with the predicted label maps, one subfolder per video.
        videos: Video names to check.
        sav: True for the SA-V layout, False for DAVIS.

    Returns:
        The problems of all videos; empty if everything is in place.
    """
    problems: list[str] = []
    for video in videos:
        problems.extend(_check_video(gt_root, pred_root, video, sav))
    return problems


def check_completeness(gt_root: Path, pred_root: Path, videos: Sequence[str], *, sav: bool) -> None:
    """Raise IncompleteError listing every missing video/frame, wrong size or non-P PNG."""
    problems = find_problems(gt_root, pred_root, videos, sav)
    if problems:
        raise IncompleteError(problems)
