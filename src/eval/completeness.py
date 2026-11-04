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
        self.problems = list(problems)
        shown = self.problems[:MAX_LISTED]
        more = len(self.problems) - len(shown)
        tail = f"\n... and {more} more" if more > 0 else ""
        super().__init__(
            f"{len(self.problems)} problem(s) in the predictions:\n" + "\n".join(shown) + tail
        )


def _pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"))


def davis_gt_frames(gt_root: Path, video: str) -> list[str]:
    return [p.stem for p in _pngs(gt_root / video)]


def sav_object_folders(gt_root: Path, video: str) -> list[str]:
    return sorted(p.name for p in (gt_root / video).iterdir() if p.is_dir())


def sav_gt_frames(gt_root: Path, video: str) -> list[str]:
    stems: set[str] = set()
    for folder in sav_object_folders(gt_root, video):
        stems.update(p.stem for p in _pngs(gt_root / video / folder))
    return sorted(stems)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return width, height


def _gt_reference(gt_root: Path, video: str, sav: bool) -> Path:
    folder = gt_root / video
    if sav:
        folder = folder / sav_object_folders(gt_root, video)[0]
    return _pngs(folder)[0]


def _check_frame(path: Path, size: tuple[int, int], max_id: int | None) -> str | None:
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
    first = _pngs(gt_root / video)[0]
    with Image.open(first) as image:
        values = np.array(image)
    return int(np.max(values[values != 255], initial=0))


def _check_video(gt_root: Path, pred_root: Path, video: str, sav: bool) -> list[str]:
    if not (gt_root / video).is_dir():
        return [f"{video}: no ground truth at {gt_root / video}"]
    if not (pred_root / video).is_dir():
        return [f"{video}: missing prediction folder {pred_root / video}"]
    frames = sav_gt_frames(gt_root, video) if sav else davis_gt_frames(gt_root, video)
    size = _image_size(_gt_reference(gt_root, video, sav))
    max_id = None if sav else _davis_object_count(gt_root, video)
    problems: list[str] = []
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
    problems: list[str] = []
    for video in videos:
        problems.extend(_check_video(gt_root, pred_root, video, sav))
    return problems


def check_completeness(gt_root: Path, pred_root: Path, videos: Sequence[str], *, sav: bool) -> None:
    """Raise IncompleteError listing every missing video/frame, wrong size or non-P PNG."""
    problems = find_problems(gt_root, pred_root, videos, sav)
    if problems:
        raise IncompleteError(problems)
