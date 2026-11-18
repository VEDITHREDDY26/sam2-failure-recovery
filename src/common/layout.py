"""Output folder layout under the configured output_root.

<output_root>/<method>/<split>/masks/<video>/<frame_stem>.png
<output_root>/<method>/<split>/signals/<video>.csv
<output_root>/<method>/<split>/events/<video>.csv
"""

from __future__ import annotations

import re
from pathlib import Path

SPLITS = frozenset({"davis_val", "sav_val", "sav_test"})
ORACLE_TOKEN = "oracle"
_METHOD_PATTERN = re.compile(r"[a-z0-9_]+")


def validate_method(name: str) -> str:
    """Check that a method name is safe to use as a folder name.

    Args:
        name: Method name, for example "sam2_tiny".

    Returns:
        The same name.

    Raises:
        ValueError: If it is not lowercase letters, digits and underscores only.
    """
    if not _METHOD_PATTERN.fullmatch(name):
        raise ValueError(f"invalid method name '{name}': use lowercase letters, digits, underscore")
    return name


def is_oracle_method(name: str) -> bool:
    """True if the method name contains 'oracle' anywhere (false positives are harmless)."""
    return ORACLE_TOKEN in validate_method(name)


def validate_split(split: str) -> str:
    """Check that a split name is one of davis_val, sav_val, sav_test.

    Args:
        split: Split name.

    Returns:
        The same name.

    Raises:
        ValueError: If the split is not known.
    """
    if split not in SPLITS:
        raise ValueError(f"invalid split '{split}': expected one of {sorted(SPLITS)}")
    return split


def validate_name(name: str, what: str) -> str:
    """Check that a video name or frame stem cannot escape its folder.

    Args:
        name: Video name or frame stem.
        what: Word used in the error message, for example "video".

    Returns:
        The same name.

    Raises:
        ValueError: If it is empty, "." or "..", or contains a path separator.
    """
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"invalid {what} '{name}': must be non-empty without path separators")
    return name


def run_dir(output_root: Path, method: str, split: str) -> Path:
    """Folder that holds everything one method wrote for one split.

    Args:
        output_root: Root folder for run outputs.
        method: Method name.
        split: Split name.

    Returns:
        output_root / method / split.
    """
    return output_root / validate_method(method) / validate_split(split)


def masks_dir(output_root: Path, method: str, split: str, video: str) -> Path:
    """Folder of the per-frame mask PNGs of one video.

    Args:
        output_root: Root folder for run outputs.
        method: Method name.
        split: Split name.
        video: Video name.

    Returns:
        The masks/<video> folder under the run folder.
    """
    return run_dir(output_root, method, split) / "masks" / validate_name(video, "video")


def mask_path(output_root: Path, method: str, split: str, video: str, frame_stem: str) -> Path:
    """Path of one frame's mask PNG.

    Args:
        output_root: Root folder for run outputs.
        method: Method name.
        split: Split name.
        video: Video name.
        frame_stem: Frame file name without extension, for example "00012".

    Returns:
        The path <masks folder>/<frame_stem>.png.
    """
    stem = validate_name(frame_stem, "frame stem")
    # Layout: <masks folder>/<frame_stem>.png; both names were validated above.
    return masks_dir(output_root, method, split, video) / f"{stem}.png"


def signals_path(output_root: Path, method: str, split: str, video: str) -> Path:
    """Path of the per-frame signal CSV of one video.

    Args:
        output_root: Root folder for run outputs.
        method: Method name.
        split: Split name.
        video: Video name.

    Returns:
        The path signals/<video>.csv under the run folder.
    """
    return run_dir(output_root, method, split) / "signals" / f"{validate_name(video, 'video')}.csv"


def events_path(output_root: Path, method: str, split: str, video: str) -> Path:
    """Path of the recovery event CSV of one video.

    Args:
        output_root: Root folder for run outputs.
        method: Method name.
        split: Split name.
        video: Video name.

    Returns:
        The path events/<video>.csv under the run folder.
    """
    return run_dir(output_root, method, split) / "events" / f"{validate_name(video, 'video')}.csv"
