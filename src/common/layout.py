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
    if not _METHOD_PATTERN.fullmatch(name):
        raise ValueError(f"invalid method name '{name}': use lowercase letters, digits, underscore")
    return name


def is_oracle_method(name: str) -> bool:
    """True if the method name contains 'oracle' anywhere (false positives are harmless)."""
    return ORACLE_TOKEN in validate_method(name)


def validate_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"invalid split '{split}': expected one of {sorted(SPLITS)}")
    return split


def validate_name(name: str, what: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"invalid {what} '{name}': must be non-empty without path separators")
    return name


def run_dir(output_root: Path, method: str, split: str) -> Path:
    return output_root / validate_method(method) / validate_split(split)


def masks_dir(output_root: Path, method: str, split: str, video: str) -> Path:
    return run_dir(output_root, method, split) / "masks" / validate_name(video, "video")


def mask_path(output_root: Path, method: str, split: str, video: str, frame_stem: str) -> Path:
    stem = validate_name(frame_stem, "frame stem")
    return masks_dir(output_root, method, split, video) / f"{stem}.png"


def signals_path(output_root: Path, method: str, split: str, video: str) -> Path:
    return run_dir(output_root, method, split) / "signals" / f"{validate_name(video, 'video')}.csv"


def events_path(output_root: Path, method: str, split: str, video: str) -> Path:
    return run_dir(output_root, method, split) / "events" / f"{validate_name(video, 'video')}.csv"
