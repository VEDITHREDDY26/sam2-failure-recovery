"""Fake SAM 2 alone run plus a fake online run next to it, for the E2 score tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.layout import events_path, mask_path
from src.common.masks import read_label_map, write_label_map
from src.recovery.events import EVENT_COLUMNS
from src.recovery.online_runner import ONLINE_TIMING_COLUMNS
from src.sam2_runner.timing import timing_path

METHOD = "fix_rules_csrt_tiny"
SPLIT = "davis_val"


def make_online_run(output_root: Path, method: str = METHOD, split: str = SPLIT) -> None:
    """Copy the SAM 2 alone run, add events and the restart timing columns."""
    source = output_root / "sam2_tiny" / split
    target = output_root / method / split
    shutil.copytree(source, target)
    for video in sorted(p.stem for p in (source / "timing").glob("*.csv")):
        path = timing_path(output_root, method, split, video)
        table = pd.read_csv(path)
        table["restarts"] = 0
        table["restart_s"] = 0.0
        table[list(ONLINE_TIMING_COLUMNS)].to_csv(path, index=False)
        write_events(output_root, method, split, video, [])


def write_events(
    output_root: Path, method: str, split: str, video: str, rows: list[tuple[int, int, str]]
) -> None:
    """rows are (object_id, frame_index, action)."""
    data = [(video, o, f, "score", a, "", "", "", "", "", 0) for o, f, a in rows]
    path = events_path(output_root, method, split, video)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=list(EVENT_COLUMNS)).to_csv(path, index=False)


def blank_object(
    output_root: Path, method: str, split: str, video: str, stem: str, obj_id: int
) -> None:
    path = mask_path(output_root, method, split, video, stem)
    label = np.array(read_label_map(path))
    label[label == obj_id] = 0
    write_label_map(path, label)
