"""Shared setup for the online runner tests: a small DAVIS-style tree for the recovery fake."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common.boxes import Box
from src.common.config import DataConfig
from src.sam2_runner.prompts import BatchPrompts, parse_batch_prompts
from src.sam2_runner.smoke import load_data_config
from tests.recovery.fakes import NUM_FRAMES, H, W

VIDEO = "vid"
SPLIT = "davis_val"
START_BOX = Box(10, 20, 22, 32)
METHOD = "fix_test"


def make_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DataConfig, BatchPrompts]:
    data_root = tmp_path / "drive"
    frames = data_root / "DAVIS" / "JPEGImages" / "480p" / VIDEO
    frames.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(NUM_FRAMES):
        pixels = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(frames / f"{index:05d}.jpg")
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(data_root))
    raw = {
        "split": SPLIT,
        "videos": {
            VIDEO: {
                "frame_index": 0,
                "boxes": {"1": [START_BOX.x0, START_BOX.y0, START_BOX.x1, START_BOX.y1]},
            }
        },
    }
    return load_data_config(), parse_batch_prompts(json.loads(json.dumps(raw)))


def tree_hash(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
