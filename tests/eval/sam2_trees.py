"""Fake DAVIS + SA-V data tree and fake SAM 2 runs on it, for the run-chain tests."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.sam2_runs.make_prompts import build_prompts, default_output
from src.sam2_runner.prompts import parse_batch_prompts
from src.sam2_runner.run import run_batch
from src.sam2_runner.smoke import load_data_config
from tests.data.helpers import build_davis_root, build_sav_root
from tests.fixtures.synthetic_video import SyntheticVideo, make_synthetic_video
from tests.sam2_runner.fakes import FakePredictor

SAV_VAL = ("vid_a", "vid_b")
SAV_TEST = ("vid_a", "vid_b", "vid_c")
SUBSET = ("vid_a", "vid_b")
DAVIS = ("zebra", "bear")
SPLITS = ("davis_val", "sav_val", "sav_test")


def build_world(root: Path) -> Path:
    """Fake DAVIS val, SA-V val and SA-V test trees under root; returns the test subset file."""
    build_davis_root(root)
    build_sav_root(root, "sav_val", SAV_VAL)
    build_sav_root(root, "sav_test", SAV_TEST)
    subset = root / "subset.txt"
    subset.write_text("# fake subset\n" + "".join(f"{v}\n" for v in SUBSET))
    return subset


def run_split(root: Path, model: str, split: str, subset: Path, source: SyntheticVideo) -> None:
    """Prompt and run the split with the fake predictor (needs SAM2FR_DATA_ROOT = root)."""
    cfg = load_data_config()
    raw = build_prompts(cfg, split, subset)
    target = default_output(cfg, split)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(raw))
    prompts = parse_batch_prompts(raw)
    run_batch(prompts, cfg, f"sam2_{model}", lambda: FakePredictor(source))


def run_all(
    root: Path,
    subset: Path,
    models: tuple[str, ...] = ("tiny", "large"),
    splits: tuple[str, ...] = SPLITS,
) -> None:
    source = make_synthetic_video(root / "fake_source", "source")
    for model in models:
        for split in splits:
            run_split(root, model, split, subset, source)
