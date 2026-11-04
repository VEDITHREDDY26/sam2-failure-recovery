"""Smoke run: prompt SAM 2 with first-frame boxes, propagate, store masks and signals.

    python -m src.sam2_runner.smoke --model tiny --video bear --prompts prompts.json

prompts.json: {"video": str, "frame_index": 0, "boxes": {"<obj_id>": [x0, y0, x1, y1]}} with
boxes in the project Box convention (right and bottom edges exclusive).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from src.common.config import DATA_ROOT_ENV, ConfigError, DataConfig, load_config
from src.sam2_runner.predictor import REPO_ROOT, VideoPredictor, build_predictor
from src.sam2_runner.prompts import Prompts, load_prompts, parse_prompts
from src.sam2_runner.runner import RunResult, frame_stems, run_video, sam2_box

DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"
SPLIT = "davis_val"
__all__ = ["Prompts", "frame_stems", "load_prompts", "parse_prompts", "sam2_box"]


def run_smoke(
    predictor: VideoPredictor,
    frames_dir: Path,
    prompts: Prompts,
    output_root: Path,
    method: str,
    video: str | None = None,
) -> RunResult:
    """Run one DAVIS video (see runner.run_video)."""
    return run_video(
        predictor, frames_dir, prompts, output_root, method, SPLIT, video or prompts.video
    )


def load_data_config() -> DataConfig:
    cfg = load_config(DATA_CONFIG)
    if not isinstance(cfg, DataConfig):
        raise ConfigError(f"{DATA_CONFIG}: not a data config")
    return cfg


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", choices=["tiny", "large"], required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default=None)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    cfg = load_data_config()
    prompts = load_prompts(args.prompts)
    if prompts.video != args.video:
        parser.error(f"prompts are for video '{prompts.video}', not '{args.video}'")
    frames_dir = cfg.data_root / cfg.davis.root / "JPEGImages" / cfg.davis.resolution / args.video
    method = args.method or f"sam2_{args.model}_smoke"
    predictor = build_predictor(args.model, cfg.data_root, device=args.device)
    result = run_smoke(predictor, frames_dir, prompts, cfg.data_root / cfg.output_root, method)
    table = result.table
    print(
        f"{method}: {args.video} {table['frame_index'].nunique()} frames, "
        f"objects {sorted(table['object_id'].unique().tolist())}"
    )
    print(result.source_summary)


if __name__ == "__main__":
    main()
