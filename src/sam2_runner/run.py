"""Batch run: SAM 2 on many videos, skipping videos whose output is already complete.

    python -m src.sam2_runner.run --model tiny --prompts prompts.json [--videos a b]
        [--method sam2_tiny] [--device cuda]

The split comes from the prompts file (see prompts.py). Outputs go under
<data_root>/<output_root>/<method>/<split>/.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from src.common.config import DATA_ROOT_ENV, DataConfig
from src.common.layout import validate_method
from src.sam2_runner.predictor import VideoPredictor, build_predictor
from src.sam2_runner.prompts import BatchPrompts, load_batch_prompts
from src.sam2_runner.runner import frame_stems, is_complete, run_video
from src.sam2_runner.smoke import load_data_config

DAVIS_SPLIT = "davis_val"
SAV_FRAMES_DIR = "JPEGImages_24fps"
PredictorFactory = Callable[[], VideoPredictor]


def frames_dir_for(cfg: DataConfig, split: str, video: str) -> Path:
    if split == DAVIS_SPLIT:
        return cfg.data_root / cfg.davis.root / "JPEGImages" / cfg.davis.resolution / video
    folder = cfg.sav.val_split if split == "sav_val" else cfg.sav.test_split
    return cfg.data_root / cfg.sav.root / folder / SAV_FRAMES_DIR / video


def run_batch(
    prompts: BatchPrompts,
    cfg: DataConfig,
    method: str,
    make_predictor: PredictorFactory,
    videos: Sequence[str] | None = None,
) -> tuple[int, int]:
    """Run every selected video that is not complete; return (run, skipped)."""
    names = list(videos) if videos else list(prompts.videos)
    unknown = [v for v in names if v not in prompts.videos]
    if unknown:
        raise ValueError(f"no prompts for videos {unknown}")
    output_root = cfg.data_root / cfg.output_root
    predictor: VideoPredictor | None = None
    ran = skipped = 0
    for video in names:
        entry = prompts.videos[video]
        frames_dir = frames_dir_for(cfg, prompts.split, video)
        stems = frame_stems(frames_dir)
        if is_complete(output_root, method, prompts.split, video, stems, sorted(entry.boxes)):
            print(f"{video}: skipped (complete)")
            skipped += 1
            continue
        if predictor is None:
            predictor = make_predictor()
        result = run_video(predictor, frames_dir, entry, output_root, method, prompts.split, video)
        table = result.table
        print(
            f"{video}: done, {table['frame_index'].nunique()} frames, "
            f"objects {sorted(table['object_id'].unique().tolist())}"
        )
        print(f"  {result.source_summary}")
        ran += 1
    print(f"{ran} run, {skipped} skipped")
    return ran, skipped


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", choices=["tiny", "large"], required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    method = validate_method(args.method or f"sam2_{args.model}")
    cfg = load_data_config()
    prompts = load_batch_prompts(args.prompts)
    run_batch(
        prompts,
        cfg,
        method,
        lambda: build_predictor(args.model, cfg.data_root, device=args.device),
        args.videos,
    )


if __name__ == "__main__":
    main()
