"""Batch CSRT + GrabCut baseline, skipping videos whose output is already complete.

    python -m src.baseline.run --split davis_val|sav_test [--videos a b] \
        [--method baseline_csrt_grabcut]

davis_val runs every DAVIS 2017 val video; sav_test runs exactly the 60 ids in
configs/splits/sav_test_subset60.txt. Outputs go under <data_root>/<output_root>/<method>/<split>/.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from src.baseline.config import METHOD, BaselineConfig, load_baseline_config
from src.baseline.runner import TrackerFactory, clear_video, is_complete, run_video
from src.common.config import DATA_ROOT_ENV, ConfigError, DataConfig, load_config
from src.common.layout import validate_method
from src.data.davis import list_davis_videos, load_davis_video
from src.data.make_subset import load_subset
from src.data.sav import load_sav_video
from src.data.video import VideoSample

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"
SUBSET_FILE = REPO_ROOT / "configs" / "splits" / "sav_test_subset60.txt"
SPLITS = ("davis_val", "sav_test")


def load_data_config() -> DataConfig:
    cfg = load_config(DATA_CONFIG)
    if not isinstance(cfg, DataConfig):
        raise ConfigError(f"{DATA_CONFIG}: not a data config")
    return cfg


def split_videos(cfg: DataConfig, split: str, subset_file: Path = SUBSET_FILE) -> tuple[str, ...]:
    if split == "davis_val":
        return list_davis_videos(cfg)
    if split == "sav_test":
        return load_subset(subset_file)
    raise ValueError(f"split must be one of {SPLITS}, got '{split}'")


def loader_for(cfg: DataConfig, split: str) -> Callable[[str], VideoSample]:
    if split == "davis_val":
        return lambda name: load_davis_video(cfg, name)
    return lambda name: load_sav_video(cfg, "test", name)


def run_batch(
    cfg: DataConfig,
    base: BaselineConfig,
    split: str,
    method: str = METHOD,
    videos: Sequence[str] | None = None,
    make_tracker: TrackerFactory | None = None,
    subset_file: Path = SUBSET_FILE,
) -> tuple[int, int]:
    """Run every selected video that is not complete; return (run, skipped)."""
    allowed = split_videos(cfg, split, subset_file)
    names = list(videos) if videos else list(allowed)
    unknown = [v for v in names if v not in allowed]
    if unknown:
        raise ValueError(f"videos not in split {split}: {unknown}")
    output_root = cfg.data_root / cfg.output_root
    load = loader_for(cfg, split)
    ran = skipped = 0
    for name in names:
        video = load(name)
        if is_complete(output_root, method, split, name, video.frame_stems, list(video.object_ids)):
            print(f"{name}: skipped (complete)")
            skipped += 1
            continue
        clear_video(output_root, method, split, name)
        result = run_video(video, base, output_root, method, split, make_tracker)
        lost = int((result.status["tracker_ok"] == 0).sum())
        print(
            f"{name}: done, {video.num_frames} frames, {len(video.object_ids)} objects, "
            f"{lost} lost object-frames, {float(result.timing['fps']):.1f} fps"  # type: ignore[arg-type]
        )
        ran += 1
    print(f"{ran} run, {skipped} skipped")
    return ran, skipped


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--method", default=METHOD)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    run_batch(
        load_data_config(),
        load_baseline_config(),
        args.split,
        validate_method(args.method),
        args.videos,
    )


if __name__ == "__main__":
    main()
