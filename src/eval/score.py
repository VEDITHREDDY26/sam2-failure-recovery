"""Score stored predictions with the official scorers and write the per-video CSV.

    python -m src.eval.score --dataset davis --split davis_val --method sam2_tiny \
        --out results/davis_sam2_tiny.csv [--videos bear blackswan]
    python -m src.eval.score --dataset davis --split davis_val --pred-dir <label-map folder> \
        --out results/check.csv

Predictions are read from <output_root>/<method>/<split>/masks unless --pred-dir is given.
The CSV has the columns video,object,J,F,J&F, one row per (video, object), values in [0,1].
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

from src.common.config import DATA_ROOT_ENV, ConfigError, DataConfig, load_config
from src.common.layout import run_dir, validate_method
from src.eval.davis_scores import score_davis
from src.eval.evaluators import REPO_ROOT
from src.eval.rows import ScoreRow, write_scores_csv
from src.eval.sav_scores import score_sav

DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"
SAV_GT_DIR = "Annotations_6fps"
DATASET_SPLITS = {"davis": ("davis_val",), "sav": ("sav_val", "sav_test")}


def load_data_config() -> DataConfig:
    """Load the data settings (data root, dataset folders, output root) from configs/data.yaml.

    Raises:
        ConfigError: If the file is not a data config.
    """
    cfg = load_config(DATA_CONFIG)
    if not isinstance(cfg, DataConfig):
        raise ConfigError(f"{DATA_CONFIG}: not a data config")
    return cfg


def davis_root(cfg: DataConfig) -> Path:
    """DAVIS folder under the data root."""
    return cfg.data_root / cfg.davis.root


def sav_gt_root(cfg: DataConfig, split: str) -> Path:
    """SA-V ground-truth folder (Annotations_6fps) of a split.

    Args:
        cfg: Data config.
        split: "sav_val"; any other value is looked up in the SA-V test folder.
    """
    folder = cfg.sav.val_split if split == "sav_val" else cfg.sav.test_split
    return cfg.data_root / cfg.sav.root / folder / SAV_GT_DIR


def default_videos(cfg: DataConfig, dataset: str, split: str) -> list[str]:
    """Names of the videos to score when none are given.

    Args:
        cfg: Data config.
        dataset: "davis" (the DAVIS image-set list) or "sav" (every video folder of the split).
        split: Split name.
    """
    # DAVIS lists its videos in an ImageSets file; SA-V has one ground-truth folder per video.
    if dataset == "davis":
        listing = davis_root(cfg) / "ImageSets" / str(cfg.davis.year) / f"{cfg.davis.split}.txt"
        return [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    return sorted(p.name for p in sav_gt_root(cfg, split).iterdir() if p.is_dir())


def score(
    cfg: DataConfig, dataset: str, split: str, pred_root: Path, videos: Sequence[str] | None
) -> list[ScoreRow]:
    """Score predictions with the official scorer of a dataset.

    Args:
        cfg: Data config.
        dataset: "davis" or "sav".
        split: Split name; must belong to the dataset.
        pred_root: Folder with one subfolder of predicted label-map PNGs per video.
        videos: Video names, or None for all videos of the split.

    Returns:
        One ScoreRow per (video, object), J and F as fractions in [0, 1].

    Raises:
        ValueError: If the split does not belong to the dataset.
    """
    if split not in DATASET_SPLITS[dataset]:
        raise ValueError(f"split {split} does not belong to dataset {dataset}")
    names = list(videos) if videos else default_videos(cfg, dataset, split)
    if dataset == "davis":
        return score_davis(davis_root(cfg), pred_root, names, cfg.davis.resolution, cfg.davis.split)
    return score_sav(sav_gt_root(cfg, split), pred_root, names)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: score predictions and write the per-video CSV.

    Reads label maps from <output_root>/<method>/<split>/masks, or from --pred-dir, scores them
    and writes the rows to --out. Prints the object count and the mean J, F and J&F.

    Args:
        argv: Command-line arguments; None means sys.argv.

    Raises:
        SystemExit: Status 2 from argparse on bad arguments, when the data-root environment
            variable is not set, or when no videos were scored.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPLITS), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--pred-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    args = parser.parse_args(argv)
    # Predictions come from exactly one place: a named run or an explicit folder.
    if (args.method is None) == (args.pred_dir is None):
        parser.error("give exactly one of --method and --pred-dir")
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    cfg = load_data_config()
    if args.pred_dir is not None:
        pred_root = args.pred_dir
    else:
        pred_root = (
            run_dir(cfg.data_root / cfg.output_root, validate_method(args.method), args.split)
            / "masks"
        )
    # Scoring runs the completeness check first, then the official scorer.
    rows = score(cfg, args.dataset, args.split, pred_root, args.videos)
    if not rows:
        parser.error("nothing was scored: no videos selected")
    # Report the mean over all objects, as in the results tables.
    write_scores_csv(args.out, rows)
    sys.stdout.write(
        f"{len(rows)} objects: J {fmean([r.j for r in rows]):.4f}  "
        f"F {fmean([r.f for r in rows]):.4f}  J&F {fmean([r.jf for r in rows]):.4f} -> {args.out}\n"
    )


if __name__ == "__main__":
    main()
