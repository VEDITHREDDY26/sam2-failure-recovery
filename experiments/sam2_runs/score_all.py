"""Score the finished SAM 2 runs and collect their timing into the repo's results folder.

    python -m experiments.sam2_runs.score_all [--models tiny large]
        [--splits davis_val sav_val sav_test] [--results-dir results]

Writes results/sam2_<model>_<split>.csv (video,object,J,F,J&F, official scorers) and
results/sam2_<model>_timing_<split>.csv. A run that fails the completeness check is refused
and nothing is written for it; the exit status is 1 if any run was refused.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

from src.common.config import DATA_ROOT_ENV
from src.common.layout import run_dir, validate_method
from src.eval.check_complete import MODELS, SPLIT_NAMES, SUBSET_FILE, check_run, split_videos
from src.eval.rows import ScoringError, write_scores_csv
from src.eval.score import load_data_config, score
from src.sam2_runner.predictor import REPO_ROOT
from src.sam2_runner.timing import collect_timing

RESULTS_DIR = REPO_ROOT / "results"


def score_path(results_dir: Path, model: str, split: str) -> Path:
    return results_dir / f"sam2_{model}_{split}.csv"


def timing_csv_path(results_dir: Path, model: str, split: str) -> Path:
    return results_dir / f"sam2_{model}_timing_{split}.csv"


def score_run(model: str, split: str, results_dir: Path, subset_file: Path, prefix: str) -> bool:
    """Score one run; False (after saying why) if it is refused."""
    cfg = load_data_config()
    method = validate_method(f"{prefix}_{model}")
    name = f"{method}/{split}"
    check = check_run(cfg, method, split, subset_file)
    if check.missing:
        sys.stdout.write("\n".join(check.lines()) + f"\n{name}: refused, run is not complete\n")
        return False
    videos = split_videos(cfg, split, subset_file)
    output_root = cfg.data_root / cfg.output_root
    dataset = "davis" if split == "davis_val" else "sav"
    pred_root = run_dir(output_root, method, split) / "masks"
    try:
        rows = score(cfg, dataset, split, pred_root, videos)
        timing = collect_timing(output_root, method, split, videos)
    except (ScoringError, FileNotFoundError) as error:
        sys.stdout.write(f"{name}: refused, {error}\n")
        return False
    write_scores_csv(score_path(results_dir, model, split), rows)
    out = timing_csv_path(results_dir, model, split)
    timing.to_csv(out, index=False)
    sys.stdout.write(
        f"{name}: {len(videos)} videos, {len(rows)} objects, "
        f"J&F {fmean([r.jf for r in rows]):.4f}, "
        f"{timing['frames'].sum() / timing['compute_s'].sum():.1f} fps -> "
        f"{score_path(results_dir, model, split)}, {out}\n"
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    parser.add_argument("--method-prefix", default="sam2")
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    refused = 0
    for model in args.models:
        for split in args.splits:
            ok = score_run(model, split, args.results_dir, args.subset_file, args.method_prefix)
            refused += not ok
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
