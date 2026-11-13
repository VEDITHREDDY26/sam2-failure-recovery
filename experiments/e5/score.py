"""Score a finished learned-detector run of E5 through the E2 scoring code.

    python -m experiments.e5.score --config configs/recovery/learned_csrt_tiny.yaml \
        --splits davis_val sav_test [--subset-file f] [--results-dir results] [--videos a b]

Same outputs and completeness checks as experiments.e2.score (it refuses incomplete runs and
sam2_ / baseline names); learned configs only.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from experiments.e2.run_online import model_of
from experiments.e2.score import RESULTS_DIR, score_split
from experiments.e5.run_learned import E5_SPLITS
from src.common.config import DATA_ROOT_ENV
from src.eval.check_complete import SUBSET_FILE
from src.recovery.config import load_recovery_config
from src.recovery.online_runner import check_online_method


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=E5_SPLITS, default=list(E5_SPLITS))
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--videos", nargs="+", default=None)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    run_cfg = load_recovery_config(args.config)
    if run_cfg.detector != "learned" or run_cfg.box_source != "csrt":
        raise SystemExit(f"{args.config}: E5 scores learned + csrt runs only")
    method = check_online_method(run_cfg.method)
    model = model_of(run_cfg)
    refused = 0
    for split in args.splits:
        ok = score_split(method, model, split, args.results_dir, args.subset_file, args.videos)
        refused += not ok
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
