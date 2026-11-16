"""Score a finished oracle arm of E7: same outputs as experiments.e2.score, oracle names only.

    python -m experiments.e7.score --config configs/oracle/oracle_flags_csrt_tiny.yaml \
        --splits davis_val sav_test [--subset-file f] [--results-dir results]

Writes results/<method>_<split>.csv (and _timing_, _frames, events_ companions) through the E2
scoring code, which refuses incomplete runs and sam2_ / baseline names. The method must contain
"oracle" so oracle numbers can never be mistaken for the real system.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from experiments.e2.run_online import model_of
from experiments.e2.score import RESULTS_DIR, score_split
from experiments.e7.run_oracle import E7_SPLITS, check_config
from src.common.config import DATA_ROOT_ENV
from src.eval.check_complete import SUBSET_FILE
from src.recovery.config import load_recovery_config
from src.recovery.online_runner import check_online_method


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=E7_SPLITS, default=list(E7_SPLITS))
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--videos", nargs="+", default=None)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    run_cfg = load_recovery_config(args.config)
    check_config(run_cfg, args.config)
    method = check_online_method(run_cfg.method)
    model = model_of(run_cfg)
    refused = 0
    for split in args.splits:
        ok = score_split(method, model, split, args.results_dir, args.subset_file, args.videos)
        refused += not ok
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
