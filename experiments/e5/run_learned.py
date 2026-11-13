"""Run the real system with the learned detector (SAM 2 + logistic regression + aligned CSRT).

    python -m experiments.e5.run_learned --config configs/recovery/learned_csrt_tiny.yaml \
        --prompts prompts/davis_val.json [--videos a b] [--device cuda] [--offload-video]

Refused unless configs/learned/logreg_<model>.json exists and its sha256 equals the hash pinned
in configs/learned/logreg_<model>.sha256 (written by experiments.e5.train; record that hash in
docs/decisions.md before any test-set run). davis_val and sav_test only: SA-V val is the data the
detector was fitted on. Everything else is the E2 run (same recovery settings, same CSRT source).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections.abc import Sequence
from pathlib import Path

from experiments.e2.run_online import model_of
from src.common.config import DATA_ROOT_ENV
from src.detection.learned import DEFAULT_DIR, ModelFileError, OnlineLearnedDetector, load_pinned
from src.recovery.box_check import BoxCheck, load_box_check_config
from src.recovery.config import load_recovery_config
from src.recovery.online_runner import check_online_method, run_batch
from src.recovery.settings import CONFIG_PATH, load_recovery_settings
from src.recovery.sources import CSRTBoxSource
from src.sam2_runner.predictor import build_predictor
from src.sam2_runner.prompts import load_batch_prompts
from src.sam2_runner.smoke import load_data_config

E5_SPLITS = ("davis_val", "sav_test")
LEARNED_DIR = DEFAULT_DIR


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offload-video", action="store_true")
    args = parser.parse_args(argv)
    run_cfg = load_recovery_config(args.config)
    if run_cfg.detector != "learned" or run_cfg.box_source != "csrt":
        raise SystemExit(f"{args.config}: run_learned is for learned + csrt configs only")
    check_online_method(run_cfg.method)
    model = model_of(run_cfg)
    try:
        detector_model = load_pinned(model, LEARNED_DIR)
    except ModelFileError as error:
        raise SystemExit(f"learned detector refused: {error}") from error
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    prompts = load_batch_prompts(args.prompts)
    if prompts.split not in E5_SPLITS:
        raise SystemExit(f"split '{prompts.split}' refused: E5 runs only {E5_SPLITS}")
    settings = load_recovery_settings(CONFIG_PATH)
    if run_cfg.tracker_mode:
        settings = dataclasses.replace(settings, tracker_mode=run_cfg.tracker_mode)
    data = load_data_config()
    run_batch(
        prompts,
        data,
        run_cfg.method,
        lambda: build_predictor(model, data.data_root, device=args.device),
        lambda video: OnlineLearnedDetector(detector_model),
        lambda video: CSRTBoxSource(settings.tracker_mode),
        BoxCheck(load_box_check_config()),
        settings,
        args.videos,
        args.offload_video,
    )


if __name__ == "__main__":
    main()
