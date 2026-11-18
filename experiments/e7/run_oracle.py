"""Run an oracle arm of E7 (oracle flags and/or oracle boxes) on the prompted videos.

    python -m experiments.e7.run_oracle --config configs/oracle/oracle_flags_csrt_tiny.yaml \
        --prompts prompts/davis_val.json [--videos a b] [--device cuda] [--offload-video]

Oracle configs only (the real system runs through experiments.e2.run_online); the method name
carries "oracle", so outputs land in oracle-named folders and files. The oracle pieces read ground
truth through the dataset loaders. sav_val is refused. The frozen rule thresholds are needed only
by the arm that still uses the rule detector.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections.abc import Sequence
from pathlib import Path

from experiments.e2.run_online import model_of
from src.common.config import DATA_ROOT_ENV
from src.data.video import VideoSample
from src.detection.online import OnlineRuleDetector
from src.detection.rules import DEFAULT_RULES_PATH
from src.eval.check_complete import load_video
from src.oracle.online import OracleBoxSource, OracleDetector
from src.recovery.box_check import BoxCheck, load_box_check_config
from src.recovery.config import RecoveryRunConfig, load_recovery_config
from src.recovery.online_runner import check_online_method, run_batch
from src.recovery.settings import CONFIG_PATH, load_recovery_settings
from src.recovery.sources import BoxSource, CSRTBoxSource, Detector
from src.sam2_runner.predictor import build_predictor
from src.sam2_runner.prompts import load_batch_prompts
from src.sam2_runner.smoke import load_data_config

E7_SPLITS = ("davis_val", "sav_test")


def check_config(cfg: RecoveryRunConfig, path: Path) -> None:
    """Refuse a config that uses no oracle piece.

    Args:
        cfg: Recovery run config.
        path: Config path, used in the message.

    Raises:
        SystemExit: If neither the detector nor the box source is "oracle".
        ValueError: If the method name is invalid or names a SAM 2 alone or baseline folder.
    """
    if "oracle" not in (cfg.detector, cfg.box_source):
        raise SystemExit(
            f"{path}: run_oracle is for oracle arms only; use experiments.e2.run_online"
        )
    check_online_method(cfg.method)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point that runs one oracle arm of E7.

    Reads an oracle recovery config, a prompts file and the dataset through the loaders (the
    oracle pieces read ground truth there). Writes the run's masks and events through the online
    runner under oracle-named folders. Exits with a message (SystemExit) if the config has no
    oracle piece, the frozen rules are needed but missing, or the split is not a test set;
    argparse exits with code 2 if the data-root variable is not set.

    Args:
        argv: Argument list; None means sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offload-video", action="store_true")
    args = parser.parse_args(argv)
    # Refuse non-oracle configs and a missing rules file before loading anything.
    run_cfg = load_recovery_config(args.config)
    check_config(run_cfg, args.config)
    if run_cfg.detector == "rules" and not DEFAULT_RULES_PATH.is_file():
        raise SystemExit(f"detector not frozen yet: {DEFAULT_RULES_PATH} does not exist")
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    # Only the test sets are run; SA-V val is refused.
    prompts = load_batch_prompts(args.prompts)
    if prompts.split not in E7_SPLITS:
        raise SystemExit(f"split '{prompts.split}' refused: E7 runs only {E7_SPLITS}")
    model = model_of(run_cfg)
    # Recovery settings are the same as in the E2 run; the config may set the tracker.
    settings = load_recovery_settings(CONFIG_PATH)
    if run_cfg.tracker_mode:
        settings = dataclasses.replace(settings, tracker_mode=run_cfg.tracker_mode)
    data = load_data_config()
    # Each video is loaded once and shared by the oracle detector and box source.
    samples: dict[str, VideoSample] = {}

    def sample(video: str) -> VideoSample:
        """Load the dataset sample of a video on first use and cache it."""
        if video not in samples:
            samples[video] = load_video(data, prompts.split, video)
        return samples[video]

    def make_detector(video: str) -> Detector:
        """Oracle flags if the config asks for them, otherwise the rule detector."""
        if run_cfg.detector == "oracle":
            return OracleDetector(sample(video))
        return OnlineRuleDetector.from_config(model, DEFAULT_RULES_PATH, run_cfg.enabled_rules)

    def make_boxes(video: str) -> BoxSource:
        """Oracle box if the config asks for it, otherwise the CSRT box source."""
        if run_cfg.box_source == "oracle":
            return OracleBoxSource(sample(video))
        return CSRTBoxSource(settings.tracker_mode)

    # Run the online loop with the chosen detector and box source.
    run_batch(
        prompts,
        data,
        run_cfg.method,
        lambda: build_predictor(model, data.data_root, device=args.device),
        make_detector,
        make_boxes,
        BoxCheck(load_box_check_config()),
        settings,
        args.videos,
        args.offload_video,
    )


if __name__ == "__main__":
    main()
