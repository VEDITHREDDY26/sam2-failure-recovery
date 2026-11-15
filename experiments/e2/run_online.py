"""Run the real system (SAM 2 + rule detector + aligned CSRT recovery) on the prompted videos.

    python -m experiments.e2.run_online --config configs/recovery/rules_csrt_tiny.yaml \
        --prompts prompts/davis_val.json [--videos a b] [--device cuda] [--offload-video]

Real system only: oracle configs are refused. A verified config (fix_verified_*) additionally
needs the frozen configs/verify/verify.yaml and runs on davis_val or sav_test only.
--log-candidates (accept-all runs on sav_val only) writes the per-candidate feature log used to
tune the verification rule; it does not change the run. The rule thresholds must have been frozen
(configs/detector/rules.yaml); without that file nothing runs. Outputs go under
<data_root>/<output_root>/<method>/<split>/; complete videos are skipped on a rerun.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections.abc import Sequence
from pathlib import Path

from src.common.config import DATA_ROOT_ENV, ConfigError
from src.detection.online import OnlineRuleDetector
from src.detection.rules import DEFAULT_RULES_PATH
from src.recovery.box_check import BoxCheck, load_box_check_config
from src.recovery.config import RecoveryRunConfig, VerifiedRunConfig, load_recovery_config
from src.recovery.online_runner import check_online_method, run_batch
from src.recovery.settings import CONFIG_PATH, load_recovery_settings
from src.recovery.sources import CSRTBoxSource
from src.recovery.verify import VerifyThresholds, load_verify_thresholds
from src.sam2_runner.predictor import MODELS, build_predictor
from src.sam2_runner.prompts import load_batch_prompts
from src.sam2_runner.smoke import load_data_config


def model_of(cfg: RecoveryRunConfig) -> str:
    for model in MODELS:
        if cfg.method.endswith(f"_{model}"):
            return model
    raise SystemExit(f"cannot tell the SAM 2 model from method '{cfg.method}' (needs _tiny/_large)")


def choose_verifier(
    run_cfg: RecoveryRunConfig, model: str, split: str, log_candidates: bool
) -> VerifyThresholds | None:
    """Frozen thresholds for a verified run; refuses the tuning split and a missing freeze."""
    if not isinstance(run_cfg, VerifiedRunConfig):
        if log_candidates and split != "sav_val":
            raise SystemExit(f"--log-candidates is for the sav_val tuning run, not '{split}'")
        return None
    if log_candidates:
        raise SystemExit("--log-candidates is for the accept-all run, not a verified config")
    if split == "sav_val":
        raise SystemExit(
            "verified runs are for davis_val and sav_test; sav_val is the tuning split"
        )
    try:
        return load_verify_thresholds(model, run_cfg.verify_path())
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offload-video", action="store_true")
    parser.add_argument("--log-candidates", action="store_true")
    args = parser.parse_args(argv)
    run_cfg = load_recovery_config(args.config)
    if run_cfg.detector != "rules" or run_cfg.box_source != "csrt":
        raise SystemExit(
            f"{args.config}: run_online is the real system only (rules + csrt); oracle runs "
            "have their own entry script"
        )
    check_online_method(run_cfg.method)
    if not DEFAULT_RULES_PATH.is_file():
        raise SystemExit(f"detector not frozen yet: {DEFAULT_RULES_PATH} does not exist")
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    model = model_of(run_cfg)
    settings = load_recovery_settings(CONFIG_PATH)
    if run_cfg.tracker_mode:
        settings = dataclasses.replace(settings, tracker_mode=run_cfg.tracker_mode)
    data = load_data_config()
    prompts = load_batch_prompts(args.prompts)
    verifier = choose_verifier(run_cfg, model, prompts.split, args.log_candidates)
    run_batch(
        prompts,
        data,
        run_cfg.method,
        lambda: build_predictor(model, data.data_root, device=args.device),
        lambda video: OnlineRuleDetector.from_config(
            model, DEFAULT_RULES_PATH, run_cfg.enabled_rules
        ),
        lambda video: CSRTBoxSource(settings.tracker_mode),
        BoxCheck(load_box_check_config()),
        settings,
        args.videos,
        args.offload_video,
        verifier,
        args.log_candidates,
    )


if __name__ == "__main__":
    main()
