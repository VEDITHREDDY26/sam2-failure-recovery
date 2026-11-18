"""Check a finished gated (or any online) run against SAM 2 alone: untouched videos must be equal.

    python -m experiments.e13.check_gated --config configs/recovery/rules_gated_motion_tiny.yaml \
        --splits davis_val sav_test [--subset-file f] [--results-dir results]

For every video of the split, a video with no "reprompted" event must equal the cached SAM 2 alone
label map on every frame (a gate that blocks every box leaves the run untouched). A video with
re-prompts must still equal SAM 2 alone on every frame before the first one. Writes
results/e13_gated_check_<method>_<split>.csv (video, reprompts, gated, compared_frames, identical)
and exits 1 if any video differs or a mask is missing.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.e2.pilot_check import check_video, read_events
from experiments.e2.run_online import model_of
from experiments.e2.score import RESULTS_DIR
from src.common.config import DATA_ROOT_ENV
from src.eval.check_complete import SUBSET_FILE, split_videos
from src.eval.score import load_data_config
from src.recovery.config import load_recovery_config
from src.recovery.online_runner import check_online_method, frame_paths
from src.sam2_runner.run import frames_dir_for

SPLITS = ("davis_val", "sav_test")
CHECK_COLUMNS = ("video", "reprompts", "gated", "compared_frames", "identical")


def check_name(method: str, split: str) -> str:
    return f"e13_gated_check_{check_online_method(method)}_{split}.csv"


def check_split(
    output_root: Path,
    method: str,
    reference: str,
    split: str,
    videos: Sequence[str],
    frames_of: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[tuple[object, ...]] = []
    for video in videos:
        events = read_events(output_root, method, split, video)
        verdict = check_video(output_root, method, reference, split, video, frames_of[video])
        rows.append(
            (
                video,
                int((events["action"] == "reprompted").sum()),
                int((events["action"] == "gated").sum()),
                verdict.compared,
                verdict.ok,
            )
        )
    return pd.DataFrame(rows, columns=list(CHECK_COLUMNS))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    run_cfg = load_recovery_config(args.config)
    if run_cfg.detector != "rules" or run_cfg.box_source == "oracle":
        raise SystemExit(f"{args.config}: the check is for the real system only")
    reference = f"sam2_{model_of(run_cfg)}"
    data = load_data_config()
    output_root = data.data_root / data.output_root
    failed = False
    for split in args.splits:
        videos = split_videos(data, split, args.subset_file)
        stems = {v: [p.stem for p in frame_paths(frames_dir_for(data, split, v))] for v in videos}
        try:
            table = check_split(output_root, run_cfg.method, reference, split, videos, stems)
        except FileNotFoundError as error:
            sys.stdout.write(f"{run_cfg.method}/{split}: refused, {error}\n")
            failed = True
            continue
        args.results_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.results_dir / check_name(run_cfg.method, split), index=False)
        bad = table[~table["identical"]]
        untouched = int((table["reprompts"] == 0).sum())
        sys.stdout.write(
            f"{run_cfg.method}/{split}: {len(table)} videos, {untouched} without a re-prompt, "
            f"{len(bad)} not identical\n"
        )
        failed |= not bad.empty
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
