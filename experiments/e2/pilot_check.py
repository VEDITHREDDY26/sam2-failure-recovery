"""Check an online run against the cached SAM 2 alone masks and write the pilot event tables.

    python -m experiments.e2.pilot_check --config configs/recovery/rules_csrt_tiny.yaml \
        --prompts prompts/davis_val.json [--videos a b] [--out results/events_pilot.csv]

For each video, every frame before the first "reprompted" event must equal the cached
sam2_<model> label map (all frames if there was no re-prompt): recovery re-prompts the live
predictor and must not change earlier frames. Exit code 1 if any video differs or is missing.
Writes <out> (all event rows of the checked videos) and <out stem>_check.csv (one verdict row
per video).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e2.run_online import model_of
from src.common.config import DATA_ROOT_ENV
from src.common.layout import events_path, mask_path
from src.common.masks import read_label_map
from src.recovery.config import load_recovery_config
from src.recovery.events import EVENT_COLUMNS
from src.recovery.online_runner import frame_paths
from src.sam2_runner.prompts import load_batch_prompts
from src.sam2_runner.run import frames_dir_for
from src.sam2_runner.smoke import load_data_config

CHECK_COLUMNS = ("video", "frames", "first_reprompt", "compared_frames", "verdict", "first_diff")
DEFAULT_OUT = Path("results/events_pilot.csv")


@dataclass(frozen=True)
class VideoCheck:
    video: str
    frames: int
    first_reprompt: int | None
    compared: int
    verdict: str
    first_diff: int | None

    @property
    def ok(self) -> bool:
        return self.verdict == "identical"

    def as_row(self) -> tuple[object, ...]:
        first = "" if self.first_reprompt is None else self.first_reprompt
        diff = "" if self.first_diff is None else self.first_diff
        return (self.video, self.frames, first, self.compared, self.verdict, diff)


def read_events(output_root: Path, method: str, split: str, video: str) -> pd.DataFrame:
    path = events_path(output_root, method, split, video)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run the online runner first")
    return pd.read_csv(path)


def first_reprompt(events: pd.DataFrame) -> int | None:
    frames = events.loc[events["action"] == "reprompted", "frame_index"]
    return int(frames.min()) if len(frames) else None


def check_video(
    output_root: Path, method: str, reference: str, split: str, video: str, stems: Sequence[str]
) -> VideoCheck:
    first = first_reprompt(read_events(output_root, method, split, video))
    limit = len(stems) if first is None else first
    for index in range(limit):
        ours = mask_path(output_root, method, split, video, stems[index])
        cached = mask_path(output_root, reference, split, video, stems[index])
        if not ours.is_file() or not cached.is_file():
            return VideoCheck(video, len(stems), first, index, "missing_mask", index)
        if not np.array_equal(read_label_map(ours), read_label_map(cached)):
            return VideoCheck(video, len(stems), first, index, "differs", index)
    return VideoCheck(video, len(stems), first, limit, "identical", None)


def write_tables(out: Path, events: pd.DataFrame, checks: Sequence[VideoCheck]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out, index=False, columns=list(EVENT_COLUMNS))
    summary = out.with_name(out.stem + "_check.csv")
    rows = [c.as_row() for c in checks]
    pd.DataFrame(rows, columns=list(CHECK_COLUMNS)).to_csv(summary, index=False)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    run_cfg = load_recovery_config(args.config)
    reference = f"sam2_{model_of(run_cfg)}"
    data = load_data_config()
    prompts = load_batch_prompts(args.prompts)
    names = list(args.videos) if args.videos else list(prompts.videos)
    output_root = data.data_root / data.output_root
    checks: list[VideoCheck] = []
    tables: list[pd.DataFrame] = []
    for video in names:
        stems = [p.stem for p in frame_paths(frames_dir_for(data, prompts.split, video))]
        checks.append(
            check_video(output_root, run_cfg.method, reference, prompts.split, video, stems)
        )
        tables.append(read_events(output_root, run_cfg.method, prompts.split, video))
    events = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=EVENT_COLUMNS)
    summary = write_tables(args.out, events, checks)
    for check in checks:
        where = "" if check.ok else f" at frame {check.first_diff}"
        print(f"{check.video}: {check.verdict}{where} ({check.compared} frames compared)")
    print(f"{len(events)} events -> {args.out}; verdicts -> {summary}")
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
