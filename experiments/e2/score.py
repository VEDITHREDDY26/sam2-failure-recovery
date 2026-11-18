"""Score a finished real-system run (SAM 2 + rules + recovery) and collect its side tables.

    python -m experiments.e2.score --config configs/recovery/rules_csrt_tiny.yaml \
        --splits davis_val sav_test [--subset-file f] [--videos a b] [--results-dir results]

Per split, only if every expected video is complete (nothing is written otherwise, exit status 1):
  results/<method>_<split>.csv         video,object,J,F,J&F from the official scorers
  results/<method>_timing_<split>.csv  per-video timing with the restart columns
  results/<method>_<split>_frames.csv  per object: labelled frames, SAM 2 alone and fix failures,
                                       frames recovered (alone failed, fix did not) and broken
  results/events_<method>_<split>.csv  every recovery event of the run
Any real-system run is scored (rules detector; CSRT, ViT or motion box source; gated or not); oracle
runs are refused. Output names starting with sam2_ or baseline are refused, so the SAM 2 alone
results are never overwritten. Only davis_val and sav_test exist for E2.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

import pandas as pd

from experiments.e2.run_online import model_of
from src.common.config import DATA_ROOT_ENV, DataConfig
from src.common.layout import events_path, run_dir
from src.data.video import VideoSample
from src.detection.labels import video_labels
from src.eval.check_complete import (
    SUBSET_FILE,
    RunCheck,
    _gt_root_and_kind,
    load_video,
    prompts_file,
    split_videos,
)
from src.eval.completeness import find_problems
from src.eval.rows import ScoringError, write_scores_csv
from src.eval.score import load_data_config, score
from src.recovery.config import load_recovery_config
from src.recovery.events import EVENT_COLUMNS
from src.recovery.online_runner import (
    ONLINE_TIMING_COLUMNS,
    check_online_method,
    is_online_complete,
)
from src.sam2_runner.predictor import REPO_ROOT
from src.sam2_runner.prompts import load_batch_prompts
from src.sam2_runner.run import frames_dir_for
from src.sam2_runner.runner import frame_stems
from src.sam2_runner.timing import timing_path

E2_SPLITS = ("davis_val", "sav_test")
RESULTS_DIR = REPO_ROOT / "results"
FRAME_COLUMNS = (
    "video",
    "object_id",
    "labelled_frames",
    "sam2_failures",
    "fix_failures",
    "frames_recovered",
    "frames_broken",
)
LABEL_KEY = ["video", "object_id", "frame_index"]


def output_names(method: str, split: str) -> dict[str, str]:
    """Build the result file names of a run and refuse names reserved for other runs.

    Args:
        method: Method name of the run.
        split: Split name.

    Returns:
        File names for "scores", "timing", "frames" and "events".

    Raises:
        ValueError: If a name starts with "sam2_" or "baseline", which would overwrite the SAM 2
            alone or baseline results.
    """
    names = {
        "scores": f"{method}_{split}.csv",
        "timing": f"{method}_timing_{split}.csv",
        "frames": f"{method}_{split}_frames.csv",
        "events": f"events_{method}_{split}.csv",
    }
    for name in names.values():
        if name.startswith(("sam2_", "baseline")):
            raise ValueError(
                f"refusing to write '{name}': SAM 2 alone and baseline names are fixed"
            )
    return names


def check_method_run(cfg: DataConfig, method: str, split: str, videos: Sequence[str]) -> RunCheck:
    """Completeness of an online run: runner outputs for every object, then the scorer's check."""
    path = prompts_file(cfg, split)
    prompted = load_batch_prompts(path).videos if path.is_file() else None
    output_root = cfg.data_root / cfg.output_root
    gt_root, sav = _gt_root_and_kind(cfg, split)
    pred_root = run_dir(output_root, method, split) / "masks"
    result = RunCheck(method, split, len(videos))
    # Every video needs prompts, frames, a finished online run and a passing scorer check.
    for video in videos:
        if prompted is None or video not in prompted:
            result.missing[video] = "no prompts for this video"
            continue
        try:
            stems = frame_stems(frames_dir_for(cfg, split, video))
        except FileNotFoundError as error:
            result.missing[video] = str(error)
            continue
        obj_ids = sorted(prompted[video].boxes)
        if not is_online_complete(output_root, method, split, video, stems, obj_ids):
            result.missing[video] = "not finished by the online runner"
            continue
        problems = find_problems(gt_root, pred_root, [video], sav)
        if problems:
            result.missing[video] = f"fails the scorer check: {problems[0]}"
    return result


def frame_counts(alone: pd.DataFrame, fix: pd.DataFrame) -> pd.DataFrame:
    """Per (video, object): failures of both runs on the same labelled frames, recovered, broken."""
    # Align both runs on (video, object, frame); an unmatched frame is an error.
    merged = alone.merge(
        fix, on=LABEL_KEY, how="outer", suffixes=("_alone", "_fix"), indicator=True
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("SAM 2 alone and the fix are not labelled on the same frames")
    # Recovered: SAM 2 alone failed and the fix did not. Broken: the reverse.
    merged["recovered"] = merged["failure_alone"] & ~merged["failure_fix"]
    merged["broken"] = ~merged["failure_alone"] & merged["failure_fix"]
    # Sum the per-frame flags into one row per object.
    grouped = merged.groupby(["video", "object_id"], sort=True)
    counts = grouped.agg(
        labelled_frames=("failure_alone", "size"),
        sam2_failures=("failure_alone", "sum"),
        fix_failures=("failure_fix", "sum"),
        frames_recovered=("recovered", "sum"),
        frames_broken=("broken", "sum"),
    ).reset_index()
    return counts.astype({c: int for c in FRAME_COLUMNS[2:]})[list(FRAME_COLUMNS)]


def run_frame_counts(
    cfg: DataConfig, model: str, method: str, split: str, videos: Sequence[str]
) -> pd.DataFrame:
    """Count failure frames of SAM 2 alone and of the run on the same labelled frames.

    Args:
        cfg: Data config.
        model: SAM 2 model size, "tiny" or "large".
        method: Method name of the run.
        split: Split name.
        videos: Videos to count over.

    Returns:
        The per-object table from frame_counts.

    Raises:
        ValueError: If SAM 2 alone and the run are not labelled on the same frames.
    """
    output_root = cfg.data_root / cfg.output_root
    alone, fix = [], []
    for name in videos:
        sample: VideoSample = load_video(cfg, split, name)
        alone.append(video_labels(sample, output_root, f"sam2_{model}", split))
        fix.append(video_labels(sample, output_root, method, split))
    return frame_counts(pd.concat(alone, ignore_index=True), pd.concat(fix, ignore_index=True))


def collect_timing(
    output_root: Path, method: str, split: str, videos: Sequence[str]
) -> pd.DataFrame:
    """Read and stack the per-video timing files of a run.

    Args:
        output_root: Root folder that holds all run outputs.
        method: Method name of the run.
        split: Split name.
        videos: Videos to read.

    Returns:
        One row per video.

    Raises:
        ScoringError: If a timing file has the wrong columns or not exactly one row.
    """
    tables = []
    for video in videos:
        table = pd.read_csv(timing_path(output_root, method, split, video), dtype={"video": str})
        if list(table.columns) != list(ONLINE_TIMING_COLUMNS) or len(table) != 1:
            raise ScoringError(f"unusable timing file for {video}")
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def collect_events(
    output_root: Path, method: str, split: str, videos: Sequence[str]
) -> pd.DataFrame:
    """Read and stack the per-video event files of a run.

    Args:
        output_root: Root folder that holds all run outputs.
        method: Method name of the run.
        split: Split name.
        videos: Videos to read.

    Returns:
        All events (an empty table with the event columns if there are no videos).

    Raises:
        ScoringError: If an events file has the wrong columns.
    """
    tables = []
    for video in videos:
        table = pd.read_csv(events_path(output_root, method, split, video), dtype={"video": str})
        if list(table.columns) != list(EVENT_COLUMNS):
            raise ScoringError(f"unusable events file for {video}")
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=EVENT_COLUMNS)


def score_split(
    run_method: str,
    model: str,
    split: str,
    results_dir: Path,
    subset_file: Path,
    only: Sequence[str] | None = None,
) -> bool:
    """Score one split of the run; False (after saying why) if it is refused."""
    names = output_names(run_method, split)
    cfg = load_data_config()
    # Optionally narrow to the videos named on the command line.
    videos = split_videos(cfg, split, subset_file)
    if only:
        unknown = [v for v in only if v not in videos]
        if unknown:
            raise SystemExit(f"{split}: videos not in the split: {unknown}")
        videos = list(only)
    label = f"{run_method}/{split}"
    # Refuse to score a split whose run is incomplete, and say why.
    check = check_method_run(cfg, run_method, split, videos)
    if check.missing:
        sys.stdout.write("\n".join(check.lines()) + f"\n{label}: refused, run is not complete\n")
        return False
    output_root = cfg.data_root / cfg.output_root
    pred_root = run_dir(output_root, run_method, split) / "masks"
    # DAVIS val uses the DAVIS scorer; the SA-V test subset uses the SA-V scorer.
    dataset = "davis" if split == "davis_val" else "sav"
    # Gather scores and side tables first so nothing is written if one of them fails.
    try:
        rows = score(cfg, dataset, split, pred_root, videos)
        timing = collect_timing(output_root, run_method, split, videos)
        events = collect_events(output_root, run_method, split, videos)
        frames = run_frame_counts(cfg, model, run_method, split, videos)
    except (ScoringError, FileNotFoundError, ValueError) as error:
        sys.stdout.write(f"{label}: refused, {error}\n")
        return False
    # All four tables are written only after every input was read successfully.
    results_dir.mkdir(parents=True, exist_ok=True)
    write_scores_csv(results_dir / names["scores"], rows)
    timing.to_csv(results_dir / names["timing"], index=False)
    frames.to_csv(results_dir / names["frames"], index=False)
    events.to_csv(results_dir / names["events"], index=False)
    # Print a one-line summary of the scored split.
    reprompts = int((events["action"] == "reprompted").sum())
    gated = int((events["action"] == "gated").sum())
    sys.stdout.write(
        f"{label}: {len(videos)} videos, {len(rows)} objects, "
        f"J&F {fmean([r.jf for r in rows]):.4f}, {reprompts} re-prompts, {gated} gated, "
        f"{int(frames['frames_recovered'].sum())} frames recovered, "
        f"{int(frames['frames_broken'].sum())} broken -> {results_dir}\n"
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Score a finished real-system run on the requested splits (command line).

    Reads the run config, the run's outputs and the ground truth under the data root (environment
    variable). Writes the score, timing, frames and events CSVs for each complete split into
    --results-dir and prints one summary line per split.

    Args:
        argv: Command-line arguments; None means sys.argv.

    Returns:
        0 if every split was scored, 1 if any split was refused.

    Raises:
        SystemExit: If the config is an oracle or non-rules run.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=E2_SPLITS, default=list(E2_SPLITS))
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--videos", nargs="+", default=None)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    # Refuse oracle configs: scoring is for the real system only.
    run_cfg = load_recovery_config(args.config)
    if run_cfg.detector != "rules" or run_cfg.box_source == "oracle":
        raise SystemExit(
            f"{args.config}: scoring is for the real system only (rules detector, no oracle box)"
        )
    method = check_online_method(run_cfg.method)
    model = model_of(run_cfg)
    refused = 0
    # Score each split; a refused split makes the exit code 1 but the others still run.
    for split in args.splits:
        ok = score_split(method, model, split, args.results_dir, args.subset_file, args.videos)
        refused += not ok
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
