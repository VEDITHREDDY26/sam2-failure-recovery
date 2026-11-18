"""Label the logged re-prompt candidates of the accept-all run on SA-V val (needs ground truth).

    python -m experiments.verify.label_candidates --config configs/recovery/rules_csrt_tiny.yaml \
        [--videos-file experiments/verify/sav_val_large_subset.txt] [--results-dir results]

For each candidate (the accept-all method run with --log-candidates on sav_val), the first
annotated frame g at or after the candidate's frame t is found; then
    gain = IoU(method mask at g, GT) - IoU(SAM 2 alone mask at g, GT).
With no annotated frame left the candidate is unscored (blank). Writes
results/verify_candidates_<model>_sav_val.csv: the candidate log columns plus gt_frame_index,
iou_method, iou_alone and gain. Only sav_val is read; test splits and verified methods are
refused, so no test-set ground truth reaches the tuning.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.e2.run_online import model_of
from experiments.verify.draw_subset import read_names
from src.common.config import DATA_ROOT_ENV, DataConfig
from src.common.layout import mask_path
from src.common.masks import read_label_map
from src.data.video import VideoSample
from src.detection.labels import mask_iou
from src.eval.check_complete import load_video, split_videos
from src.eval.score import load_data_config
from src.recovery.config import VerifiedRunConfig, load_recovery_config
from src.recovery.online_runner import check_online_method
from src.recovery.verify import CANDIDATE_COLUMNS, candidate_path
from src.sam2_runner.predictor import REPO_ROOT

SPLIT = "sav_val"
LABEL_COLUMNS = (*CANDIDATE_COLUMNS, "gt_frame_index", "iou_method", "iou_alone", "gain")
RESULTS_DIR = REPO_ROOT / "results"


def labels_name(model: str) -> str:
    """File name of the labelled candidates of a model on SA-V val."""
    return f"verify_candidates_{model}_{SPLIT}.csv"


def _iou_at(
    video: VideoSample, output_root: Path, method: str, obj_id: int, frame: int
) -> float | None:
    """IoU of one saved method mask with the ground truth at a frame.

    Args:
        video: Video with its ground truth.
        output_root: Folder holding the saved masks.
        method: Method whose mask is read.
        obj_id: Object id in the label map.
        frame: Frame index.

    Returns:
        The IoU, or None when the object has no ground truth mask at that frame.
    """
    gt = video.gt_mask(obj_id, frame)
    if gt is None:
        return None
    # Read the saved mask of that frame and compare the object's pixels with the ground truth.
    stem = video.frame_stems[frame]
    label = read_label_map(mask_path(output_root, method, SPLIT, video.name, stem))
    return mask_iou(label == obj_id, gt)


def label_candidate(
    video: VideoSample, output_root: Path, method: str, alone: str, obj_id: int, frame: int
) -> tuple[object, object, object, object]:
    """(gt frame, iou method, iou alone, gain); blanks if no annotated frame at or after t."""
    # Gain is measured at the first annotated frame at or after the candidate.
    later = [f for f in video.annotated_frames(obj_id) if f >= frame]
    if not later:
        return "", "", "", ""
    gt_frame = later[0]
    with_fix = _iou_at(video, output_root, method, obj_id, gt_frame)
    without = _iou_at(video, output_root, alone, obj_id, gt_frame)
    if with_fix is None or without is None:
        return "", "", "", ""
    return gt_frame, with_fix, without, with_fix - without


def label_table(cfg: DataConfig, method: str, model: str, videos: Sequence[str]) -> pd.DataFrame:
    """Label all candidates of the given videos with their ground-truth gain.

    Args:
        cfg: Data configuration (data root and output folder).
        method: The accept-all method whose candidates were logged.
        model: SAM 2 size; its SAM 2 alone masks are the reference.
        videos: Videos to read candidate logs from.

    Returns:
        The candidate rows with gt_frame_index, iou_method, iou_alone and gain added; blank
        values where a candidate cannot be scored. Empty with the full header when there are
        no candidates.

    Raises:
        FileNotFoundError: If a video has no candidate log.
        ValueError: If a candidate log has the wrong columns.
    """
    output_root = cfg.data_root / cfg.output_root
    frames: list[pd.DataFrame] = []
    # One candidate log per video; an unreadable or missing log stops the run.
    for name in videos:
        path = candidate_path(output_root, method, SPLIT, name)
        if not path.is_file():
            raise FileNotFoundError(f"no candidate log for {name}: {path}")
        table = pd.read_csv(path, dtype={"video": str})
        if list(table.columns) != list(CANDIDATE_COLUMNS):
            raise ValueError(f"unusable candidate log for {name}")
        if table.empty:
            continue
        video = load_video(cfg, SPLIT, name)
        alone = f"sam2_{model}"
        # Score each candidate; the extra columns are appended to the log columns.
        extra = [
            label_candidate(video, output_root, method, alone, int(obj), int(frame))
            for obj, frame in zip(table["object_id"], table["frame_index"], strict=True)
        ]
        labelled = pd.concat(
            [table, pd.DataFrame(extra, columns=list(LABEL_COLUMNS[len(CANDIDATE_COLUMNS) :]))],
            axis=1,
        )
        frames.append(labelled)
    # No candidates at all: return an empty table with the full header.
    if not frames:
        return pd.DataFrame(columns=list(LABEL_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: label the accept-all candidates on SA-V val.

    Reads the candidate logs, saved masks and ground truth of the accept-all method named in
    --config, for the videos in --videos-file (default: all SA-V val videos); the data root
    comes from the environment. Writes verify_candidates_<model>_sav_val.csv in --results-dir
    and prints the candidate and scored counts.

    Args:
        argv: Command line arguments; None means sys.argv.

    Returns:
        Exit code 0 on success.

    Raises:
        SystemExit: If the split is not sav_val or the config is a verified one. A missing data
            root variable ends with the usage error code 2.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--videos-file", type=Path, default=None)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    # Only SA-V val may be labelled, so test-set ground truth never reaches the tuning.
    if args.split != SPLIT:
        raise SystemExit(f"split '{args.split}' refused: candidates are labelled on {SPLIT} only")
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    # Load the config of the accept-all method and find the videos to label.
    run_cfg = load_recovery_config(args.config)
    if isinstance(run_cfg, VerifiedRunConfig):
        raise SystemExit("label the accept-all method (rules_csrt_*), not a verified config")
    method = check_online_method(run_cfg.method)
    model = model_of(run_cfg)
    cfg = load_data_config()
    videos = read_names(args.videos_file) if args.videos_file else split_videos(cfg, SPLIT)
    # Label the candidates and write the table.
    table = label_table(cfg, method, model, videos)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results_dir / labels_name(model), index=False)
    scored = int(pd.to_numeric(table["gain"], errors="coerce").notna().sum())
    sys.stdout.write(f"{method}/{SPLIT}: {len(table)} candidates, {scored} scored\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
