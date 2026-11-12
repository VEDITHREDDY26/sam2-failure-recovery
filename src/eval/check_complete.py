"""Check that the SAM 2 runs cover every expected video before anything is scored.

    python -m src.eval.check_complete [--models tiny large]
        [--splits davis_val sav_val sav_test] [--method-prefix sam2]

For each model x split the expected videos are the DAVIS val list, the SA-V val list or the
seeded SA-V test subset. A video counts as complete only if the runner finished it (signals,
timing and one PNG per frame, for every promptable object) and the PNGs pass the scorer's own
completeness check against the ground truth. Exit status 1 if any video is missing.
Expected objects come from <output_root>/prompts/<split>.json when it exists, else from the
loaders (objects that are never visible are not expected).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.common.config import DATA_ROOT_ENV, DataConfig
from src.common.layout import run_dir, validate_method, validate_name, validate_split
from src.data.davis import list_davis_videos, load_davis_video
from src.data.make_subset import load_subset
from src.data.sav import list_sav_videos, load_sav_video
from src.data.video import NeverVisibleError, VideoSample
from src.eval.completeness import find_problems
from src.eval.score import davis_root, load_data_config, sav_gt_root
from src.sam2_runner.predictor import REPO_ROOT
from src.sam2_runner.prompts import Prompts, load_batch_prompts
from src.sam2_runner.run import frames_dir_for
from src.sam2_runner.runner import frame_stems, is_complete

SPLIT_NAMES = ("davis_val", "sav_val", "sav_test")
MODELS = ("tiny", "large")
SUBSET_FILE = REPO_ROOT / "configs" / "splits" / "sav_test_subset60.txt"
MAX_LISTED = 20


@dataclass
class RunCheck:
    method: str
    split: str
    total: int
    missing: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> int:
        return self.total - len(self.missing)

    def lines(self) -> list[str]:
        head = f"{self.method}/{self.split}: {self.complete}/{self.total} complete, "
        out = [head + f"{len(self.missing)} missing"]
        shown = list(self.missing.items())[:MAX_LISTED]
        out.extend(f"  missing {video}: {reason}" for video, reason in shown)
        if len(self.missing) > len(shown):
            out.append(f"  ... and {len(self.missing) - len(shown)} more")
        return out


def split_videos(cfg: DataConfig, split: str, subset_file: Path = SUBSET_FILE) -> list[str]:
    """The videos a run of this split covers."""
    validate_split(split)
    if split == "davis_val":
        return list(list_davis_videos(cfg))
    if split == "sav_val":
        return list(list_sav_videos(cfg, "val"))
    wanted = list(load_subset(subset_file))
    available = set(list_sav_videos(cfg, "test"))
    missing = [v for v in wanted if v not in available]
    if missing:
        raise ValueError(
            f"{len(missing)} subset video(s) not in the SA-V test list: {missing[:10]}"
        )
    return wanted


def load_video(cfg: DataConfig, split: str, name: str) -> VideoSample:
    if split == "davis_val":
        return load_davis_video(cfg, name)
    return load_sav_video(cfg, "val" if split == "sav_val" else "test", name)


def prompts_file(cfg: DataConfig, split: str) -> Path:
    return cfg.data_root / cfg.output_root / "prompts" / f"{split}.json"


def promptable_objects(video: VideoSample) -> list[int]:
    found = []
    for obj_id in video.object_ids:
        try:
            video.first_box(obj_id)
        except NeverVisibleError:
            continue
        found.append(obj_id)
    return found


def _gt_root_and_kind(cfg: DataConfig, split: str) -> tuple[Path, bool]:
    if split == "davis_val":
        return davis_root(cfg) / "Annotations" / cfg.davis.resolution, False
    return sav_gt_root(cfg, split), True


def check_run(
    cfg: DataConfig, method: str, split: str, subset_file: Path = SUBSET_FILE
) -> RunCheck:
    videos = split_videos(cfg, split, subset_file)
    path = prompts_file(cfg, split)
    prompted = load_batch_prompts(path).videos if path.is_file() else None
    output_root = cfg.data_root / cfg.output_root
    gt_root, sav = _gt_root_and_kind(cfg, split)
    pred_root = run_dir(output_root, method, split) / "masks"
    result = RunCheck(method, split, len(videos))
    for video in videos:
        reason = _video_problem(cfg, output_root, method, split, video, prompted, gt_root, sav)
        if reason:
            result.missing[video] = reason
        else:
            problems = find_problems(gt_root, pred_root, [video], sav)
            if problems:
                result.missing[video] = f"fails the scorer check: {problems[0]}"
    return result


def _video_problem(
    cfg: DataConfig,
    output_root: Path,
    method: str,
    split: str,
    video: str,
    prompted: dict[str, Prompts] | None,
    gt_root: Path,
    sav: bool,
) -> str | None:
    if prompted is not None:
        if video not in prompted:
            return "no prompts for this video"
        obj_ids = sorted(prompted[video].boxes)
    else:
        obj_ids = promptable_objects(load_video(cfg, split, video))
    try:
        stems = frame_stems(frames_dir_for(cfg, split, video))
    except FileNotFoundError as error:
        return str(error)
    if not is_complete(output_root, method, split, validate_name(video, "video"), stems, obj_ids):
        return "not finished by the runner (signals, timing or PNGs missing)"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    parser.add_argument("--method-prefix", default="sam2")
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    cfg = load_data_config()
    failed = False
    for model in args.models:
        method = validate_method(f"{args.method_prefix}_{model}")
        for split in args.splits:
            check = check_run(cfg, method, split, args.subset_file)
            sys.stdout.write("\n".join(check.lines()) + "\n")
            failed = failed or bool(check.missing)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
