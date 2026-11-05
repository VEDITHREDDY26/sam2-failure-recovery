"""API spike: add a box at frame t > 0 and resume propagation, three ways.

    python -m experiments.spikes.reprompt --model tiny [--videos a b c] [--device cuda]

Sequences (each compared with its own no-re-prompt baseline run):
  correct_tracked       baseline state, box on the already tracked frame t, propagate from t
  fresh_cond            new state, propagate frames 0..t-1 only, box on the untracked frame t,
                        propagate from t
  correct_tracked_cond  as correct_tracked, with a predictor built with
                        add_all_frames_to_correct_as_cond=true (official builder override)

The test box at frame t is the tight box of the ground-truth mask of the smallest object id.
Ground truth is used only here, to get a test box for this API check; it is not a recovery
box. The script reports the comparison and does not choose between the sequences. Results go
to <data_root>/<output_root>/spikes/reprompt_<model>.csv, never into results/.

SAM 2 methods are already decorated with torch.inference_mode, and the project smoke run uses
no autocast, so neither is added here.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.common.boxes import Box, box_from_mask
from src.common.config import DATA_ROOT_ENV, DataConfig
from src.data.davis import list_davis_videos, load_davis_video
from src.data.video import VideoSample
from src.sam2_runner.predictor import MODELS, VideoPredictor, load_models_config, to_numpy
from src.sam2_runner.smoke import load_data_config, sam2_box

Array = npt.NDArray[Any]
SEQUENCES = ("correct_tracked", "fresh_cond", "correct_tracked_cond")
COLUMNS = (
    "video",
    "model",
    "sequence",
    "t",
    "reprompt_object",
    "object_id",
    "frames_yielded",
    "before_identical",
    "before_max_abs_logit_diff",
    "after_frames_differ",
    "after_frames_total",
    "pixels_changed_at_t",
    "mask_returned_by_add_differs_at_t",
)
DEFAULT_VIDEO_COUNT = 3
ADD_ALL_OVERRIDE = "++model.add_all_frames_to_correct_as_cond=true"


@dataclasses.dataclass
class Recording:
    """Mask logits yielded by propagate_in_video, keyed by (frame, object), plus the add result."""

    logits: dict[tuple[int, int], Array] = dataclasses.field(default_factory=dict)
    yielded: list[int] = dataclasses.field(default_factory=list)
    added: dict[int, Array] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Reprompt:
    frame: int
    obj_id: int
    box: Box
    note: str


def choose_reprompt(video: VideoSample) -> Reprompt:
    """Frame num_frames // 2 (or the next frame where the smallest object is visible)."""
    obj_id = video.object_ids[0]
    start = video.num_frames // 2
    for frame in range(start, video.num_frames):
        mask = video.gt_mask(obj_id, frame)
        box = None if mask is None else box_from_mask(mask)
        if box is not None:
            note = "" if frame == start else f"object empty at frame {start}"
            return Reprompt(frame, obj_id, box, note)
    raise ValueError(f"{video.name}: object {obj_id} not visible from frame {start} on")


def record(results: Any, recording: Recording, frames: Sequence[int] | None = None) -> Recording:
    for frame_idx, obj_ids, logits in results:
        if frames is not None and frame_idx not in frames:
            raise RuntimeError(f"unexpected frame {frame_idx}")
        recording.yielded.append(int(frame_idx))
        for row, obj_id in enumerate(obj_ids):
            recording.logits[(int(frame_idx), int(obj_id))] = _logits(logits[row, 0])
    return recording


def _logits(value: Any) -> Array:
    return np.asarray(to_numpy(value), dtype=np.float32)


def prompt_first_frame(predictor: VideoPredictor, state: Any, video: VideoSample) -> None:
    for obj_id in video.object_ids:
        prompt = video.first_box(obj_id)
        predictor.add_new_points_or_box(state, prompt.frame_index, obj_id, box=sam2_box(prompt.box))


def add_reprompt(
    predictor: VideoPredictor, state: Any, reprompt: Reprompt, recording: Recording
) -> None:
    box = sam2_box(reprompt.box)
    _, obj_ids, logits = predictor.add_new_points_or_box(
        state, reprompt.frame, reprompt.obj_id, box=box
    )
    for row, obj_id in enumerate(obj_ids):
        recording.added[int(obj_id)] = _logits(logits[row, 0])


def run_baseline(predictor: VideoPredictor, video: VideoSample) -> tuple[Any, Recording]:
    state = predictor.init_state(str(video.frames_dir))
    prompt_first_frame(predictor, state, video)
    return state, record(predictor.propagate_in_video(state), Recording())


def run_correct_tracked(predictor: VideoPredictor, state: Any, reprompt: Reprompt) -> Recording:
    """Box on a tracked frame of the baseline state, then propagate from that frame."""
    rec = Recording()
    add_reprompt(predictor, state, reprompt, rec)
    results = predictor.propagate_in_video(state, start_frame_idx=reprompt.frame)
    return record(results, rec)


def run_fresh_cond(predictor: VideoPredictor, video: VideoSample, reprompt: Reprompt) -> Recording:
    """Track frames 0..t-1, then add the box on the untracked frame t and propagate from t."""
    t = reprompt.frame
    state = predictor.init_state(str(video.frames_dir))
    prompt_first_frame(predictor, state, video)
    rec = Recording()
    first = predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=t - 1)
    record(first, rec, frames=range(t))
    if rec.yielded != list(range(t)):
        raise RuntimeError(f"expected frames 0..{t - 1}, got {rec.yielded}")
    add_reprompt(predictor, state, reprompt, rec)
    return record(predictor.propagate_in_video(state, start_frame_idx=t), rec)


def _binary(logits: Array) -> Array:
    return logits > 0


def _frames_text(frames: Sequence[int]) -> str:
    return f"{min(frames)}..{max(frames)}" if frames else "none"


def summarize(
    video: str,
    model: str,
    sequence: str,
    reprompt: Reprompt,
    baseline: Recording,
    run: Recording,
    object_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """One row per object. Frames before t that were not re-yielded are the stored baseline."""
    t = reprompt.frame
    rows = []
    for obj_id in object_ids:
        before = [f for f in sorted(set(run.yielded)) if f < t and (f, obj_id) in run.logits]
        after = [f for f in sorted(set(run.yielded)) if f >= t and (f, obj_id) in run.logits]
        diffs = [
            float(np.abs(run.logits[(f, obj_id)] - baseline.logits[(f, obj_id)]).max())
            for f in before
        ]
        identical = all(
            np.array_equal(_binary(run.logits[(f, obj_id)]), _binary(baseline.logits[(f, obj_id)]))
            for f in before
        )
        differ = sum(
            not np.array_equal(
                _binary(run.logits[(f, obj_id)]), _binary(baseline.logits[(f, obj_id)])
            )
            for f in after
        )
        base_t = _binary(baseline.logits[(t, obj_id)])
        changed = int((_binary(run.logits[(t, obj_id)]) != base_t).sum()) if t in after else 0
        added = run.added.get(obj_id)
        added_differs = added is not None and not np.array_equal(_binary(added), base_t)
        rows.append(
            {
                "video": video,
                "model": model,
                "sequence": sequence,
                "t": t,
                "reprompt_object": reprompt.obj_id,
                "object_id": obj_id,
                "frames_yielded": _frames_text(sorted(set(run.yielded))),
                "before_identical": identical,
                "before_max_abs_logit_diff": max(diffs) if diffs else float("nan"),
                "after_frames_differ": differ,
                "after_frames_total": len(after),
                "pixels_changed_at_t": changed,
                "mask_returned_by_add_differs_at_t": added_differs,
            }
        )
    return rows


def acceptance(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], bool]:
    """Per (video, sequence): frames < t identical for all objects and frames >= t differ."""
    verdict: dict[tuple[str, str], bool] = {}
    for row in rows:
        key = (str(row["video"]), str(row["sequence"]))
        ok = bool(row["before_identical"])
        if row["object_id"] == row["reprompt_object"]:
            ok = ok and int(row["after_frames_differ"]) > 0
        verdict[key] = verdict.get(key, True) and ok
    return verdict


def run_video(
    build: Callable[[bool], VideoPredictor], video: VideoSample, model: str
) -> list[dict[str, Any]]:
    """All three sequences on one video. `build(add_all)` makes a fresh predictor."""
    reprompt = choose_reprompt(video)
    ids = video.object_ids
    default = build(False)
    state, base = run_baseline(default, video)
    rows: list[dict[str, Any]] = []
    sequences = {"fresh_cond": run_fresh_cond(default, video, reprompt)}
    sequences["correct_tracked"] = run_correct_tracked(default, state, reprompt)
    for name in ("correct_tracked", "fresh_cond"):
        rows += summarize(video.name, model, name, reprompt, base, sequences[name], ids)
    cond = build(True)
    cond_state, cond_base = run_baseline(cond, video)
    cond_run = run_correct_tracked(cond, cond_state, reprompt)
    rows += summarize(video.name, model, "correct_tracked_cond", reprompt, cond_base, cond_run, ids)
    return rows


def build_spike_predictor(
    model: str, data_root: Path, device: str, add_all_frames: bool
) -> VideoPredictor:
    """Official builder; the flag is a Hydra override, SAM 2 code is unchanged (Colab only)."""
    cfg = load_models_config()
    spec = cfg.tiny if model == "tiny" else cfg.large
    ckpt = cfg.checkpoint_path(data_root, spec)
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} not found; run scripts/get_checkpoints.sh")
    build = importlib.import_module("sam2.build_sam").build_sam2_video_predictor
    extra = [ADD_ALL_OVERRIDE] if add_all_frames else []
    predictor: VideoPredictor = build(
        spec.model_cfg, str(ckpt), device=device, hydra_overrides_extra=extra
    )
    return predictor


def output_path(cfg: DataConfig, model: str) -> Path:
    return cfg.data_root / cfg.output_root / "spikes" / f"reprompt_{model}.csv"


def write_table(rows: Sequence[Mapping[str, Any]], path: Path) -> pd.DataFrame:
    table = pd.DataFrame(list(rows), columns=list(COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def verdict_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    verdict = acceptance(rows)
    lines = []
    for sequence in SEQUENCES:
        results = {v: ok for (v, s), ok in verdict.items() if s == sequence}
        if not results:
            continue
        passed = sorted(v for v, ok in results.items() if ok)
        lines.append(f"{sequence}: {len(passed)}/{len(results)} videos pass {passed}")
    return lines


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Re-prompt API spike")
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    cfg = load_data_config()
    names = args.videos or list(list_davis_videos(cfg)[:DEFAULT_VIDEO_COUNT])

    def build(add_all: bool) -> VideoPredictor:
        return build_spike_predictor(args.model, cfg.data_root, args.device, add_all)

    rows: list[dict[str, Any]] = []
    for name in names:
        rows += run_video(build, load_davis_video(cfg, name), args.model)
    path = output_path(cfg, args.model)
    table = write_table(rows, path)
    print(table.to_string(index=False))
    print(f"wrote {path}")
    for line in verdict_lines(rows):
        print(line)


if __name__ == "__main__":
    main()
