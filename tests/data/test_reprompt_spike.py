from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.spikes.reprompt import (
    COLUMNS,
    SEQUENCES,
    Recording,
    acceptance,
    choose_reprompt,
    output_path,
    run_baseline,
    run_fresh_cond,
    run_video,
    summarize,
    verdict_lines,
    write_table,
)
from src.data.davis import load_davis_video
from src.data.video import VideoSample
from tests.data.helpers import build_davis_root, make_config

HEIGHT, WIDTH = 16, 96


class FakePredictor:
    """Models the SAM 2 rules that matter here.

    A frame given a box is conditioning if untracked or if add_all is set; otherwise the box
    only changes the mask returned by the add call. propagate_in_video reuses conditioning
    frames and re-runs every other frame from the previous frame (memory), without the box.
    """

    def __init__(self, add_all: bool) -> None:
        self.add_all = add_all
        self.calls: list[tuple[int | None, int | None]] = []

    def init_state(self, video_path: str, **_: Any) -> dict[str, Any]:
        n = len(list(Path(video_path).glob("*.jpg")))
        return {"n": n, "ids": [], "cond": {}, "memory": {}, "tracked": {}}

    @staticmethod
    def _band(center: float) -> np.ndarray:
        out = np.full((HEIGHT, WIDTH), -1.0, dtype=np.float32)
        lo = int(np.clip(center - 3, 0, WIDTH - 6))
        out[4:12, lo : lo + 6] = 1.0
        return out

    def _stack(self, state: dict[str, Any], frame: int, centers: dict[int, float]) -> np.ndarray:
        return np.stack([self._band(centers[o])[None] for o in state["ids"]])

    def add_new_points_or_box(
        self, state: dict[str, Any], frame_idx: int, obj_id: int, box: Any = None, **_: Any
    ) -> tuple[int, list[int], np.ndarray]:
        if obj_id not in state["ids"]:
            state["ids"].append(obj_id)
            state["cond"][obj_id], state["memory"][obj_id], state["tracked"][obj_id] = {}, {}, set()
        center = (box[0] + box[2]) / 2
        is_cond = frame_idx not in state["tracked"][obj_id] or self.add_all
        state["tracked"][obj_id].add(frame_idx)
        (state["cond"] if is_cond else state["memory"])[obj_id][frame_idx] = center
        centers = {
            o: center if o == obj_id else self._stored(state, o, frame_idx) for o in state["ids"]
        }
        return frame_idx, list(state["ids"]), self._stack(state, frame_idx, centers)

    @staticmethod
    def _stored(state: dict[str, Any], obj_id: int, frame: int) -> float:
        merged = {**state["memory"][obj_id], **state["cond"][obj_id]}
        return merged.get(frame, 0.0)

    def propagate_in_video(
        self,
        state: dict[str, Any],
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], np.ndarray]]:
        self.calls.append((start_frame_idx, max_frame_num_to_track))
        if start_frame_idx is None:
            start_frame_idx = min(f for d in state["cond"].values() for f in d)
        span = state["n"] if max_frame_num_to_track is None else max_frame_num_to_track
        for frame in range(start_frame_idx, min(start_frame_idx + span, state["n"] - 1) + 1):
            centers = {}
            for obj in state["ids"]:
                if frame not in state["cond"][obj]:
                    prev = self._stored(state, obj, frame - 1)
                    state["memory"][obj][frame] = 0.8 * prev + 0.2 * (40 + frame)
                    state["tracked"][obj].add(frame)
                centers[obj] = self._stored(state, obj, frame)
            yield frame, list(state["ids"]), self._stack(state, frame, centers)


def _video(tmp_path: Path) -> VideoSample:
    build_davis_root(tmp_path)
    return load_davis_video(make_config(tmp_path), "zebra")


def test_choose_reprompt_uses_middle_frame_and_smallest_object(tmp_path: Path) -> None:
    video = _video(tmp_path)
    reprompt = choose_reprompt(video)
    # Object 1 is fully occluded (empty ground truth) at frame 10, so the next visible frame wins.
    assert (reprompt.frame, reprompt.obj_id) == (11, 1)
    assert reprompt.note == "object empty at frame 10"
    mask = video.gt_mask(1, 11)
    assert mask is not None
    assert reprompt.box.x0 == int(np.flatnonzero(mask.any(axis=0))[0])


def test_fresh_cond_tracks_zero_to_t_minus_one_then_t_onward(tmp_path: Path) -> None:
    video = _video(tmp_path)
    reprompt = choose_reprompt(video)
    predictor = FakePredictor(add_all=False)
    rec = run_fresh_cond(predictor, video, reprompt)
    assert predictor.calls == [(0, reprompt.frame - 1), (reprompt.frame, None)]
    assert rec.yielded == list(range(video.num_frames))
    assert set(rec.added) == set(video.object_ids)


def _rows(tmp_path: Path) -> list[dict[str, Any]]:
    video = _video(tmp_path)
    return run_video(FakePredictor, video, "fake")


def test_default_flag_correction_on_tracked_frame_must_fail(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    verdict = acceptance(rows)
    assert verdict[("zebra", "correct_tracked")] is False
    assert verdict[("zebra", "fresh_cond")] is True
    assert verdict[("zebra", "correct_tracked_cond")] is True
    tracked = [r for r in rows if r["sequence"] == "correct_tracked" and r["object_id"] == 1][0]
    assert tracked["after_frames_differ"] == 0
    assert tracked["frames_yielded"] == "11..19"
    assert tracked["mask_returned_by_add_differs_at_t"] is True


def test_fresh_cond_rows_show_identical_before_and_changed_after(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    row = [r for r in rows if r["sequence"] == "fresh_cond" and r["object_id"] == 1][0]
    assert row["before_identical"] is True
    assert row["before_max_abs_logit_diff"] == 0.0
    assert row["frames_yielded"] == "0..19"
    assert row["after_frames_total"] == 9
    assert row["after_frames_differ"] > 0
    assert row["pixels_changed_at_t"] > 0
    other = [r for r in rows if r["sequence"] == "fresh_cond" and r["object_id"] == 2][0]
    assert other["after_frames_differ"] == 0


def test_changed_frame_before_t_fails_acceptance(tmp_path: Path) -> None:
    video = _video(tmp_path)
    reprompt = choose_reprompt(video)
    _, base = run_baseline(FakePredictor(False), video)
    run = run_fresh_cond(FakePredictor(False), video, reprompt)
    run.logits[(3, 1)] = -run.logits[(3, 1)]
    rows = summarize("zebra", "fake", "fresh_cond", reprompt, base, run, video.object_ids)
    row = [r for r in rows if r["object_id"] == 1][0]
    assert row["before_identical"] is False
    assert row["before_max_abs_logit_diff"] == 2.0
    assert acceptance(rows)[("zebra", "fresh_cond")] is False


def test_unchanged_after_t_fails_even_if_before_is_identical() -> None:
    from experiments.spikes.reprompt import Reprompt
    from src.common.boxes import Box

    frames = range(6)
    logits = {(f, 1): np.full((4, 4), 1.0, dtype=np.float32) for f in frames}
    base = Recording(logits=dict(logits), yielded=list(frames))
    run = Recording(logits=dict(logits), yielded=list(frames))
    rows = summarize("v", "m", "fresh_cond", Reprompt(3, 1, Box(0, 0, 2, 2), ""), base, run, [1])
    assert rows[0]["before_identical"] is True and rows[0]["after_frames_differ"] == 0
    assert acceptance(rows)[("v", "fresh_cond")] is False


def test_csv_columns_and_verdict_lines(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    cfg = make_config(tmp_path)
    path = output_path(cfg, "tiny")
    assert path == tmp_path / "out" / "spikes" / "reprompt_tiny.csv"
    write_table(rows, path)
    table = pd.read_csv(path)
    assert tuple(table.columns) == COLUMNS
    assert len(table) == len(SEQUENCES) * 2
    lines = verdict_lines(rows)
    assert lines[0].startswith("correct_tracked: 0/1 videos pass")
    assert lines[1].startswith("fresh_cond: 1/1 videos pass")
    assert lines[2].startswith("correct_tracked_cond: 1/1 videos pass")
