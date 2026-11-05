import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline.config import METHOD, BaselineConfig
from src.baseline.opencv_ops import BgrImage, TrackResult
from src.baseline.runner import (
    STATUS_COLUMNS,
    TIMING_COLUMNS,
    clear_video,
    is_complete,
    run_video,
    timing_path,
    tracker_path,
)
from src.common.boxes import Box
from src.common.layout import mask_path
from src.common.masks import BoolMask, label_map_to_masks, read_label_map
from src.data.davis import load_davis_video
from src.data.sav import load_sav_video
from src.data.video import GtReader, VideoSample
from tests.data.helpers import (
    LATE_FIRST_FRAME,
    SAV_NAME,
    build_davis_root,
    build_sav_root,
    make_config,
)
from tests.fixtures.synthetic_video import NUM_FRAMES

CFG = BaselineConfig(grabcut_iterations=3, roi_margin_frac=0.25, min_box_side=4)


class SpyReader:
    """Wraps a reader; records every ground-truth read and fails after the allowed frame."""

    def __init__(self, inner: GtReader, last_allowed: dict[int, int]) -> None:
        self.inner = inner
        self.last_allowed = last_allowed
        self.reads: list[tuple[int, int]] = []

    def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
        return self.inner.annotated_frames(obj_id)

    def read_mask(self, obj_id: int, frame_index: int) -> BoolMask:
        self.reads.append((obj_id, frame_index))
        if frame_index > self.last_allowed[obj_id]:
            raise AssertionError(f"ground truth read after the start frame: {obj_id}@{frame_index}")
        return self.inner.read_mask(obj_id, frame_index)


class ScriptedTracker:
    """Reports the start box until `lose_from`, then failure."""

    def __init__(self, lose_from: int, log: list[int]) -> None:
        self.lose_from = lose_from
        self.log = log
        self.box = Box(0, 0, 1, 1)
        self.frame = 0

    def init(self, frame: BgrImage, box: Box) -> None:
        self.box = box
        self.frame = 0

    def update(self, frame: BgrImage) -> TrackResult:
        self.frame += 1
        self.log.append(self.frame)
        if self.frame >= self.lose_from:
            return TrackResult(False, None)
        return TrackResult(True, self.box)


@pytest.fixture
def davis_video(tmp_path: Path) -> VideoSample:
    build_davis_root(tmp_path / "data", ("zebra",))
    return load_davis_video(make_config(tmp_path / "data"), "zebra")


def test_one_png_per_frame_and_all_objects_present(
    davis_video: VideoSample, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = run_video(davis_video, CFG, out, METHOD, "davis_val")
    for stem in davis_video.frame_stems:
        assert mask_path(out, METHOD, "davis_val", "zebra", stem).is_file()
    first = label_map_to_masks(
        read_label_map(mask_path(out, METHOD, "davis_val", "zebra", davis_video.frame_stems[0]))
    )
    assert set(first) == {1, 2}
    assert len(result.status) == NUM_FRAMES * 2
    assert list(result.status.columns) == list(STATUS_COLUMNS)
    assert is_complete(
        out, METHOD, "davis_val", "zebra", davis_video.frame_stems, list(davis_video.object_ids)
    )


def test_timing_row_has_hardware_and_fps(davis_video: VideoSample, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_video(davis_video, CFG, out, METHOD, "davis_val")
    row = pd.read_csv(timing_path(out, METHOD, "davis_val", "zebra"))
    assert list(row.columns) == list(TIMING_COLUMNS)
    assert row.loc[0, "frames"] == NUM_FRAMES and row.loc[0, "objects"] == 2
    assert row.loc[0, "compute_s"] <= row.loc[0, "wall_s"]
    assert row.loc[0, "fps"] == pytest.approx(NUM_FRAMES / row.loc[0, "compute_s"])


def test_late_object_is_empty_before_its_start_frame(tmp_path: Path) -> None:
    build_sav_root(tmp_path / "data", "sav_test")
    video = load_sav_video(make_config(tmp_path / "data"), "test", SAV_NAME)
    out = tmp_path / "out"
    result = run_video(video, CFG, out, METHOD, "sav_test")
    late = result.status[result.status.object_id == 3].set_index("frame_index")
    assert (late.loc[: LATE_FIRST_FRAME - 1, "area"] == 0).all()
    assert late.loc[: LATE_FIRST_FRAME - 1, "tracker_ok"].isna().all()
    assert late.loc[LATE_FIRST_FRAME, "tracker_ok"] == 1 and late.loc[LATE_FIRST_FRAME, "area"] > 0
    for index in range(LATE_FIRST_FRAME):
        label = read_label_map(mask_path(out, METHOD, "sav_test", SAV_NAME, f"{index:05d}"))
        assert 3 not in label_map_to_masks(label)


def test_tracker_loss_gives_empty_mask_and_no_ground_truth(
    davis_video: VideoSample, tmp_path: Path
) -> None:
    spy = SpyReader(davis_video.reader, {1: 0, 2: 0})
    video = dataclasses.replace(davis_video, reader=spy)
    log: list[int] = []
    out = tmp_path / "out"
    factories = iter([ScriptedTracker(6, log), ScriptedTracker(10_000, [])])
    result = run_video(video, CFG, out, METHOD, "davis_val", lambda: next(factories))
    one = result.status[result.status.object_id == 1].set_index("frame_index")
    assert (one.loc[:5, "tracker_ok"] == 1).all() and (one.loc[:5, "area"] > 0).all()
    assert (one.loc[6:, "tracker_ok"] == 0).all()
    assert (one.loc[6:, "area"] == 0).all() and one.loc[6:, "x0"].isna().all()
    assert log == list(range(1, NUM_FRAMES))  # kept updating, never re-initialised
    two = result.status[result.status.object_id == 2]
    assert (two["area"] > 0).all()
    label = read_label_map(mask_path(out, METHOD, "davis_val", "zebra", "00010"))
    assert set(label_map_to_masks(label)) == {2}
    assert {frame for _, frame in spy.reads} == {0}


def test_grabcut_fallback_is_recorded(davis_video: VideoSample, tmp_path: Path) -> None:
    flat = dataclasses.replace(davis_video, frames_dir=tmp_path / "flat")
    flat.frames_dir.mkdir()
    from PIL import Image

    for stem in flat.frame_stems:
        Image.fromarray(np.full((64, 96, 3), 120, dtype=np.uint8)).save(
            flat.frames_dir / f"{stem}.jpg"
        )
    result = run_video(
        flat, CFG, tmp_path / "out", METHOD, "davis_val", lambda: ScriptedTracker(99, [])
    )
    assert result.status["grabcut_fallback"].sum() >= 1
    fell = result.status[result.status.grabcut_fallback == 1]
    assert (fell["area"] > 0).all()


def test_incomplete_video_is_not_complete_and_clear_removes_outputs(
    davis_video: VideoSample, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_video(davis_video, CFG, out, METHOD, "davis_val")
    ids, stems = list(davis_video.object_ids), davis_video.frame_stems
    mask_path(out, METHOD, "davis_val", "zebra", stems[3]).unlink()
    assert not is_complete(out, METHOD, "davis_val", "zebra", stems, ids)
    clear_video(out, METHOD, "davis_val", "zebra")
    assert not tracker_path(out, METHOD, "davis_val", "zebra").exists()
    assert not list((out / METHOD / "davis_val" / "masks" / "zebra").glob("*.png"))


class RecordingReader:
    """Records every ground-truth read; reads are allowed only at each object's start frame."""

    def __init__(self, inner: GtReader, starts: dict[int, int]) -> None:
        self.inner = inner
        self.starts = starts
        self.reads: list[tuple[int, int]] = []

    def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
        return self.inner.annotated_frames(obj_id)

    def read_mask(self, obj_id: int, frame_index: int) -> BoolMask:
        self.reads.append((obj_id, frame_index))
        if frame_index != self.starts[obj_id]:
            raise AssertionError(f"ground truth read at {obj_id}@{frame_index}")
        return self.inner.read_mask(obj_id, frame_index)


def test_late_start_reads_ground_truth_only_at_each_start_frame(tmp_path: Path) -> None:
    build_sav_root(tmp_path / "data", "sav_test")
    base = load_sav_video(make_config(tmp_path / "data"), "test", SAV_NAME)
    starts = {1: 0, 2: 0, 3: LATE_FIRST_FRAME}
    spy = RecordingReader(base.reader, starts)
    video = dataclasses.replace(base, reader=spy)
    out = tmp_path / "out"
    result = run_video(video, CFG, out, METHOD, "sav_test")
    assert sorted(spy.reads) == sorted(starts.items())
    late = result.status[result.status.object_id == 3].set_index("frame_index")
    assert (late.loc[: LATE_FIRST_FRAME - 1, "area"] == 0).all()
    assert late.loc[LATE_FIRST_FRAME, "area"] > 0
