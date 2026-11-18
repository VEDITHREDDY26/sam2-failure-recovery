"""Tests for the gate pilot script using a fake ground-truth reader and generated inputs."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from experiments.e13 import pilot
from src.common.boxes import Box
from src.common.masks import LabelMap
from src.data.video import VideoSample
from src.recovery.box_check import BoxCheck
from src.recovery.tracker import TrackResult
from src.sam2_runner.prompts import Prompts
from tests.recovery.fakes import ScriptedDetector
from tests.recovery.test_vit_sources import FakeViT

H, W, N = 64, 96, 40
FIRED = ("low_score",)


class FakeReader:
    def __init__(self, annotated: tuple[int, ...], masks: dict[int, npt.NDArray[np.bool_]]):
        self.annotated, self.masks = annotated, masks

    def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
        return self.annotated

    def read_mask(self, obj_id: int, frame_index: int) -> npt.NDArray[np.bool_]:
        return self.masks[frame_index]


def square(x0: int, y0: int = 20, size: int = 16) -> npt.NDArray[np.bool_]:
    mask = np.zeros((H, W), dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


def make_inputs(
    annotated: tuple[int, ...] = tuple(range(0, N, 4)),
    gt: dict[int, npt.NDArray[np.bool_]] | None = None,
    sam2: dict[int, npt.NDArray[np.bool_]] | None = None,
) -> pilot.VideoInputs:
    """SAM 2 alone follows the object (square at 10 + f) unless `sam2` overrides a frame."""
    truth = gt or {f: square(10 + f) for f in annotated}
    sam2 = sam2 or {}
    reader = FakeReader(annotated, truth)
    video = VideoSample(
        "sav",
        "sav_val",
        "vid",
        Path("."),
        tuple(str(i) for i in range(N)),
        (1,),
        {1: "o"},
        reader,
    )
    prompts = Prompts("vid", 0, {1: Box(10, 20, 26, 36)})

    def labels(index: int) -> LabelMap:
        mask = sam2.get(index, square(10 + index))
        return mask.astype(np.uint8)

    def read_frame(index: int) -> npt.NDArray[np.uint8]:
        return np.random.default_rng(index).integers(0, 255, (H, W, 3), dtype=np.uint8)

    return pilot.VideoInputs(video, prompts, labels, {1: dict.fromkeys(range(N), 1.0)}, read_frame)


class Recorder:
    """A box source that logs its calls and proposes a fixed box."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.box = Box(10, 20, 26, 36)

    def reset(self) -> None:
        self.calls.clear()

    def start(self, obj_id: int, frame_index: int, frame: Any, mask: Any) -> None:
        self.calls.append(("start", frame_index))

    def observe(self, obj_id: int, frame_index: int, frame: Any, mask: Any) -> None:
        self.calls.append(("observe", frame_index))

    def skip(self, obj_id: int, frame_index: int, frame: Any) -> None:
        self.calls.append(("skip", frame_index))

    def propose(self, obj_id: int, frame_index: int, frame: Any) -> TrackResult:
        self.calls.append(("propose", frame_index))
        return TrackResult("ok", self.box, self.box, 0.8)

    def last_reliable_box(self, obj_id: int) -> Box | None:
        return Box(10, 20, 26, 36)


def replay(
    inputs: pilot.VideoInputs, flags: dict[int, tuple[str, ...]], cooldown: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Recorder]]:
    sources = {arm: Recorder() for arm in pilot.ARMS}
    detector = ScriptedDetector({(1, f): fired for f, fired in flags.items()})
    cand, timing = pilot.replay_video(inputs, detector, sources, BoxCheck(), cooldown)  # type: ignore[arg-type]
    return cand, timing, sources


def test_candidates_start_the_cooldown_and_nothing_restarts() -> None:
    cand, _, sources = replay(make_inputs(), {5: FIRED, 8: FIRED, 17: FIRED, 18: FIRED})
    for arm in pilot.ARMS:
        assert cand[cand["arm"] == arm]["frame_index"].tolist() == [5, 17]
    calls = sources["vit_reinit"].calls
    assert calls[0] == ("start", 0) and [c for c in calls if c[0] == "start"] == [("start", 0)]


def test_loop_order_start_observe_skip_propose() -> None:
    _, _, sources = replay(make_inputs(), {3: FIRED, 5: FIRED, 6: FIRED})
    calls = sources["csrt_reinit"].calls
    assert calls[:8] == [
        ("start", 0),
        ("observe", 1),
        ("observe", 2),
        ("propose", 3),
        ("observe", 4),
        ("skip", 5),
        ("skip", 6),
        ("observe", 7),
    ]


def test_timing_counts_every_source_call() -> None:
    _, timing, _ = replay(make_inputs(), {3: FIRED})
    assert timing["arm"].tolist() == list(pilot.ARMS)
    assert (timing["calls"] == N).all() and (timing["seconds"] >= 0).all()


def test_scored_frame_is_t_when_annotated_else_the_next_annotated_frame() -> None:
    inputs = make_inputs()
    cand, _, _ = replay(inputs, {5: FIRED, 8: FIRED, 29: FIRED}, cooldown=1)
    by_frame = cand[cand["arm"] == "motion"].set_index("frame_index")
    assert by_frame.loc[5, "scored_frame"] == 8 and by_frame.loc[8, "scored_frame"] == 8
    assert by_frame.loc[29, "scored_frame"] == 32


def test_no_annotated_frame_left_is_unscored_and_has_blank_gt_columns() -> None:
    inputs = make_inputs(annotated=(0, 4, 8))
    cand, _, _ = replay(inputs, {20: FIRED})
    row = cand.iloc[0]
    assert row["cls"] == "unscored" and row["scored_frame"] == "" and row["iou_gt"] == ""


def test_class_assignment() -> None:
    annotated = (0, 4, 8, 12)
    gt = {f: square(10 + f) for f in annotated}
    gt[8] = np.zeros((H, W), dtype=bool)  # object absent
    sam2 = {4: square(60), 12: square(22)}  # wrong at 4; right at 12
    cand, _, _ = replay(make_inputs(annotated, gt, sam2), {4: FIRED, 8: FIRED, 12: FIRED}, 1)
    got = cand[cand["arm"] == "motion"].set_index("frame_index")["cls"].to_dict()
    assert got == {4: "real_failure", 8: "not_visible", 12: "false_alarm"}


def test_row_fields_for_a_good_and_a_bad_box() -> None:
    cand, _, _ = replay(make_inputs(), {4: FIRED})
    row = cand[cand["arm"] == "motion"].iloc[0]
    assert row["status"] == "ok" and row["score"] == 0.8
    assert row["iou_sam2"] == pytest.approx(row["iou_gt"]) and row["gt_visible"]
    assert row["box_check_ok"] and row["first_of_object"] and not row["sam2_mask_empty"]
    assert row["fired"] == "low_score"


def test_first_of_object_and_empty_sam2_mask() -> None:
    sam2 = {17: np.zeros((H, W), dtype=bool)}
    cand, _, _ = replay(make_inputs(sam2=sam2), {5: FIRED, 17: FIRED})
    rows = cand[cand["arm"] == "vit_anchored"].set_index("frame_index")
    assert rows["first_of_object"].tolist() == [True, False]
    assert rows.loc[17, "sam2_mask_empty"] and rows.loc[17, "iou_sam2"] == 0.0


def test_real_sources_with_a_fake_vit_give_every_arm_per_candidate() -> None:
    inputs = make_inputs()
    sources = {a: pilot.make_source(a, lambda: FakeViT((12, 20, 16, 16)), 0.5) for a in pilot.ARMS}
    detector = ScriptedDetector({(1, 6): FIRED, (1, 30): FIRED})
    cand, _ = pilot.replay_video(inputs, detector, sources, BoxCheck(), 12)  # type: ignore[arg-type]
    assert list(cand.columns) == list(pilot.CANDIDATE_COLUMNS)
    assert len(cand) == 2 * len(pilot.ARMS)
    assert set(cand["status"]) <= {"ok", "lost", "not_initialised"}


def test_other_splits_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="refused"):
        pilot.check_split("davis_val")
    monkeypatch.setenv("SAM2FR_DATA_ROOT", "/nonexistent")
    with pytest.raises(SystemExit, match="refused"):
        pilot.main(["--model", "tiny", "--split", "sav_test"])


def test_select_videos_shards_are_disjoint_and_cover_everything() -> None:
    names = [f"v{i:02d}" for i in range(10)]
    shards = [pilot.select_videos(names, None, k, 3) for k in range(3)]
    assert sorted(sum(shards, [])) == names and shards[0] == ["v00", "v03", "v06", "v09"]
    assert pilot.select_videos(names, ["v05", "v01"], 0, 1) == ["v01", "v05"]
    with pytest.raises(ValueError):
        pilot.select_videos(names, ["nope"], 0, 1)
    with pytest.raises(ValueError):
        pilot.select_videos(names, None, 3, 3)


def test_resume_and_merge(tmp_path: Path) -> None:
    out, results = tmp_path / "out", tmp_path / "results"
    inputs = make_inputs()
    assert not pilot.is_done(out, "tiny", "vid")
    pilot.run_part(
        inputs,
        out,
        "tiny",
        ScriptedDetector({(1, 5): FIRED}),  # type: ignore[arg-type]
        lambda: FakeViT(),
        0.5,
        BoxCheck(),
        12,
    )
    assert pilot.is_done(out, "tiny", "vid")
    assert not list(out.rglob("*.tmp"))
    assert pilot.merge_parts(out, "tiny", ["vid", "other"], results) == ["other"]
    assert not results.exists()
    cand_path, timing_path = pilot.merge_parts(out, "tiny", ["vid"], results)  # type: ignore[misc]
    assert cand_path.name == "e13_pilot_candidates_tiny_sav_val.csv"
    assert len(pd.read_csv(cand_path)) == len(pilot.ARMS)
    assert len(pd.read_csv(timing_path)) == len(pilot.ARMS)
