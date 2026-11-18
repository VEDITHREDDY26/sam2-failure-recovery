"""Tests for the ViT tracker box sources using a fake tracker."""

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from src.common.boxes import Box
from src.common.config import ConfigError
from src.recovery.sources import MotionBoxSource, ViTBoxSource
from src.recovery.vit_tracker import (
    ViTConfig,
    ViTTracker,
    load_vit_config,
    make_vit_factory,
    vit_model_path,
)
from tests.recovery.test_tracker import H, W, frame_with_square, mask_for

BLANK = np.zeros((H, W, 3), dtype=np.uint8)


class FakeViT:
    """Records init boxes and update calls; update returns the scripted rect and score."""

    created: list["FakeViT"] = []

    def __init__(self, rect: tuple[int, int, int, int] = (10, 30, 24, 24), score: float = 0.9):
        self.rect, self.score = rect, score
        self.inits: list[tuple[int, int, int, int]] = []
        self.updates = 0
        self.found = True
        self.fail_init = False

    def init(self, image: Any, bounding_box: Any) -> None:
        if self.fail_init:
            raise cv2.error("refused")
        self.inits.append(tuple(bounding_box))

    def update(self, image: Any) -> tuple[bool, tuple[int, int, int, int]]:
        self.updates += 1
        return self.found, self.rect

    def getTrackingScore(self) -> float:  # noqa: N802
        return self.score


def make_source(mode: str, rect: tuple[int, int, int, int] = (10, 30, 24, 24)) -> ViTBoxSource:
    FakeViT.created = []

    def factory() -> FakeViT:
        tracker = FakeViT(rect)
        FakeViT.created.append(tracker)
        return tracker

    return ViTBoxSource(factory, mode, anchor_iou=0.5)


def total(attr: str) -> int:
    return sum(
        len(getattr(t, attr)) if attr == "inits" else getattr(t, attr) for t in FakeViT.created
    )


def test_wrapper_reports_box_and_score_and_converts_xywh() -> None:
    tracker = ViTTracker(lambda: FakeViT((10, 30, 24, 24), 0.7))
    assert tracker.update(BLANK).status == "not_initialised"
    assert tracker.reinit(BLANK, mask_for(10), 0) == "reinitialised"
    result = tracker.update(BLANK)
    assert (result.status, result.box, result.score) == ("ok", Box(10, 30, 34, 54), 0.7)


def test_reinit_reuses_one_tracker_in_place() -> None:
    made: list[FakeViT] = []

    def factory() -> FakeViT:
        made.append(FakeViT())
        return made[-1]

    tracker = ViTTracker(factory)
    for index in range(3):
        tracker.reinit(BLANK, mask_for(10 + index), index)
    assert len(made) == 1 and len(made[0].inits) == 3


def test_wrapper_init_failure_and_empty_mask_keep_state() -> None:
    fake = FakeViT()
    tracker = ViTTracker(lambda: fake)
    assert tracker.reinit(BLANK, np.zeros((H, W), dtype=bool), 0) == "empty_mask"
    fake.fail_init = True
    assert tracker.reinit(BLANK, mask_for(10), 0) == "init_failed"
    assert not tracker.initialised


@pytest.mark.parametrize("found,rect", [(False, (1, 1, 5, 5)), (True, (1, 1, 0, 5))])
def test_wrapper_lost(found: bool, rect: tuple[int, int, int, int]) -> None:
    fake = FakeViT(rect)
    fake.found = found
    tracker = ViTTracker(lambda: fake)
    tracker.reinit(BLANK, mask_for(10), 0)
    assert tracker.update(BLANK).status == "lost"


def test_wrapper_clips_to_frame() -> None:
    tracker = ViTTracker(lambda: FakeViT((W - 10, 30, 24, 24)))
    tracker.reinit(BLANK, mask_for(10), 0)
    result = tracker.update(BLANK)
    assert result.box == Box(W - 10, 30, W, 54) and result.raw_box == Box(W - 10, 30, W + 14, 54)


def test_reinit_mode_restarts_every_unflagged_frame_and_never_updates() -> None:
    source = make_source("reinit")
    source.start(1, 0, BLANK, mask_for(10))
    source.observe(1, 1, BLANK, mask_for(12))
    source.skip(1, 2, BLANK)
    assert (total("inits"), total("updates")) == (2, 0)
    assert source.propose(1, 3, BLANK).score == 0.9
    assert source.last_reliable_box(1) == Box(12, 30, 36, 54)


def test_continuous_mode_only_updates_and_skip_updates() -> None:
    source = make_source("continuous")
    source.start(1, 0, BLANK, mask_for(10))
    source.observe(1, 1, BLANK, mask_for(40))
    source.skip(1, 2, BLANK)
    assert (total("inits"), total("updates")) == (1, 2)


def test_anchored_restarts_only_when_it_agrees_with_sam2() -> None:
    source = make_source("anchored", rect=(10, 30, 24, 24))
    source.start(1, 0, BLANK, mask_for(10))
    source.observe(1, 1, BLANK, mask_for(12))  # IoU high: restart
    assert total("inits") == 2
    source.observe(1, 2, BLANK, mask_for(90))  # far from the tracker box: keep own track
    assert (total("inits"), total("updates")) == (2, 2)
    source.skip(1, 3, BLANK)
    assert total("updates") == 3


def test_empty_mask_start_is_ignored_and_proposal_not_initialised() -> None:
    source = make_source("reinit")
    source.start(1, 0, BLANK, np.zeros((H, W), dtype=bool))
    assert source.propose(1, 1, BLANK).status == "not_initialised"
    assert source.last_reliable_box(1) is None


def test_unknown_mode_refused() -> None:
    with pytest.raises(ValueError):
        ViTBoxSource(lambda: FakeViT(), "nope")


def box_mask(box: Box) -> npt.NDArray[np.bool_]:
    mask = np.zeros((H, W), dtype=bool)
    mask[box.y0 : box.y1, box.x0 : box.x1] = True
    return mask


def test_motion_extrapolates_constant_velocity() -> None:
    source = MotionBoxSource()
    source.start(1, 0, BLANK, box_mask(Box(10, 10, 20, 20)))
    source.observe(1, 2, BLANK, box_mask(Box(14, 12, 24, 22)))
    result = source.propose(1, 5, BLANK)
    assert result.box == Box(20, 15, 30, 25) and result.score is None


def test_motion_uses_only_the_last_two_and_one_box_is_returned_as_is() -> None:
    source = MotionBoxSource()
    assert source.propose(1, 3, BLANK).status == "not_initialised"
    source.start(1, 0, BLANK, box_mask(Box(10, 10, 20, 20)))
    assert source.propose(1, 4, BLANK).box == Box(10, 10, 20, 20)
    source.observe(1, 1, BLANK, box_mask(Box(12, 10, 22, 20)))
    source.observe(1, 2, BLANK, box_mask(Box(14, 10, 24, 20)))
    source.observe(1, 3, BLANK, np.zeros((H, W), dtype=bool))
    assert source.propose(1, 4, BLANK).box == Box(18, 10, 28, 20)


def test_motion_clips_and_reports_lost_outside() -> None:
    source = MotionBoxSource()
    source.start(1, 0, BLANK, box_mask(Box(100, 10, 110, 20)))
    source.observe(1, 1, BLANK, box_mask(Box(110, 10, 120, 20)))
    assert source.propose(1, 2, BLANK).box == Box(120, 10, W, 20)
    result = source.propose(1, 5, BLANK)
    assert result.status == "lost" and result.raw_box == Box(150, 10, 160, 20)


def test_motion_shrinking_to_nothing_is_lost() -> None:
    source = MotionBoxSource()
    source.start(1, 0, BLANK, box_mask(Box(10, 10, 30, 20)))
    source.observe(1, 1, BLANK, box_mask(Box(15, 10, 25, 20)))
    assert source.propose(1, 4, BLANK).status == "lost"


def config_for(sha: str) -> ViTConfig:
    return ViTConfig("m/model.onnx", sha, 0.5, 0.0)


def test_sha256_mismatch_and_missing_file_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        vit_model_path(config_for("0" * 64), tmp_path)
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "model.onnx").write_bytes(b"abc")
    with pytest.raises(ConfigError, match="sha256"):
        vit_model_path(config_for("0" * 64), tmp_path)
    good = hashlib.sha256(b"abc").hexdigest()
    assert vit_model_path(config_for(good), tmp_path) == tmp_path / "m" / "model.onnx"


def test_shipped_config_loads() -> None:
    cfg = load_vit_config()
    assert (cfg.anchor_iou, cfg.score_threshold_internal) == (0.5, 0.0)


def test_real_model_follows_a_moving_square() -> None:
    root = os.environ.get("SAM2FR_DATA_ROOT", "")
    cfg = load_vit_config()
    if not root or not (Path(root) / cfg.model).is_file():
        pytest.skip("ViT model file not present under SAM2FR_DATA_ROOT")
    tracker = ViTTracker(make_vit_factory(cfg, Path(root)))
    assert tracker.reinit(frame_with_square(10), mask_for(10), 0) == "reinitialised"
    result = tracker.update(frame_with_square(14))
    assert result.status == "ok" and result.box is not None and result.score is not None
    assert abs(result.box.x0 - 14) <= 6 and abs(result.box.y0 - 30) <= 6


def test_anchored_restarts_a_lost_or_unstarted_tracker_on_a_non_empty_mask() -> None:
    source = make_source("anchored")
    source.start(1, 0, BLANK, np.zeros((H, W), dtype=bool))  # never started
    source.observe(1, 1, BLANK, mask_for(10))
    assert total("inits") == 1
    FakeViT.created[-1].found = False
    source.observe(1, 2, BLANK, mask_for(12))  # lost: restart
    assert total("inits") == 2
    FakeViT.created[-1].found = False
    source.observe(1, 3, BLANK, np.zeros((H, W), dtype=bool))  # lost + empty: no restart
    assert total("inits") == 2


def test_continuous_never_restarts_a_lost_tracker() -> None:
    source = make_source("continuous")
    source.start(1, 0, BLANK, mask_for(10))
    FakeViT.created[-1].found = False
    source.observe(1, 1, BLANK, mask_for(12))
    assert total("inits") == 1


def test_real_model_in_place_reinit_equals_a_fresh_tracker() -> None:
    root = os.environ.get("SAM2FR_DATA_ROOT", "")
    cfg = load_vit_config()
    if not root or not (Path(root) / cfg.model).is_file():
        pytest.skip("ViT model file not present under SAM2FR_DATA_ROOT")
    factory = make_vit_factory(cfg, Path(root))
    reused = ViTTracker(factory)
    reused.reinit(frame_with_square(60), mask_for(60), 0)
    reused.update(frame_with_square(62))
    for offset in (10, 16):
        reused.reinit(frame_with_square(offset), mask_for(offset), 0)
        fresh = ViTTracker(factory)
        fresh.reinit(frame_with_square(offset), mask_for(offset), 0)
        for step in range(1, 5):
            frame = frame_with_square(offset + 2 * step)
            assert reused.update(frame) == fresh.update(frame)
