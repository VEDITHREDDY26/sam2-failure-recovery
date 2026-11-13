from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.common.boxes import Box
from src.common.config import DataConfig
from src.common.layout import events_path, mask_path
from src.common.masks import read_label_map
from src.data.video import VideoSample
from src.oracle.online import OracleBoxSource, OracleDetector
from src.recovery.box_check import BoxCheck
from src.recovery.online_runner import run_batch
from src.recovery.settings import RecoverySettings
from src.recovery.sources import BoxSource, Detector
from src.sam2_runner.prompts import BatchPrompts
from tests.recovery.fakes import NUM_FRAMES, H, StatefulFake, W, expected_mask
from tests.recovery.online_helpers import SPLIT, START_BOX, VIDEO, make_tree

METHOD = "oracle_flags_oracle_box_test"
SPEED = 2


class MovingGt:
    """The object really moves SPEED px per frame; the fake predictor moves 1 px per frame."""

    def __init__(self, annotated: tuple[int, ...]) -> None:
        self.annotated = annotated

    def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
        return self.annotated

    def read_mask(self, obj_id: int, frame_index: int) -> Any:
        return expected_mask(START_BOX, 0, SPEED * frame_index)


def sample(cfg: DataConfig, annotated: tuple[int, ...] | None = None) -> VideoSample:
    stems = tuple(f"{i:05d}" for i in range(NUM_FRAMES))
    frames = cfg.data_root / "DAVIS" / "JPEGImages" / "480p" / VIDEO
    return VideoSample(
        "davis",
        "val",
        VIDEO,
        frames,
        stems,
        (1,),
        {1: "obj"},
        MovingGt(annotated if annotated is not None else tuple(range(NUM_FRAMES))),
    )


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DataConfig, BatchPrompts, Path]:
    cfg, prompts = make_tree(tmp_path, monkeypatch)
    return cfg, prompts, cfg.data_root / cfg.output_root


def test_adapters_satisfy_the_loop_protocols(tree: Any) -> None:
    cfg, _, _ = tree
    detector: Detector = OracleDetector(sample(cfg))
    boxes: BoxSource = OracleBoxSource(sample(cfg))
    assert detector.update(1, 0, expected_mask(START_BOX, 0, 0), 5.0, (H, W)).flag is False
    assert boxes.last_reliable_box(1) is None


def test_detector_flags_only_where_ground_truth_disagrees(tree: Any) -> None:
    cfg, _, _ = tree
    detector = OracleDetector(sample(cfg))
    pred = expected_mask(START_BOX, 0, 3)
    assert detector.update(1, 3, pred, 0.0, (H, W)).flag is False
    wrong = expected_mask(START_BOX, 0, 3)
    decision = detector.update(1, 10, wrong, 0.0, (H, W))
    assert decision.flag is True and decision.fired == ("oracle",)


def test_unannotated_frame_gives_no_flag_and_no_box(tree: Any) -> None:
    cfg, _, _ = tree
    video = sample(cfg, annotated=(0, 4, 8))
    far = expected_mask(START_BOX, 0, 0)
    assert OracleDetector(video).update(1, 5, far, 0.0, (H, W)).flag is False
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    assert OracleBoxSource(video).propose(1, 5, frame).status == "lost"


def test_box_is_the_clipped_ground_truth_box(tree: Any) -> None:
    cfg, _, _ = tree
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    result = OracleBoxSource(sample(cfg)).propose(1, 5, frame)
    assert result.status == "ok" and result.box == Box(20, 20, 32, 32)
    late = OracleBoxSource(sample(cfg)).propose(1, 28, frame)
    assert late.status == "lost"


def run(cfg: DataConfig, prompts: BatchPrompts, method: str = METHOD) -> None:
    run_batch(
        prompts,
        cfg,
        method,
        StatefulFake,
        lambda video: OracleDetector(sample(cfg)),
        lambda video: OracleBoxSource(sample(cfg)),
        BoxCheck(),
        RecoverySettings(12, "reinit"),
    )


def test_runner_reprompts_where_ground_truth_says_failure(tree: Any) -> None:
    cfg, prompts, out = tree
    run(cfg, prompts)
    events = pd.read_csv(events_path(out, METHOD, SPLIT, VIDEO))
    first = events.iloc[0]
    assert (first["action"], first["frame_index"], first["fired"]) == ("reprompted", 5, "oracle")
    assert (first["box_x0"], first["box_x1"]) == (20, 32)
    label = read_label_map(mask_path(out, METHOD, SPLIT, VIDEO, "00005"))
    assert np.array_equal(label == 1, expected_mask(Box(20, 20, 32, 32), 5, 5))
    before = read_label_map(mask_path(out, METHOD, SPLIT, VIDEO, "00004"))
    assert np.array_equal(before == 1, expected_mask(START_BOX, 0, 4))


def test_oracle_method_name_is_needed_for_oracle_pieces_not_enforced_here(tree: Any) -> None:
    cfg, prompts, out = tree
    with pytest.raises(ValueError, match="never writes"):
        run(cfg, prompts, method="sam2_tiny")
