import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.common.boxes import Box
from src.common.config import ConfigError
from src.recovery.box_check import BoxCheck
from src.recovery.recovery import RecoveryRun
from src.recovery.settings import RecoverySettings
from src.recovery.verify import (
    CANDIDATE_COLUMNS,
    VerifyThresholds,
    area_ratio,
    box_iou,
    candidate_path,
    load_verify_thresholds,
    mask_iou,
    write_candidates,
)
from tests.recovery.fakes import (
    NUM_FRAMES,
    ScriptedBoxes,
    ScriptedDetector,
    StatefulFake,
    expected_mask,
    ok,
)
from tests.recovery.golden_cases import run_cases
from tests.recovery.test_recovery import FIRED, NEW_BOX, OTHER_BOX, START_BOX, Collector, read_frame

OFF = VerifyThresholds(-math.inf, math.inf, -math.inf, False)
MIN_SCORE_5 = VerifyThresholds(5.0, math.inf, -math.inf, False)
MAX_RATIO_2 = VerifyThresholds(-math.inf, 2.0, -math.inf, False)
BIG_BOX = Box(24, 40, 42, 58)
BIG_FIRST = Box(24, 22, 42, 40)


def execute(
    flags: dict[tuple[int, int], tuple[str, ...]],
    results: dict[Any, Any],
    verifier: VerifyThresholds | None,
    prompts: dict[int, Box] | None = None,
    log: bool = False,
    predictor: StatefulFake | None = None,
    cooldown: int = 12,
) -> tuple[RecoveryRun, list[Any], Collector, StatefulFake]:
    predictor = predictor or StatefulFake()
    run = RecoveryRun(
        predictor,
        ScriptedDetector(flags),
        ScriptedBoxes(results),
        BoxCheck(),
        RecoverySettings(cooldown, "reinit"),
        verifier,
        log,
    )
    sink = Collector()
    events = run.run("vid", "frames", NUM_FRAMES, read_frame, prompts or {1: START_BOX}, 0, sink)
    return run, events, sink, predictor


def test_verification_off_matches_the_recorded_recovery_behaviour() -> None:
    golden = json.loads((Path(__file__).parent / "golden_e2.json").read_text())
    assert json.loads(json.dumps(run_cases())) == golden


def test_logging_alone_and_accept_all_do_not_change_the_run() -> None:
    flags, results = {(1, 5): FIRED}, {(1, 5): ok(NEW_BOX)}
    _, base_events, base, base_pred = execute(flags, results, None)
    for verifier, log in ((None, True), (OFF, False)):
        run, events, sink, predictor = execute(flags, results, verifier, log=log)
        assert events == base_events and predictor.calls == base_pred.calls
        assert sink.masks.keys() == base.masks.keys()
        assert all(np.array_equal(sink.masks[k], base.masks[k]) for k in base.masks)
        assert len(run.candidates) == 1 and run.candidates[0].accepted


def test_candidate_features_at_the_recovered_frame() -> None:
    run, _, _, _ = execute({(1, 5): FIRED}, {(1, 5): ok(NEW_BOX)}, None, log=True)
    row = run.candidates[0]
    assert (row.video, row.object_id, row.frame_index) == ("vid", 1, 5)
    assert row.cand_score == 4.0 and row.orig_score == 4.0 and row.reason == ""
    assert row.area_ratio == 1.0 and row.box_iou == 1.0
    flagged = expected_mask(START_BOX, 0, 5)
    assert row.mask_iou_orig == mask_iou(expected_mask(NEW_BOX, 5, 5), flagged)


def test_rejected_candidate_leaves_the_unprompted_run_and_is_not_replayed() -> None:
    _, _, base, _ = execute({}, {}, None)
    flags = {(1, 5): FIRED, (1, 20): FIRED}
    results = {(1, 5): ok(NEW_BOX), (1, 20): ok(NEW_BOX)}
    run, events, sink, predictor = execute(flags, results, MIN_SCORE_5, log=True)
    assert all(np.array_equal(sink.masks[k], base.masks[k]) for k in base.masks)
    assert [(e.frame_index, e.action, e.reason) for e in events] == [
        (5, "verify_rejected", "cand_score"),
        (20, "verify_rejected", "cand_score"),
    ]
    assert [c.accepted for c in run.candidates] == [False, False]
    assert predictor.calls.count(("reset",)) == 4
    replay = predictor.calls[predictor.calls.index(("propagate", 0, 4)) :]
    assert ("add", 5, 1, (30, 20, 42, 32)) in replay[:3]
    later_adds = [c for c in predictor.calls[predictor.calls.index(("propagate", 0, 19)) - 3 :]]
    assert ("add", 5, 1, (30, 20, 42, 32)) not in later_adds[:2]
    assert predictor.state["prompts"][1] == [(0, START_BOX)]


def test_second_restart_after_rejection_carries_only_accepted_boxes() -> None:
    prompts = {1: START_BOX, 2: OTHER_BOX}
    flags = {(1, 6): FIRED, (2, 6): FIRED}
    results = {(1, 6): ok(NEW_BOX), (2, 6): ok(BIG_BOX)}
    run, events, sink, predictor = execute(flags, results, MAX_RATIO_2, prompts, log=True)
    assert [(e.object_id, e.action, e.reason) for e in events] == [
        (1, "reprompted", ""),
        (2, "verify_rejected", "area_ratio"),
    ]
    assert predictor.calls.count(("reset",)) == 2
    assert np.array_equal(sink.masks[(1, 9)], expected_mask(NEW_BOX, 6, 9))
    assert np.array_equal(sink.masks[(2, 9)], expected_mask(OTHER_BOX, 0, 9))
    assert predictor.state["prompts"][2] == [(0, OTHER_BOX)]
    assert [(c.object_id, c.accepted) for c in run.candidates] == [(1, True), (2, False)]


def test_a_rejection_starts_the_cooldown() -> None:
    flags = {(1, 5): FIRED, (1, 8): FIRED, (1, 20): FIRED}
    results = {(1, 5): ok(NEW_BOX), (1, 20): ok(NEW_BOX)}
    _, events, _, _ = execute(flags, results, MIN_SCORE_5)
    assert [(e.frame_index, e.action) for e in events] == [
        (5, "verify_rejected"),
        (8, "cooldown"),
        (20, "verify_rejected"),
    ]


def test_tracker_state_is_untouched_by_a_rejection() -> None:
    detector, boxes = ScriptedDetector({(1, 5): FIRED}), ScriptedBoxes({(1, 5): ok(NEW_BOX)})
    run = RecoveryRun(
        StatefulFake(), detector, boxes, BoxCheck(), RecoverySettings(12, "reinit"), MIN_SCORE_5
    )
    run.run("vid", "f", NUM_FRAMES, read_frame, {1: START_BOX}, 0, Collector())
    assert boxes.started == [(1, 0)]


class Spy(StatefulFake):
    def __init__(self) -> None:
        super().__init__()
        self.yielded = 0

    def propagate_in_video(self, *args: Any, **kwargs: Any) -> Any:
        stream = super().propagate_in_video(*args, **kwargs)
        self.yielded = 0

        def counted() -> Any:
            for item in stream:
                self.yielded += 1
                yield item

        return counted()


def test_features_are_read_from_the_first_output_at_t_only() -> None:
    predictor = Spy()
    seen: list[int] = []

    class Probe(VerifyThresholds):
        def failed(self, *args: float) -> list[str]:
            seen.append(predictor.yielded)
            return []

    probe = Probe(-math.inf, math.inf, -math.inf, False)
    execute({(1, 5): FIRED}, {(1, 5): ok(NEW_BOX)}, probe, predictor=predictor)
    assert seen == [1]


def test_reliable_area_is_the_last_unflagged_mask() -> None:
    run, _, _, _ = execute({(1, 5): FIRED}, {(1, 5): ok(BIG_FIRST)}, None, log=True)
    assert run.candidates[0].area_ratio == 2.25
    assert run.candidates[0].box_iou == 1.0


def test_feature_helpers() -> None:
    assert area_ratio(0, 5) == math.inf and area_ratio(5, 0) == math.inf
    assert area_ratio(4, 2) == area_ratio(2, 4) == 2.0
    assert box_iou(Box(0, 0, 2, 2), Box(1, 0, 3, 2)) == pytest.approx(1 / 3)
    assert box_iou(None, Box(0, 0, 2, 2)) == 0.0 and box_iou(Box(0, 0, 1, 1), Box(2, 2, 3, 3)) == 0
    empty = np.zeros((4, 4), dtype=bool)
    assert mask_iou(empty, empty) == 0.0


def test_thresholds_failed_names_and_off_values() -> None:
    rule = VerifyThresholds(0.0, 2.0, 0.5, False)
    assert rule.failed(1.0, 1.0, 0.9) == []
    assert rule.failed(-1.0, 3.0, 0.1) == ["cand_score", "area_ratio", "box_iou"]
    assert rule.failed(math.nan, math.inf, 0.9) == ["cand_score", "area_ratio"]
    assert OFF.failed(-100.0, math.inf, 0.0) == []
    assert VerifyThresholds(0, 1, 0, True).failed(9.0, 1.0, 1.0) == ["accept_nothing"]


def test_missing_verify_file_refuses(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not frozen"):
        load_verify_thresholds("tiny", tmp_path / "verify.yaml")


def test_verify_file_loads_per_model(tmp_path: Path) -> None:
    path = tmp_path / "verify.yaml"
    body = "s_min: 2.0\n  r_max: .inf\n  b_min: 0.3\n  accept_nothing: false\n"
    nothing = "s_min: -.inf\n  r_max: .inf\n  b_min: -.inf\n  accept_nothing: true\n"
    path.write_text(f"kind: verify\ntiny:\n  {body}large:\n  {nothing}")
    tiny, large = load_verify_thresholds("tiny", path), load_verify_thresholds("large", path)
    assert tiny == VerifyThresholds(2.0, math.inf, 0.3, False) and large.accept_nothing
    with pytest.raises(ConfigError):
        load_verify_thresholds("huge", path)


def test_candidate_file_layout_and_columns(tmp_path: Path) -> None:
    run, _, _, _ = execute({(1, 5): FIRED}, {(1, 5): ok(NEW_BOX)}, None, log=True)
    path = candidate_path(tmp_path, "fix_rules_csrt_tiny", "sav_val", "vid")
    assert path == tmp_path / "fix_rules_csrt_tiny" / "sav_val" / "candidates" / "vid.csv"
    write_candidates(path, run.candidates)
    header = path.read_text().splitlines()[0]
    assert header.split(",") == list(CANDIDATE_COLUMNS)
