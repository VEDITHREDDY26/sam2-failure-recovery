from pathlib import Path

import numpy as np
import pytest

from src.common.boxes import Box
from src.common.config import ConfigError
from src.recovery.config import GatedRunConfig, RecoveryRunConfig
from src.recovery.gate import AGREES, LOW_SCORE, BoxGate, load_box_gate
from src.recovery.sources import CSRTBoxSource
from src.recovery.tracker import TrackResult
from tests.recovery.fakes import NUM_FRAMES, ScriptedBoxes, ScriptedDetector, ok
from tests.recovery.test_recovery import FIRED, NEW_BOX, START_BOX, execute, reprompts

SAM2 = Box(10, 10, 30, 30)


def scored(box: Box, score: float | None) -> TrackResult:
    return TrackResult("ok", box, box, score)


def test_gate_off_never_blocks() -> None:
    gate = BoxGate()
    assert not gate.active and gate.reason(scored(SAM2, None), SAM2) is None


def test_low_score_needs_a_score() -> None:
    gate = BoxGate(s_min=0.4)
    assert gate.reason(scored(SAM2, 0.39), None) == LOW_SCORE
    assert gate.reason(scored(SAM2, None), None) == LOW_SCORE
    assert gate.reason(scored(SAM2, 0.4), None) is None


def test_agreement_uses_iou_and_empty_sam2_mask_never_agrees() -> None:
    gate = BoxGate(a_max=0.5)
    assert gate.reason(scored(SAM2, None), SAM2) == AGREES
    assert gate.reason(scored(Box(10, 10, 30, 30), None), Box(25, 25, 45, 45)) is None
    assert gate.reason(scored(SAM2, None), None) is None


def test_low_score_is_checked_before_agreement() -> None:
    assert BoxGate(0.5, 0.5).reason(scored(SAM2, 0.1), SAM2) == LOW_SCORE


def test_gate_file_non_finite_means_off(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text("kind: box_gate\ns_min: .inf\na_max: 0.5\n")
    assert load_box_gate(path).gate() == BoxGate(None, 0.5)
    path.write_text("kind: recovery\ncooldown_frames: 1\ntracker_mode: reinit\n")
    with pytest.raises(ConfigError):
        load_box_gate(path)


def test_gated_candidate_is_logged_not_reprompted_and_starts_no_cooldown() -> None:
    from src.recovery.box_check import BoxCheck
    from src.recovery.recovery import RecoveryRun
    from src.recovery.settings import RecoverySettings
    from tests.recovery.fakes import StatefulFake
    from tests.recovery.test_recovery import Collector, read_frame

    flags = {(1, 5): FIRED, (1, 6): FIRED}
    boxes = ScriptedBoxes({(1, 5): scored(NEW_BOX, 0.1), (1, 6): scored(NEW_BOX, 0.9)})
    predictor = StatefulFake()
    run = RecoveryRun(
        predictor,
        ScriptedDetector(flags),
        boxes,
        BoxCheck(),
        RecoverySettings(12, "reinit"),
        gate=BoxGate(s_min=0.5),
    )
    events = run.run("vid", "frames", NUM_FRAMES, read_frame, {1: START_BOX}, 0, Collector())
    assert [(e.frame_index, e.action, e.reason) for e in events] == [
        (5, "gated", LOW_SCORE),
        (6, "reprompted", ""),
    ]
    assert events[0].raw_box == NEW_BOX and reprompts(predictor) == 1
    assert boxes.proposed == [(1, 5), (1, 6)]


def test_inactive_gate_equals_no_gate() -> None:
    flags = {(1, 5): FIRED, (1, 9): FIRED}
    results = {(1, 5): ok(NEW_BOX), (1, 9): ok(NEW_BOX)}
    plain, a, pa = execute(ScriptedDetector(flags), ScriptedBoxes(dict(results)))
    gated, b, pb = execute(ScriptedDetector(flags), ScriptedBoxes(dict(results)), gate=BoxGate())
    assert plain == gated and pa.calls == pb.calls
    assert all(np.array_equal(a.masks[k], b.masks[k]) for k in a.masks)


def test_cooldown_frames_call_skip_after_the_cooldown_event() -> None:
    flags = {(1, 5): FIRED, (1, 8): FIRED, (1, 9): FIRED}
    boxes = ScriptedBoxes({(1, 5): ok(NEW_BOX)})
    events, _, _ = execute(ScriptedDetector(flags), boxes)
    assert [e.action for e in events] == ["reprompted", "cooldown", "cooldown"]
    assert boxes.skipped == [(1, 8), (1, 9)]


def test_csrt_source_skip_is_a_no_op() -> None:
    source = CSRTBoxSource("reinit")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    source.skip(1, 3, frame)
    assert source.propose(1, 4, frame).status == "not_initialised"


def test_run_config_variants_and_name_rules() -> None:
    base = {"detector": "rules", "tracker_mode": "", "disabled_rules": []}
    RecoveryRunConfig(method="fix_rules_vit_reinit_tiny", box_source="vit_reinit", **base)
    RecoveryRunConfig(method="fix_rules_vit_continuous_tiny", box_source="vit_continuous", **base)
    RecoveryRunConfig(method="fix_rules_motion_tiny", box_source="motion", **base)
    GatedRunConfig(
        method="fix_gated_rules_vit_anchored_tiny",
        box_source="vit_anchored",
        gate_config="configs/recovery/g.yaml",
        **base,
    )
    with pytest.raises(ConfigError):
        RecoveryRunConfig(method="fix_rules_csrt_tiny", box_source="vit_reinit", **base)
    with pytest.raises(ConfigError):
        RecoveryRunConfig(method="fix_gated_rules_csrt_tiny", box_source="csrt", **base)
    with pytest.raises(ConfigError):
        GatedRunConfig(
            method="fix_gated_rules_csrt_tiny", box_source="csrt", gate_config="", **base
        )
    with pytest.raises(ConfigError):
        RecoveryRunConfig(
            method="fix_rules_vit_reinit_tiny",
            box_source="vit_reinit",
            detector="rules",
            tracker_mode="continuous",
            disabled_rules=[],
        )
    with pytest.raises(ConfigError):
        RecoveryRunConfig(method="fix_oracle_vit_reinit_tiny", box_source="vit_reinit", **base)
