import ast
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from experiments.tune_rules.export_sav_val import BOX_COLUMNS
from src.common.boxes import Box
from src.common.signals import signals_from_masks
from src.detection.online import OnlineRuleDetector
from src.detection.rules import (
    ALL_RULES,
    CENTRE_JUMP,
    DISAPPEARANCE,
    RuleThresholds,
    batch_flags,
)
from src.detection.signals import join_signals_boxes
from src.recovery.box_check import BoxCheck
from src.recovery.recovery import RecoveryRun
from src.recovery.settings import RecoverySettings
from src.recovery.sources import Detector
from src.sam2_runner.predictor import REPO_ROOT
from tests.detection.test_label_isolation import _modules
from tests.recovery.fakes import NUM_FRAMES, H, ScriptedBoxes, StatefulFake, W, ok
from tests.recovery.test_recovery import Collector, read_frame

Mask = npt.NDArray[np.bool_]
CFG = RuleThresholds(-100.0, 100.0, 100.0, 0.01, 2.0)
JUMPY = RuleThresholds(-100.0, 100.0, 0.5, 0.01, 2.0)
START = Box(10, 20, 22, 32)
RECOVERED = Box(22, 20, 34, 32)
FAIL_FRAME = 8
SHAPE = (H, W)


class FailingFake(StatefulFake):
    """Object 1 vanishes from FAIL_FRAME on until it is re-prompted."""

    def _mask(self, state: Any, obj_id: int, frame: int) -> Mask:
        if obj_id == 1 and len(state["prompts"][obj_id]) == 1 and frame >= FAIL_FRAME:
            return np.zeros((H, W), dtype=bool)
        return super()._mask(state, obj_id, frame)


def drive(predictor: StatefulFake, boxes: ScriptedBoxes) -> tuple[list[Any], Collector]:
    run = RecoveryRun(
        predictor, OnlineRuleDetector(CFG), boxes, BoxCheck(), RecoverySettings(12, "reinit")
    )
    sink = Collector()
    events = run.run("vid", "frames", NUM_FRAMES, read_frame, {1: START}, 0, sink)
    return events, sink


def square(x: int, size: int = 12) -> Mask:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[20 : 20 + size, x : x + size] = True
    return mask


def test_adapter_satisfies_the_recovery_protocol() -> None:
    detector: Detector = OnlineRuleDetector(CFG)
    assert detector.update(1, 0, square(10), 1.0, SHAPE).flag is False


def test_injected_failure_triggers_exactly_one_recovery() -> None:
    _, clean = drive(StatefulFake(), ScriptedBoxes())
    boxes = ScriptedBoxes({(1, FAIL_FRAME): ok(RECOVERED)})
    predictor = FailingFake()
    events, sink = drive(predictor, boxes)
    assert [(e.action, e.frame_index, e.fired) for e in events] == [
        ("reprompted", FAIL_FRAME, (DISAPPEARANCE,))
    ]
    assert sum(1 for c in predictor.calls if c[0] == "reset") == 1
    for f in range(FAIL_FRAME):
        assert np.array_equal(sink.masks[(1, f)], clean.masks[(1, f)])
    assert all(sink.masks[(1, f)].any() for f in range(FAIL_FRAME, NUM_FRAMES))


def test_clean_video_has_no_events() -> None:
    events, sink = drive(StatefulFake(), ScriptedBoxes())
    assert events == []
    assert sink.frames == list(range(NUM_FRAMES))


def test_observe_sets_history_without_flagging() -> None:
    adapter = OnlineRuleDetector(JUMPY, enabled=(CENTRE_JUMP,))
    adapter.update(1, 0, square(10), 1.0, SHAPE)
    adapter.observe(1, 1, square(40), 1.0, SHAPE)
    assert not adapter.update(1, 2, square(41), 1.0, SHAPE).flag
    control = OnlineRuleDetector(JUMPY, enabled=(CENTRE_JUMP,))
    control.update(1, 0, square(10), 1.0, SHAPE)
    assert control.update(1, 1, square(40), 1.0, SHAPE).flag


def test_objects_have_separate_history_and_reset_clears_it() -> None:
    adapter = OnlineRuleDetector(JUMPY, enabled=(CENTRE_JUMP,))
    adapter.update(1, 0, square(10), 1.0, SHAPE)
    assert not adapter.update(2, 1, square(40), 1.0, SHAPE).flag
    adapter.reset()
    assert not adapter.update(1, 5, square(40), 1.0, SHAPE).flag


def test_wrong_mask_shape_is_refused() -> None:
    with pytest.raises(ValueError):
        OnlineRuleDetector(CFG).update(1, 0, square(10), 1.0, (10, 10))


def test_unknown_rule_is_refused() -> None:
    with pytest.raises(ValueError):
        OnlineRuleDetector(CFG, enabled=("nope",))


def random_sequences(seed: int) -> tuple[pd.DataFrame, dict[int, list[Mask]], list[list[float]]]:
    rng = np.random.default_rng(seed)
    h, w = 30, 40
    sigs, boxes, masks_by_obj, scores_by_obj = [], [], {}, []
    for obj in range(3):
        masks, scores = [], []
        for _ in range(40):
            m = np.zeros((h, w), dtype=bool)
            if rng.random() > 0.2:
                x0, y0 = rng.integers(0, w - 2), rng.integers(0, h - 2)
                m[y0 : y0 + rng.integers(1, h), x0 : x0 + rng.integers(1, w)] = True
            masks.append(m)
            scores.append(float(rng.normal()))
        sigs.append(signals_from_masks("v", obj, masks, scores))
        for i, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            c = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if ys.size else (None,) * 4
            boxes.append(("v", obj, i, h, w, *c))
        masks_by_obj[obj] = masks
        scores_by_obj.append(scores)
    frame = pd.DataFrame(boxes, columns=list(BOX_COLUMNS))
    frame = frame.astype({c: "Int64" for c in BOX_COLUMNS[3:]})
    return (
        join_signals_boxes(pd.concat(sigs, ignore_index=True), frame),
        masks_by_obj,
        scores_by_obj,
    )


@pytest.mark.parametrize("seed", range(3))
def test_streaming_adapter_equals_batch_flags(seed: int) -> None:
    cfg = RuleThresholds(-0.3, 1.5, 0.3, 0.02, 3.0)
    table, masks, scores = random_sequences(seed)
    batch = batch_flags(table, cfg, ALL_RULES)
    adapter = OnlineRuleDetector(cfg)
    flags: list[bool] = []
    for obj in sorted(masks):
        for i, mask in enumerate(masks[obj]):
            decision = adapter.update(obj, i, mask, scores[obj][i], mask.shape)
            flags.append(decision.flag)
    assert flags == list(batch["flag"])
    assert any(flags)


def test_online_module_imports_no_labels_and_no_oracle() -> None:
    path = REPO_ROOT / "src" / "detection" / "online.py"
    found = _modules(ast.parse(path.read_text()), "src.detection")
    bad = [m for m in found if m.startswith(("src.detection.labels", "src.oracle"))]
    assert bad == []
