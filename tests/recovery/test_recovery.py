from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from src.common.boxes import Box
from src.recovery.box_check import BoxCheck
from src.recovery.events import EVENT_COLUMNS, Event
from src.recovery.recovery import RecoveryRun
from src.recovery.settings import RecoverySettings
from src.recovery.sources import CSRTBoxSource
from src.recovery.tracker import TrackResult
from tests.recovery.fakes import (
    NUM_FRAMES,
    H,
    ScriptedBoxes,
    ScriptedDetector,
    StatefulFake,
    W,
    expected_mask,
    ok,
)

START_BOX = Box(10, 20, 22, 32)
NEW_BOX = Box(30, 20, 42, 32)
OTHER_BOX = Box(5, 40, 17, 52)
LATER_BOX = Box(46, 22, 58, 34)
FIRED = ("low_score",)


class Collector:
    def __init__(self) -> None:
        self.frames: list[int] = []
        self.masks: dict[tuple[int, int], npt.NDArray[np.bool_]] = {}
        self.scores: dict[int, dict[int, float]] = {}

    def __call__(self, index: int, arrays: dict[int, Any], scores: dict[int, float]) -> None:
        self.frames.append(index)
        self.scores[index] = scores
        for obj, logits in arrays.items():
            self.masks[(obj, index)] = logits > 0


def read_frame(index: int) -> npt.NDArray[np.uint8]:
    return np.zeros((H, W, 3), dtype=np.uint8)


def execute(
    detector: ScriptedDetector,
    boxes: ScriptedBoxes | CSRTBoxSource,
    prompts: dict[int, Box] | None = None,
    cooldown: int = 12,
    start: int = 0,
) -> tuple[list[Event], Collector, StatefulFake]:
    predictor = StatefulFake()
    run = RecoveryRun(predictor, detector, boxes, BoxCheck(), RecoverySettings(cooldown, "reinit"))
    sink = Collector()
    events = run.run(
        "vid", "frames", NUM_FRAMES, read_frame, prompts or {1: START_BOX}, start, sink
    )
    return events, sink, predictor


def reprompts(predictor: StatefulFake) -> int:
    return sum(1 for c in predictor.calls if c[0] == "reset")


def test_no_flag_emits_every_frame_once_in_order_and_never_restarts() -> None:
    events, sink, predictor = execute(ScriptedDetector(), ScriptedBoxes())
    assert events == []
    assert sink.frames == list(range(NUM_FRAMES))
    assert predictor.calls == [("init",), ("add", 0, 1, (10, 20, 22, 32)), ("propagate", 0, None)]
    assert set(sink.scores[3]) == {1}


def test_frames_before_t_unchanged_and_from_t_regenerated() -> None:
    _, base, _ = execute(ScriptedDetector(), ScriptedBoxes())
    detector = ScriptedDetector({(1, 5): FIRED})
    events, sink, predictor = execute(detector, ScriptedBoxes({(1, 5): ok(NEW_BOX)}))
    assert sink.frames == list(range(NUM_FRAMES))
    for f in range(5):
        assert np.array_equal(sink.masks[(1, f)], base.masks[(1, f)])
    for f in range(5, NUM_FRAMES):
        assert np.array_equal(sink.masks[(1, f)], expected_mask(NEW_BOX, 5, f))
        assert not np.array_equal(sink.masks[(1, f)], base.masks[(1, f)])
    assert [e.action for e in events] == ["reprompted"]
    assert events[0].restart_frames == 5 and events[0].raw_box == NEW_BOX


def test_call_sequence_follows_the_contract() -> None:
    execute_args = (ScriptedDetector({(1, 5): FIRED}), ScriptedBoxes({(1, 5): ok(NEW_BOX)}))
    _, _, predictor = execute(*execute_args)
    assert predictor.calls == [
        ("init",),
        ("add", 0, 1, (10, 20, 22, 32)),
        ("propagate", 0, None),
        ("reset",),
        ("add", 0, 1, (10, 20, 22, 32)),
        ("propagate", 0, 4),
        ("add", 5, 1, (30, 20, 42, 32)),
        ("propagate", 5, None),
    ]


def test_earlier_recovery_is_replayed_at_its_own_frame() -> None:
    detector = ScriptedDetector({(1, 5): FIRED, (1, 15): FIRED})
    boxes = ScriptedBoxes({(1, 5): ok(NEW_BOX), (1, 15): ok(LATER_BOX)})
    events, sink, predictor = execute(detector, boxes, cooldown=10)
    assert predictor.calls[8:] == [
        ("reset",),
        ("add", 0, 1, (10, 20, 22, 32)),
        ("add", 5, 1, (30, 20, 42, 32)),
        ("propagate", 0, 14),
        ("add", 15, 1, (46, 22, 58, 34)),
        ("propagate", 15, None),
    ]
    assert [e.restart_frames for e in events] == [5, 15]
    for f in range(5, 15):
        assert np.array_equal(sink.masks[(1, f)], expected_mask(NEW_BOX, 5, f))
    assert np.array_equal(sink.masks[(1, 20)], expected_mask(LATER_BOX, 15, 20))


def test_rejected_box_makes_no_restart_and_records_the_reason() -> None:
    tiny = Box(30, 20, 32, 22)
    detector = ScriptedDetector({(1, 5): FIRED})
    events, sink, predictor = execute(detector, ScriptedBoxes({(1, 5): ok(tiny)}))
    assert reprompts(predictor) == 0
    assert [(e.action, e.reason, e.fired) for e in events] == [("rejected", "too_small", FIRED)]
    assert events[0].raw_box == tiny
    assert sink.frames == list(range(NUM_FRAMES))


def test_tracker_lost_is_an_event_without_restart() -> None:
    detector = ScriptedDetector({(1, 5): FIRED})
    lost = TrackResult("lost", None, None)
    events, _, predictor = execute(detector, ScriptedBoxes({(1, 5): lost}))
    assert reprompts(predictor) == 0
    assert [(e.action, e.reason, e.raw_box) for e in events] == [("tracker_lost", "lost", None)]


def test_cooldown_blocks_repeat_reprompts_and_allows_one_after() -> None:
    flags = {(1, 5): FIRED, (1, 16): FIRED, (1, 17): FIRED}
    results = {(1, 5): ok(NEW_BOX), (1, 17): ok(NEW_BOX)}
    events, _, predictor = execute(ScriptedDetector(flags), ScriptedBoxes(results), cooldown=12)
    assert [(e.frame_index, e.action) for e in events] == [
        (5, "reprompted"),
        (16, "cooldown"),
        (17, "reprompted"),
    ]
    assert reprompts(predictor) == 2


def test_two_objects_flagged_together_share_one_restart() -> None:
    prompts = {1: START_BOX, 2: OTHER_BOX}
    detector = ScriptedDetector({(1, 5): FIRED, (2, 5): ("area_change", "centre_jump")})
    boxes = ScriptedBoxes({(1, 5): ok(NEW_BOX), (2, 5): ok(Box(30, 40, 42, 52))})
    events, sink, predictor = execute(detector, boxes, prompts)
    assert reprompts(predictor) == 1
    assert [(e.object_id, e.action, e.fired) for e in events] == [
        (1, "reprompted", FIRED),
        (2, "reprompted", ("area_change", "centre_jump")),
    ]
    assert np.array_equal(sink.masks[(2, 9)], expected_mask(Box(30, 40, 42, 52), 5, 9))


def test_only_the_flagged_object_changes_in_a_two_object_video() -> None:
    prompts = {1: START_BOX, 2: OTHER_BOX}
    _, base, _ = execute(ScriptedDetector(), ScriptedBoxes(), prompts)
    detector = ScriptedDetector({(1, 5): FIRED})
    _, sink, _ = execute(detector, ScriptedBoxes({(1, 5): ok(NEW_BOX)}), prompts)
    for f in range(NUM_FRAMES):
        assert np.array_equal(sink.masks[(2, f)], base.masks[(2, f)])


def test_detector_and_box_source_are_restarted_from_the_recovered_frame() -> None:
    detector = ScriptedDetector({(1, 5): FIRED})
    boxes = ScriptedBoxes({(1, 5): ok(NEW_BOX)})
    execute(detector, boxes)
    assert (1, 0) in [(o, f) for o, f, _ in detector.observed]
    recovered = [m for o, f, m in detector.observed if (o, f) == (1, 5)]
    assert len(recovered) == 1 and np.array_equal(recovered[0], expected_mask(NEW_BOX, 5, 5))
    assert boxes.started == [(1, 0), (1, 5)]
    assert detector.updates.count((1, 5)) == 1 and (1, 6) in detector.updates


def test_prompt_frame_is_never_a_trigger_and_nothing_before_it_is_emitted() -> None:
    detector = ScriptedDetector({(1, 7): FIRED})
    events, sink, predictor = execute(detector, ScriptedBoxes({(1, 7): ok(NEW_BOX)}), start=2)
    assert sink.frames == list(range(2, NUM_FRAMES))
    assert (1, 2) not in detector.updates
    assert ("propagate", 2, 4) in predictor.calls
    assert events[0].restart_frames == 5


def test_module_writes_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    detector = ScriptedDetector({(1, 5): FIRED})
    execute(detector, ScriptedBoxes({(1, 5): ok(NEW_BOX)}))
    assert list(tmp_path.iterdir()) == []


def test_unusable_inputs_raise() -> None:
    run = RecoveryRun(
        StatefulFake(),
        ScriptedDetector(),
        ScriptedBoxes(),
        BoxCheck(),
        RecoverySettings(1, "reinit"),
    )
    with pytest.raises(ValueError):
        run.run("v", "f", NUM_FRAMES, read_frame, {}, 0, Collector())
    with pytest.raises(ValueError):
        run.run("v", "f", NUM_FRAMES, read_frame, {1: START_BOX}, NUM_FRAMES, Collector())


def test_event_row_matches_columns() -> None:
    row = Event("v", 1, 5, FIRED + ("area_change",), "reprompted", "", NEW_BOX, 5).as_row()
    assert len(row) == len(EVENT_COLUMNS)
    assert row[3] == "low_score+area_change" and row[6:] == (30, 20, 42, 32, 5)
    assert Event("v", 1, 5, FIRED, "cooldown").as_row()[6:10] == ("", "", "", "")


def textured_frame() -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(1)
    frame = rng.integers(190, 230, size=(H, W, 3), dtype=np.uint8)
    frame[20:32, 10:22] = (30, 60, 200)
    frame[22:26, 12:20] = (240, 240, 20)
    return frame


def test_real_csrt_source_proposes_an_accepted_box_end_to_end() -> None:
    frame = textured_frame()
    boxes = CSRTBoxSource("reinit")
    detector = ScriptedDetector({(1, 5): FIRED})
    predictor = StatefulFake()
    run = RecoveryRun(predictor, detector, boxes, BoxCheck(), RecoverySettings(12, "reinit"))
    still = Box(10, 20, 22, 32)
    prompts = {1: still}
    events = run.run("vid", "f", NUM_FRAMES, lambda i: frame, prompts, 0, Collector())
    assert [(e.action, e.frame_index) for e in events] == [("reprompted", 5)]
    assert events[0].raw_box is not None
    assert abs(events[0].raw_box.x0 - 10) <= 4 and abs(events[0].raw_box.y0 - 20) <= 4


def test_continuous_source_tracks_without_reinit() -> None:
    frame = textured_frame()
    source = CSRTBoxSource("continuous")
    mask = expected_mask(START_BOX, 0, 0)
    source.start(1, 0, frame, mask)
    source.observe(1, 1, frame, mask)
    result = source.propose(1, 2, frame)
    assert result.status == "ok"
    assert source.last_reliable_box(1) == START_BOX
    with pytest.raises(ValueError):
        CSRTBoxSource("sometimes")
