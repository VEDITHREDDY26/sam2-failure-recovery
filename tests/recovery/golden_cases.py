"""Fixed recovery scenarios on the fake predictor, summarised as plain data for golden checks."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.common.boxes import Box
from src.recovery.box_check import BoxCheck
from src.recovery.recovery import RecoveryRun
from src.recovery.settings import RecoverySettings
from src.recovery.tracker import TrackResult
from tests.recovery.fakes import NUM_FRAMES, H, ScriptedBoxes, ScriptedDetector, StatefulFake, W, ok

START = Box(10, 20, 22, 32)
OTHER = Box(5, 40, 17, 52)
NEW = Box(30, 20, 42, 32)
NEW2 = Box(8, 38, 20, 50)
LATER = Box(46, 22, 58, 34)
FIRED = ("low_score",)


def _frame(index: int) -> Any:
    return np.zeros((H, W, 3), dtype=np.uint8)


def _summary(run: RecoveryRun, predictor: StatefulFake, events: list[Any], seen: dict) -> dict:
    digest = hashlib.sha256()
    for key in sorted(seen, key=repr):
        digest.update(repr(key).encode())
        digest.update(seen[key].tobytes())
    return {
        "events": [list(map(str, e.as_row())) for e in events],
        "calls": [list(map(str, c)) for c in predictor.calls],
        "masks": digest.hexdigest(),
    }


def _case(
    flags: dict, results: dict, prompts: dict, cooldown: int = 12, extra: dict | None = None
) -> dict:
    predictor = StatefulFake()
    detector, boxes = ScriptedDetector(flags), ScriptedBoxes(results)
    run = RecoveryRun(predictor, detector, boxes, BoxCheck(), RecoverySettings(cooldown, "reinit"))
    seen: dict = {}

    def sink(index: int, arrays: dict, scores: dict) -> None:
        for obj, logits in arrays.items():
            seen[(obj, index)] = (logits > 0).astype(np.uint8)
            seen[("s", obj, index)] = np.array([scores[obj]])

    events = run.run("vid", "frames", NUM_FRAMES, _frame, prompts, 0, sink)
    return _summary(run, predictor, events, seen)


def run_cases() -> dict[str, dict]:
    lost = TrackResult("lost", None, None)
    return {
        "none": _case({}, {}, {1: START}),
        "one": _case({(1, 5): FIRED}, {(1, 5): ok(NEW)}, {1: START}),
        "two_frames": _case(
            {(1, 5): FIRED, (1, 15): FIRED},
            {(1, 5): ok(NEW), (1, 15): ok(LATER)},
            {1: START},
            cooldown=10,
        ),
        "cooldown": _case(
            {(1, 5): FIRED, (1, 8): FIRED}, {(1, 5): ok(NEW), (1, 8): ok(LATER)}, {1: START}
        ),
        "two_objects": _case(
            {(1, 6): FIRED, (2, 6): ("centre_jump",)},
            {(1, 6): ok(NEW), (2, 6): ok(NEW2)},
            {1: START, 2: OTHER},
        ),
        "lost": _case({(1, 5): FIRED}, {(1, 5): lost}, {1: START}),
    }
