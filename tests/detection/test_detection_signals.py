import math

import numpy as np
import pandas as pd
import pytest

from experiments.tune_rules.export_sav_val import BOX_COLUMNS
from src.common.boxes import Box
from src.common.signals import signals_from_masks
from src.detection.signals import (
    FrameObs,
    count_overlapped,
    join_signals_boxes,
    obs_from_mask,
    obs_from_row,
)


def _mask(x0: int, y0: int, x1: int, y1: int, h: int = 12, w: int = 20) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _tables(masks: list[np.ndarray], scores: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sig = signals_from_masks("v", 1, masks, scores)
    rows = []
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        corners = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if ys.size else (None,) * 4
        rows.append(("v", 1, i, m.shape[0], m.shape[1], *corners))
    boxes = pd.DataFrame(rows, columns=list(BOX_COLUMNS)).astype(
        {c: "Int64" for c in BOX_COLUMNS[3:]}
    )
    return sig, boxes


def test_online_observation_hand_values() -> None:
    obs = obs_from_mask(_mask(2, 3, 6, 5), -1.5)  # 4 wide, 2 high: area 8
    assert obs.area == 8
    assert obs.centre_x == pytest.approx(3.5)
    assert obs.centre_y == pytest.approx(3.5)
    assert obs.box == Box(2, 3, 6, 5)
    assert (obs.height, obs.width, obs.object_score) == (12, 20, -1.5)


def test_empty_observation() -> None:
    obs = obs_from_mask(np.zeros((4, 5), dtype=bool), 0.25)
    assert obs.empty and obs.box is None and math.isnan(obs.centre_x)


def test_box_must_match_emptiness() -> None:
    with pytest.raises(ValueError):
        FrameObs(0.0, 4, 1.0, 1.0, None, 10, 10)


def test_offline_equals_online() -> None:
    masks = [_mask(1, 1, 5, 4), _mask(0, 0, 0, 0), _mask(10, 2, 19, 12)]
    scores = [2.0, -3.0, float("nan")]
    sig, boxes = _tables(masks, scores)
    table = join_signals_boxes(sig, boxes)
    for i, (m, s) in enumerate(zip(masks, scores, strict=True)):
        online, offline = obs_from_mask(m, s), obs_from_row(table.iloc[i])
        assert offline.area == online.area and offline.box == online.box
        assert offline.height == online.height and offline.width == online.width
        assert offline.centre_x == online.centre_x or (
            math.isnan(offline.centre_x) and math.isnan(online.centre_x)
        )
        assert offline.object_score == online.object_score or (
            math.isnan(offline.object_score) and math.isnan(online.object_score)
        )


def test_join_drops_frames_before_prompt() -> None:
    masks = [_mask(1, 1, 5, 4)] * 3
    sig, boxes = _tables(masks, [1.0, 1.0, 1.0])
    table = join_signals_boxes(sig, boxes.iloc[1:])
    assert list(table["frame_index"]) == [1, 2]


def test_join_refuses_duplicates_missing_and_mismatch() -> None:
    sig, boxes = _tables([_mask(1, 1, 5, 4), _mask(1, 1, 5, 4)], [1.0, 1.0])
    with pytest.raises(ValueError, match="duplicate"):
        join_signals_boxes(sig, pd.concat([boxes, boxes.iloc[:1]]))
    with pytest.raises(ValueError, match="duplicate"):
        join_signals_boxes(pd.concat([sig, sig.iloc[:1]]), boxes)
    with pytest.raises(ValueError, match="without a matching"):
        join_signals_boxes(sig.iloc[:1], boxes)
    bad = boxes.copy()
    bad.loc[1, list(BOX_COLUMNS[5:])] = pd.NA
    with pytest.raises(ValueError, match="non-zero"):
        join_signals_boxes(sig, bad)


def test_join_refuses_gaps() -> None:
    sig, boxes = _tables([_mask(1, 1, 5, 4)] * 3, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="consecutive"):
        join_signals_boxes(sig, boxes.iloc[[0, 2]])


def _overlapped_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    sig, boxes = _tables([_mask(1, 1, 5, 4)] * 3, [1.0, 1.0, 1.0])
    boxes.loc[1, list(BOX_COLUMNS[5:])] = pd.NA
    return sig, boxes


def test_overlapped_refused_by_default_and_accepted_on_request() -> None:
    sig, boxes = _overlapped_tables()
    with pytest.raises(ValueError, match="non-zero"):
        join_signals_boxes(sig, boxes)
    table = join_signals_boxes(sig, boxes, allow_overlapped=True)
    assert count_overlapped(table) == 1
    obs = obs_from_row(table.iloc[1])
    assert obs.area == 12 and obs.box is None and obs.box_unknown
    assert not obs_from_row(table.iloc[0]).box_unknown


def test_overlapped_option_still_refuses_other_mismatches() -> None:
    sig, boxes = _tables([_mask(1, 1, 5, 4), _mask(0, 0, 0, 0), _mask(1, 1, 5, 4)], [1] * 3)
    full = boxes.copy()
    full.loc[1, list(BOX_COLUMNS[5:])] = [1, 1, 5, 4]
    with pytest.raises(ValueError, match="non-zero"):
        join_signals_boxes(sig, full, allow_overlapped=True)
    partial = boxes.copy()
    partial.loc[0, BOX_COLUMNS[5]] = pd.NA
    with pytest.raises(ValueError, match="partly"):
        join_signals_boxes(sig, partial, allow_overlapped=True)
    with pytest.raises(ValueError, match="consecutive"):
        join_signals_boxes(sig, boxes.iloc[[0, 2]], allow_overlapped=True)
    assert count_overlapped(join_signals_boxes(sig, boxes, allow_overlapped=True)) == 0
