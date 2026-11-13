import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.tune_rules.export_sav_val import BOX_COLUMNS
from src.common.boxes import Box
from src.common.config import ConfigError
from src.common.signals import signals_from_masks
from src.detection.rules import (
    ALL_RULES,
    AREA_CHANGE,
    CENTRE_JUMP,
    DISAPPEARANCE,
    LOW_SCORE,
    RuleDetector,
    RulesConfig,
    RuleThresholds,
    batch_evidence,
    batch_flags,
    evidence,
    fired_rules,
    load_rules_config,
)
from src.detection.signals import FrameObs, join_signals_boxes, obs_from_mask, obs_from_row

CFG = RuleThresholds(
    score_threshold=0.0,
    area_ratio=2.0,
    jump_ratio=0.5,
    min_area_fraction=0.05,
    border_margin_px=5.0,
)
BOX = Box(40, 40, 50, 50)


def obs(area: int = 100, cx: float = 50.0, box: Box | None = BOX, score: float = 1.0) -> FrameObs:
    if area == 0:
        return FrameObs(score, 0, math.nan, math.nan, None, 100, 100)
    return FrameObs(score, area, cx, 50.0, box, 100, 100)


def fired(cur: FrameObs, prev: FrameObs | None) -> tuple[str, ...]:
    return fired_rules(evidence(cur, prev), CFG)


def test_low_score_threshold() -> None:
    assert fired(obs(score=-0.01), obs()) == (LOW_SCORE,)
    assert fired(obs(score=0.0), obs()) == ()
    assert fired(obs(score=math.nan), obs()) == ()


def test_area_change_threshold_both_directions() -> None:
    assert fired(obs(area=200), obs(area=100)) == ()
    assert fired(obs(area=201), obs(area=100)) == (AREA_CHANGE,)
    assert fired(obs(area=50), obs(area=100)) == ()
    assert fired(obs(area=49), obs(area=100)) == (AREA_CHANGE,)


def test_centre_jump_is_scale_normalised() -> None:
    prev = obs(area=100, cx=50.0)  # sqrt(100) = 10, threshold distance 5
    assert fired(obs(area=100, cx=55.0), prev) == ()
    assert fired(obs(area=100, cx=55.1), prev) == (CENTRE_JUMP,)
    assert fired(obs(area=100, cx=44.9), prev) == (CENTRE_JUMP,)
    big = obs(area=400, cx=50.0)  # sqrt = 20: a 10 px move is 0.5, not a jump
    assert fired(obs(area=400, cx=60.0), big) == ()
    assert fired(obs(area=400, cx=60.5), big) == (CENTRE_JUMP,)


def test_disappearance_area_fraction() -> None:
    gone = obs(area=0)
    assert fired(gone, obs(area=500, box=Box(10, 10, 60, 20))) == (DISAPPEARANCE,)
    assert fired(gone, obs(area=499, box=Box(10, 10, 60, 20))) == ()


def test_border_contact_suppresses_disappearance() -> None:
    gone = obs(area=0)
    big = 600
    assert fired(gone, obs(area=big, box=Box(5, 5, 95, 95))) == (DISAPPEARANCE,)  # margin == 5
    assert fired(gone, obs(area=big, box=Box(4, 5, 95, 95))) == ()
    assert fired(gone, obs(area=big, box=Box(5, 4, 95, 95))) == ()
    assert fired(gone, obs(area=big, box=Box(5, 5, 96, 95))) == ()
    assert fired(gone, obs(area=big, box=Box(5, 5, 95, 96))) == ()
    assert fired(gone, obs(area=big, box=Box(0, 5, 95, 95))) == ()


def test_reappearance_only_low_score_can_fire() -> None:
    assert fired(obs(area=900, cx=10.0), obs(area=0)) == ()
    assert fired(obs(area=900, cx=10.0, score=-2.0), obs(area=0)) == (LOW_SCORE,)


def test_empty_to_empty_does_not_fire() -> None:
    assert fired(obs(area=0), obs(area=0)) == ()


def test_empty_history_only_low_score() -> None:
    assert fired_rules(evidence(obs(area=900, score=-1.0), None), CFG) == (LOW_SCORE,)
    assert fired_rules(evidence(obs(area=900, score=1.0), None), CFG) == ()
    assert fired_rules(evidence(obs(area=0, score=1.0), None), CFG) == ()


def test_first_frame_never_flags() -> None:
    det = RuleDetector(CFG)
    first = det.update(obs(area=900, score=-9.0))
    assert not first.flag and first.fired == ()
    assert det.update(obs(area=900, score=-9.0)).fired == (LOW_SCORE,)


def test_streaming_flag_is_or_of_enabled_rules() -> None:
    cur, prev = obs(area=300, cx=70.0, score=-1.0), obs(area=100, cx=50.0)
    full = RuleDetector(CFG)
    full.update(prev)
    assert full.update(cur).fired == (LOW_SCORE, AREA_CHANGE, CENTRE_JUMP)
    only = RuleDetector(CFG, enabled=(AREA_CHANGE,))
    only.update(prev)
    d = only.update(cur)
    assert d.flag and d.fired == (AREA_CHANGE,)
    none = RuleDetector(CFG, enabled=())
    none.update(prev)
    assert not none.update(cur).flag
    with pytest.raises(ValueError):
        RuleDetector(CFG, enabled=("nope",))


def _random_masks(
    rng: np.random.Generator, n: int, h: int = 30, w: int = 40
) -> tuple[list[np.ndarray], list[float]]:
    masks, scores = [], []
    for _ in range(n):
        m = np.zeros((h, w), dtype=bool)
        if rng.random() > 0.2:
            x0, y0 = rng.integers(0, w - 2), rng.integers(0, h - 2)
            m[y0 : y0 + rng.integers(1, h), x0 : x0 + rng.integers(1, w)] = True
        masks.append(m)
        scores.append(float("nan") if rng.random() < 0.1 else float(rng.normal()))
    return masks, scores


def _table(seed: int, objects: int = 3, frames: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigs, boxes = [], []
    for obj in range(objects):
        masks, scores = _random_masks(rng, frames)
        sigs.append(signals_from_masks("v", obj, masks, scores))
        for i, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            c = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if ys.size else (None,) * 4
            boxes.append(("v", obj, i, *m.shape, *c))
    b = pd.DataFrame(boxes, columns=list(BOX_COLUMNS)).astype({c: "Int64" for c in BOX_COLUMNS[3:]})
    return join_signals_boxes(pd.concat(sigs, ignore_index=True), b)


def _stream(
    table: pd.DataFrame, cfg: RuleThresholds, enabled: tuple[str, ...] = ALL_RULES
) -> list[bool]:
    flags: list[bool] = []
    det: RuleDetector | None = None
    last = None
    for _, row in table.iterrows():
        if last != row["object_id"]:
            det, last = RuleDetector(cfg, enabled), row["object_id"]
        assert det is not None
        flags.append(det.update(obs_from_row(row)).flag)
    return flags


@pytest.mark.parametrize("seed", range(5))
def test_batch_equals_streaming(seed: int) -> None:
    cfg = RuleThresholds(-0.3, 1.5, 0.3, 0.02, 3.0)
    table = _table(seed)
    for enabled in (ALL_RULES, (LOW_SCORE,), (DISAPPEARANCE, CENTRE_JUMP), ()):
        batch = batch_flags(table, cfg, enabled)
        assert list(batch["flag"]) == _stream(table, cfg, enabled)
    assert batch_flags(table, cfg)["flag"].any()


def test_batch_first_rows_never_flag() -> None:
    table = _table(1)
    flags = batch_flags(table, CFG)
    assert not flags.loc[table["frame_index"] == 0, "flag"].any()


def test_altering_future_frames_does_not_change_past() -> None:
    table = _table(2)
    cut = 20
    changed = table.copy()
    rng = np.random.default_rng(0)
    late = changed["frame_index"] > cut
    changed.loc[late, "object_score"] = rng.normal(size=int(late.sum()))
    changed.loc[late, "area"] = rng.integers(1, 50, size=int(late.sum()))
    changed.loc[late, "centre_x"] = 1.0
    changed.loc[late, "centre_y"] = 1.0
    for col, val in zip(("box_x0", "box_y0", "box_x1", "box_y1"), (0, 0, 3, 3), strict=True):
        changed.loc[late, col] = val
    before, after = _stream(table, CFG), _stream(changed, CFG)
    keep = (table["frame_index"] <= cut).to_numpy()
    assert [f for f, k in zip(before, keep, strict=True) if k] == [
        f for f, k in zip(after, keep, strict=True) if k
    ]
    b0, b1 = batch_flags(table, CFG), batch_flags(changed, CFG)
    pd.testing.assert_frame_equal(b0[keep], b1[keep])


def test_online_and_offline_observations_give_same_flags() -> None:
    rng = np.random.default_rng(3)
    masks, scores = _random_masks(rng, 40)
    det_on = RuleDetector(CFG)
    online = [det_on.update(obs_from_mask(m, s)).flag for m, s in zip(masks, scores, strict=True)]
    sig = signals_from_masks("v", 0, masks, scores)
    rows = []
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        c = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if ys.size else (None,) * 4
        rows.append(("v", 0, i, *m.shape, *c))
    b = pd.DataFrame(rows, columns=list(BOX_COLUMNS)).astype({c: "Int64" for c in BOX_COLUMNS[3:]})
    offline = list(batch_flags(join_signals_boxes(sig, b), CFG)["flag"])
    assert online == offline


def _write(path: Path, tiny: str, large: str) -> Path:
    path.write_text(f"kind: rules\ntiny: {tiny}\nlarge: {large}\n")
    return path


GOOD = (
    "{score_threshold: -1.0, area_ratio: 2.0, jump_ratio: 0.5,"
    " min_area_fraction: 0.05, border_margin_px: 8}"
)


def test_config_loads_per_model(tmp_path: Path) -> None:
    other = GOOD.replace("-1.0", "-2.5")
    path = _write(tmp_path / "rules.yaml", GOOD, other)
    assert load_rules_config("tiny", path).score_threshold == -1.0
    assert load_rules_config("large", path).score_threshold == -2.5
    assert load_rules_config("large", path).border_margin_px == 8.0
    with pytest.raises(ConfigError):
        load_rules_config("huge", path)


@pytest.mark.parametrize(
    "bad",
    [
        GOOD.replace("area_ratio: 2.0", "area_ratio: 1.0"),
        GOOD.replace("jump_ratio: 0.5", "jump_ratio: 0"),
        GOOD.replace("min_area_fraction: 0.05", "min_area_fraction: 0"),
        GOOD.replace("min_area_fraction: 0.05", "min_area_fraction: 1.5"),
        GOOD.replace("border_margin_px: 8", "border_margin_px: -1"),
        GOOD.replace("}", ", extra: 1}"),
        GOOD.replace("score_threshold: -1.0, ", ""),
        GOOD.replace("-1.0", ".nan"),
    ],
)
def test_config_rejects_bad_values(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ConfigError):
        load_rules_config("tiny", _write(tmp_path / "rules.yaml", bad, GOOD))


def test_config_is_frozen() -> None:
    cfg = RulesConfig(tiny=CFG, large=CFG)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.tiny = CFG  # type: ignore[misc]


def _rect() -> np.ndarray:
    m = np.zeros((30, 40), dtype=bool)
    m[5:15, 5:15] = True
    return m


def test_overlapped_frame_never_fires_disappearance_and_batch_equals_streaming() -> None:
    cfg = RuleThresholds(-100.0, 1000.0, 1000.0, 0.01, 0.0)
    masks = [_rect()] * 3 + [np.zeros((30, 40), dtype=bool)] * 2
    sig = signals_from_masks("v", 0, masks, [1.0] * 5)
    rows = []
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        c = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if ys.size else (None,) * 4
        rows.append(("v", 0, i, *m.shape, *c))
    b = pd.DataFrame(rows, columns=list(BOX_COLUMNS)).astype({c: "Int64" for c in BOX_COLUMNS[3:]})
    plain = join_signals_boxes(sig, b)
    assert list(batch_flags(plain, cfg)["flag"]) == [False, False, False, True, False]
    hidden = b.copy()
    hidden.loc[2, list(BOX_COLUMNS[5:])] = pd.NA
    table = join_signals_boxes(sig, hidden, allow_overlapped=True)
    batch = batch_flags(table, cfg)
    assert list(batch["flag"]) == [False] * 5
    assert list(batch["flag"]) == _stream(table, cfg)
    ev = batch_evidence(table)
    assert np.isnan(ev.loc[3, "prev_margin"]) and ev.loc[3, "prev_area_fraction"] > 0


def test_overlapped_frame_leaves_other_rules_unchanged() -> None:
    cfg = RuleThresholds(-0.3, 1.5, 0.3, 0.02, 3.0)
    table = _table(4)
    sig = table[
        ["video", "object_id", "frame_index", "object_score", "area", "centre_x", "centre_y"]
    ]
    boxes = table[["video", "object_id", "frame_index", "height", "width", *BOX_COLUMNS[5:]]].copy()
    hide = (table["area"] > 0).to_numpy().nonzero()[0][::3]
    boxes.loc[boxes.index[hide], list(BOX_COLUMNS[5:])] = np.nan
    other = join_signals_boxes(sig, boxes, allow_overlapped=True)
    a, c = batch_flags(table, cfg), batch_flags(other, cfg)
    for rule in (LOW_SCORE, AREA_CHANGE, CENTRE_JUMP):
        assert list(a[rule]) == list(c[rule])
    assert list(c["flag"]) == _stream(other, cfg)
