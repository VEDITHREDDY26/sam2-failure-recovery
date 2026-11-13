from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments.e4 import figure, sweep
from experiments.tune_rules.tune import TuneInputError
from src.detection.rules import RuleThresholds, load_rules_config
from src.detection.signals import join_signals_boxes
from tests.experiments.test_tune_rules import BOX_COLUMNS, H, W, write_world

FROZEN = RuleThresholds(-2.0, 3.0, 5.0, 0.01, 4.0)


def write_rules(path: Path, thresholds: RuleThresholds = FROZEN) -> Path:
    body = {
        "kind": "rules",
        "tiny": thresholds.__dict__,
        "large": thresholds.__dict__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body))
    return path


def tiny_case() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One object, frames 0-8, labels at 0, 4, 8: window (0,4] positive, (4,8] negative.

    Score -3 at frame 3 (inside the positive window); area 10 -> 25 -> 10 at frames 6 and 7
    (ratio 2.5, inside the negative window).
    """
    sig, boxes = [], []
    for f in range(9):
        area = 25 if f == 6 else 10
        score = -3.0 if f == 3 else 5.0
        sig.append(("v", 1, f, score, area, 5.0, 5.0, 0))
        boxes.append(("v", 1, f, H, W, 10, 10, 20, 20))
    signals = pd.DataFrame(
        sig,
        columns=["video", "object_id", "frame_index", "object_score", "area",
                 "centre_x", "centre_y", "empty"],
    )  # fmt: skip
    box = pd.DataFrame(
        boxes, columns=["video", "object_id", "frame_index", "height", "width", *BOX_COLUMNS]
    )
    labels = pd.DataFrame(
        [("v", 1, 0, 0.0, False), ("v", 1, 4, 0.0, True), ("v", 1, 8, 0.0, False)],
        columns=["video", "object_id", "frame_index", "iou", "failure"],
    )
    return join_signals_boxes(signals, box), labels


def pick(rows: list[dict[str, object]], swept: str, score: float, area: float) -> dict[str, object]:
    (row,) = [
        r
        for r in rows
        if (r["swept"], r["score_threshold"], r["area_ratio"]) == (swept, score, area)
    ]
    return row


def test_sweep_by_hand_on_a_tiny_case() -> None:
    table, labels = tiny_case()
    rows = sweep.sweep_model("tiny", table, labels, FROZEN)
    frozen = pick(rows, "both", -2.0, 3.0)
    assert (frozen["flagged"], frozen["tp"], frozen["positives"], frozen["windows"]) == (1, 1, 1, 2)
    assert frozen["precision"] == 1.0 and frozen["recall"] == 1.0 and frozen["frozen"] is True
    off = pick(rows, "score_threshold", -4.0, 3.0)
    assert (off["flagged"], off["tp"], off["recall"], off["f1"]) == (0, 0, 0.0, 0.0)
    loose = pick(rows, "area_ratio", -2.0, 2.0)
    assert (loose["flagged"], loose["tp"], loose["precision"], loose["recall"]) == (2, 1, 0.5, 1.0)
    assert not loose["frozen"]
    assert sum(1 for r in rows if r["frozen"]) == 3  # one per sweep kind


def test_frozen_value_is_added_to_the_grids() -> None:
    table, labels = tiny_case()
    odd = RuleThresholds(-2.25, 3.3, 5.0, 0.01, 4.0)
    rows = sweep.sweep_model("tiny", table, labels, odd)
    assert pick(rows, "both", -2.25, 3.3)["frozen"] is True
    assert len(rows) == (
        len(sweep.SCORE_GRID) + 1 + len(sweep.AREA_RATIO_GRID) + 1
        + (len(sweep.SCORE_GRID) + 1) * (len(sweep.AREA_RATIO_GRID) + 1)
    )  # fmt: skip


def test_run_writes_columns_and_figure(tmp_path: Path) -> None:
    root = write_world(tmp_path / "data", 3)
    rules = write_rules(tmp_path / "rules.yaml")
    out = tmp_path / "out" / "e4.csv"
    frame = sweep.run(("tiny", "large"), root, rules, out, expect_videos=3)
    assert list(pd.read_csv(out).columns) == list(sweep.COLUMNS)
    assert set(frame["model"]) == {"tiny", "large"} and set(frame["swept"]) == set(sweep.SWEEPS)
    path = figure.plot_e4(out, tmp_path / "fig")
    assert path.name == "e4_pr_curve.png" and path.stat().st_size > 0
    assert not list((tmp_path / "out").glob("*.tmp"))
    assert load_rules_config("tiny", rules) == FROZEN


def test_refuses_missing_rules_and_non_val_input(tmp_path: Path) -> None:
    root = write_world(tmp_path / "data", 3)
    with pytest.raises(TuneInputError, match="not frozen"):
        sweep.run(("tiny",), root, tmp_path / "none.yaml", tmp_path / "e4.csv", 3)
    rules = write_rules(tmp_path / "rules.yaml")
    bad = write_world(tmp_path / "sav_test_data", 3)
    with pytest.raises(TuneInputError, match="only SA-V val"):
        sweep.run(("tiny",), bad, rules, tmp_path / "e4.csv", 3)
    with pytest.raises(TuneInputError, match="expected 155"):
        sweep.run(("tiny",), root, rules, tmp_path / "e4.csv")
    assert not (tmp_path / "e4.csv").exists()
    assert (
        sweep.main(
            ["--data-dir", str(bad), "--rules", str(rules), "--out", str(tmp_path / "x.csv")]
        )
        == 1
    )
