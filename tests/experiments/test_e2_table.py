from pathlib import Path

import pandas as pd
import pytest

from experiments.e2 import table
from experiments.e2.table import E2Error
from src.eval.rows import ScoreRow, write_scores_csv
from src.recovery.events import EVENT_COLUMNS

SPLIT = "davis_val"


def _events(rows: list[tuple[str, int, str]]) -> pd.DataFrame:
    data = [(v, 1, f, "score", a, "", "", "", "", "", 0) for v, f, a in rows]
    return pd.DataFrame(data, columns=list(EVENT_COLUMNS))


def write_pair(
    results: Path,
    alone: list[ScoreRow],
    fix: list[ScoreRow],
    events: list[tuple[str, int, str]],
    frames: list[tuple[str, int, int, int, int, int, int]],
    model: str = "tiny",
    split: str = SPLIT,
) -> None:
    write_scores_csv(results / f"sam2_{model}_{split}.csv", alone)
    name = f"fix_rules_csrt_{model}_{split}"
    write_scores_csv(results / f"{name}.csv", fix)
    _events(events).to_csv(results / f"events_{name}.csv", index=False)
    pd.DataFrame(frames, columns=list(table.FRAME_COLUMNS)).to_csv(
        results / f"{name}_frames.csv", index=False
    )


def _rows(values: dict[tuple[str, int], tuple[float, float]]) -> list[ScoreRow]:
    return [ScoreRow(v, o, j, f) for (v, o), (j, f) in values.items()]


ALONE = _rows(
    {
        ("a", 1): (0.5, 0.5),
        ("a", 2): (0.7, 0.9),
        ("b", 1): (0.8, 0.8),
        ("c", 1): (0.6, 0.6),
        ("d", 1): (0.9, 0.9),
    }
)
FIX = _rows(
    {
        ("a", 1): (0.7, 0.7),  # video a: (0.7 + 0.8) / 2 = 0.75 vs (0.5 + 0.8) / 2 = 0.65
        ("a", 2): (0.7, 0.9),
        ("b", 1): (0.7, 0.7),  # worse by 0.1
        ("c", 1): (0.6, 0.6005),  # +0.00025: unchanged
        ("d", 1): (0.9, 0.9),  # unchanged
    }
)
FRAMES = [
    ("a", 1, 10, 4, 1, 3, 0),
    ("a", 2, 10, 0, 0, 0, 0),
    ("b", 1, 10, 0, 2, 0, 2),
    ("c", 1, 5, 1, 1, 0, 0),
    ("d", 1, 5, 0, 0, 0, 0),
]
EVENTS = [
    ("a", 3, "reprompted"),
    ("a", 9, "rejected"),
    ("b", 4, "reprompted"),
    ("c", 2, "cooldown"),
]


def test_counts_and_means_by_hand(tmp_path: Path) -> None:
    write_pair(tmp_path, ALONE, FIX, EVENTS, FRAMES)
    rows, events = table.build_tables(tmp_path, ["tiny"], [SPLIT])
    alone, fix = rows
    assert alone["method"] == "sam2_tiny" and fix["method"] == "fix_rules_csrt_tiny"
    assert (fix["videos"], fix["objects"]) == (4, 5)
    assert alone["J&F"] == pytest.approx(3.6 / 5)
    assert fix["J&F"] == pytest.approx(3.70025 / 5)
    assert fix["delta_JF"] == pytest.approx(0.10025 / 5)
    assert (fix["videos_improved"], fix["videos_worse"], fix["videos_unchanged"]) == (1, 1, 2)
    assert (fix["reprompts"], fix["rejected"], fix["tracker_lost"]) == (2, 1, 0)
    assert (fix["frames_recovered"], fix["frames_broken"]) == (3, 2)
    assert list(events.columns[:2]) == ["model", "split"] and len(events) == 4


def test_means_are_over_objects_and_delta_is_fix_minus_alone(tmp_path: Path) -> None:
    alone = [ScoreRow("a", 1, 0.4, 0.6), ScoreRow("a", 2, 0.8, 1.0), ScoreRow("b", 1, 0.2, 0.2)]
    fix = [ScoreRow("a", 1, 0.6, 0.8), ScoreRow("a", 2, 0.8, 1.0), ScoreRow("b", 1, 0.2, 0.2)]
    frames = [("a", 1, 4, 1, 0, 1, 0), ("a", 2, 4, 0, 0, 0, 0), ("b", 1, 4, 0, 0, 0, 0)]
    write_pair(tmp_path, alone, fix, [], frames)
    base, new = table.build_tables(tmp_path, ["tiny"], [SPLIT])[0]
    assert base["J"] == pytest.approx((0.4 + 0.8 + 0.2) / 3)
    assert base["J&F"] == pytest.approx((0.5 + 0.9 + 0.2) / 3)
    assert new["J"] == pytest.approx((0.6 + 0.8 + 0.2) / 3)
    assert new["delta_JF"] == pytest.approx(0.2 / 3)
    assert (new["videos_improved"], new["videos_worse"], new["videos_unchanged"]) == (1, 0, 1)


def test_unchanged_boundary() -> None:
    assert table.count_changes({"a": 0.5}, {"a": 0.5009}) == (0, 0, 1)
    assert table.count_changes({"a": 0.5}, {"a": 0.4991}) == (0, 0, 1)
    assert table.count_changes({"a": 0.5}, {"a": 0.502}) == (1, 0, 0)
    assert table.count_changes({"a": 0.5}, {"a": 0.498}) == (0, 1, 0)


def test_different_objects_are_refused(tmp_path: Path) -> None:
    write_pair(tmp_path, ALONE, FIX[:-1], EVENTS, FRAMES[:-1])
    with pytest.raises(E2Error, match="different videos or objects"):
        table.build_tables(tmp_path, ["tiny"], [SPLIT])


def test_missing_file_is_refused(tmp_path: Path) -> None:
    write_pair(tmp_path, ALONE, FIX, EVENTS, FRAMES)
    (tmp_path / f"events_fix_rules_csrt_tiny_{SPLIT}.csv").unlink()
    with pytest.raises(E2Error, match="missing input file"):
        table.build_tables(tmp_path, ["tiny"], [SPLIT])


def test_main_writes_both_files_with_model_and_split(tmp_path: Path) -> None:
    write_pair(tmp_path, ALONE, FIX, EVENTS, FRAMES, "tiny", "davis_val")
    write_pair(tmp_path, ALONE, FIX, [], FRAMES, "large", "davis_val")
    out, events_out = tmp_path / "e2.csv", tmp_path / "events_e2.csv"
    table.main(
        [
            "--results-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--events-out",
            str(events_out),
            "--splits",
            "davis_val",
        ]
    )
    rows = pd.read_csv(out)
    assert list(rows.columns) == list(table.TABLE_COLUMNS) and len(rows) == 4
    events = pd.read_csv(events_out)
    assert set(events["model"]) == {"tiny"} and len(events) == 4
