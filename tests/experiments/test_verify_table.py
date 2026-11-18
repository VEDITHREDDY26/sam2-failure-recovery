"""Tests for the verification results table: three rows with counts and deltas, documented
columns, and refusal of missing or mismatched inputs.
"""

from pathlib import Path

import pandas as pd
import pytest

from experiments.e2.table import FRAME_COLUMNS, E2Error
from experiments.verify import table
from src.eval.rows import ScoreRow, write_scores_csv
from src.recovery.events import EVENT_COLUMNS

SPLIT = "davis_val"


def rows(values: dict[str, float]) -> list[ScoreRow]:
    return [ScoreRow(v, 1, j, j) for v, j in values.items()]


def write_method(
    results: Path, method: str, scores: dict[str, float], actions: list[str], model: str = "tiny"
) -> None:
    name = f"{method}_{SPLIT}"
    write_scores_csv(results / f"{name}.csv", rows(scores))
    events = [("a", 1, i + 1, "score", act, "", "", "", "", "", 0) for i, act in enumerate(actions)]
    pd.DataFrame(events, columns=list(EVENT_COLUMNS)).to_csv(
        results / f"events_{name}.csv", index=False
    )
    frames = [(v, 1, 10, 2, 1, 1, 0) for v in scores]
    pd.DataFrame(frames, columns=list(FRAME_COLUMNS)).to_csv(
        results / f"{name}_frames.csv", index=False
    )


def build(tmp_path: Path, verified: dict[str, float] | None = None) -> None:
    write_scores_csv(tmp_path / f"sam2_tiny_{SPLIT}.csv", rows({"a": 0.6, "b": 0.8}))
    write_method(
        tmp_path, "fix_rules_csrt_tiny", {"a": 0.4, "b": 0.8}, ["reprompted", "reprompted"]
    )
    write_method(
        tmp_path,
        "fix_verified_rules_csrt_tiny",
        verified or {"a": 0.6, "b": 0.9},
        ["reprompted", "verify_rejected", "verify_rejected", "rejected"],
    )


def test_three_rows_with_counts_and_deltas(tmp_path: Path) -> None:
    build(tmp_path)
    got, events = table.build_tables(tmp_path, ["tiny"], [SPLIT])
    alone, e2, verified = got
    assert [r["method"] for r in got] == [
        "sam2_tiny",
        "fix_rules_csrt_tiny",
        "fix_verified_rules_csrt_tiny",
    ]
    assert e2["delta_JF"] == pytest.approx(-0.1) and verified["delta_JF"] == pytest.approx(0.05)
    assert verified["delta_JF_vs_e2"] == pytest.approx(0.15) and e2["delta_JF_vs_e2"] == ""
    assert (e2["videos_worse"], e2["videos_improved"], e2["videos_unchanged"]) == (1, 0, 1)
    assert (verified["videos_worse"], verified["videos_improved"]) == (0, 1)
    assert (verified["reprompts"], verified["verify_rejected"], verified["rejected"]) == (1, 2, 1)
    assert alone["delta_JF"] == 0.0 and set(events["method"]) == {
        "fix_rules_csrt_tiny",
        "fix_verified_rules_csrt_tiny",
    }


def test_written_csv_has_the_documented_columns(tmp_path: Path) -> None:
    build(tmp_path)
    out = tmp_path / "e11.csv"
    table.main(
        [
            "--results-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--events-out",
            str(tmp_path / "ev.csv"),
            "--models",
            "tiny",
            "--splits",
            SPLIT,
        ]
    )
    assert list(pd.read_csv(out).columns) == list(table.TABLE_COLUMNS)


def test_missing_or_mismatched_inputs_stop(tmp_path: Path) -> None:
    build(tmp_path, verified={"a": 0.6, "z": 0.9})
    with pytest.raises(E2Error):
        table.build_tables(tmp_path, ["tiny"], [SPLIT])
    build(tmp_path)
    (tmp_path / f"events_fix_verified_rules_csrt_tiny_{SPLIT}.csv").unlink()
    with pytest.raises(E2Error, match="missing"):
        table.build_tables(tmp_path, ["tiny"], [SPLIT])
