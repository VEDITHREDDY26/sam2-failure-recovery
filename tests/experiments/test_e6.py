import csv
from pathlib import Path

import pandas as pd
import pytest

from experiments.e2.table import FRAME_COLUMNS, E2Error
from experiments.e6 import table
from src.eval.rows import ScoreRow, write_scores_csv

SPLIT = "davis_val"
# (video, J&F alone, J&F fix) per model; one improved, one worse, one unchanged
JF = {
    "tiny": [("v1", 0.5, 0.8), ("v2", 0.6, 0.5), ("v3", 0.7, 0.7005)],
    "large": [("v1", 0.8, 0.85), ("v2", 0.7, 0.7), ("v3", 0.9, 0.9)],
}


def write_model(results: Path, model: str) -> None:
    for column, name in ((1, f"sam2_{model}"), (2, f"fix_rules_csrt_{model}")):
        rows = [ScoreRow(entry[0], 1, entry[column], entry[column]) for entry in JF[model]]
        write_scores_csv(results / f"{name}_{SPLIT}.csv", rows)
    frames = [(v, 1, 20, 4, 2, 3 - i, i) for i, (v, _, _) in enumerate(JF[model])]
    pd.DataFrame(frames, columns=list(FRAME_COLUMNS)).to_csv(
        results / f"fix_rules_csrt_{model}_{SPLIT}_frames.csv", index=False
    )


def write_stats(results: Path) -> None:
    rows = [
        ("tiny", SPLIT, 3, 3, 0, 60, 12, 0.2, 3, 2),
        ("large", SPLIT, 3, 3, 0, 60, 6, 0.1, 2, 1),
        ("tiny", "sav_val", 9, 9, 0, 90, 30, 0.3, 9, 9),
    ]
    cols = [
        "model", "split", "videos", "objects", "skipped_objects", "labelled_frames",
        "failure_frames", "failure_rate", "objects_with_failure", "videos_with_failure",
    ]  # fmt: skip
    pd.DataFrame(rows, columns=cols).to_csv(results / "failure_stats.csv", index=False)


@pytest.fixture
def results(tmp_path: Path) -> Path:
    path = tmp_path / "results"
    path.mkdir()
    for model in JF:
        write_model(path, model)
    write_stats(path)
    return path


def test_per_model_gain_counts_and_failure_columns(results: Path, tmp_path: Path) -> None:
    out = tmp_path / "e6.csv"
    table.main(["--results-dir", str(results), "--out", str(out), "--splits", SPLIT])
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0]) == list(table.TABLE_COLUMNS)
    tiny, large = rows
    assert (tiny["model"], large["model"]) == ("tiny", "large")
    assert float(tiny["JF_alone"]) == pytest.approx(0.6)
    assert float(tiny["JF_fix"]) == pytest.approx((0.8 + 0.5 + 0.7005) / 3)
    assert float(tiny["gain_JF"]) == pytest.approx((0.3 - 0.1 + 0.0005) / 3)
    assert (tiny["videos_improved"], tiny["videos_worse"], tiny["videos_unchanged"]) == (
        "1",
        "1",
        "1",
    )
    assert (tiny["failure_frames"], tiny["failure_videos"]) == ("12", "2")
    assert (tiny["frames_recovered"], tiny["frames_broken"]) == ("6", "3")
    assert float(large["gain_JF"]) == pytest.approx(0.05 / 3)
    assert (large["failure_frames"], large["failure_videos"]) == ("6", "1")
    assert (large["videos_improved"], large["videos_worse"]) == ("1", "0")


def test_a_missing_input_is_refused(results: Path) -> None:
    (results / f"fix_rules_csrt_large_{SPLIT}_frames.csv").unlink()
    with pytest.raises(E2Error, match="missing input"):
        table.build_table(results, splits=[SPLIT])


def test_stats_row_must_exist_once(results: Path) -> None:
    stats = pd.read_csv(results / "failure_stats.csv")
    stats[stats["model"] != "large"].to_csv(results / "failure_stats.csv", index=False)
    with pytest.raises(E2Error, match="expected 1"):
        table.build_table(results, splits=[SPLIT])


def test_different_objects_are_refused(results: Path) -> None:
    write_scores_csv(results / f"fix_rules_csrt_tiny_{SPLIT}.csv", [ScoreRow("v1", 1, 0.1, 0.1)])
    with pytest.raises(E2Error, match="different videos"):
        table.build_table(results, splits=[SPLIT])
