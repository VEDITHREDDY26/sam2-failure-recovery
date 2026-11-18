"""Tests for the by-quality analysis: bin edges, counts, the all row matching the published
table and a deterministic bootstrap.
"""

from pathlib import Path

import pandas as pd
import pytest

from experiments.e12 import by_quality as bq

ALONE = {("a", 1): 0.1, ("a", 2): 0.5, ("b", 1): 0.79, ("b", 2): 0.8, ("c", 1): 0.95}
FIX = {("a", 1): 0.3, ("a", 2): 0.4, ("b", 1): 0.79, ("b", 2): 0.9, ("c", 1): 0.9}
METHOD = bq.Method("tiny", "fix_rules_csrt_tiny", False, "e2")


def write_scores(path: Path, scores: dict[tuple[str, int], float]) -> None:
    rows = [(v, o, s, s, s) for (v, o), s in scores.items()]
    pd.DataFrame(rows, columns=["video", "object", "J", "F", "J&F"]).to_csv(path, index=False)


def make_results(tmp_path: Path, table_delta: float | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    write_scores(tmp_path / "sam2_tiny_davis_val.csv", ALONE)
    write_scores(tmp_path / "fix_rules_csrt_tiny_davis_val.csv", FIX)
    delta = sum(FIX.values()) / 5 - sum(ALONE.values()) / 5 if table_delta is None else table_delta
    pd.DataFrame(
        [{"model": "tiny", "test_set": "davis_val", "method": METHOD.name, "delta_JF": delta}]
    ).to_csv(tmp_path / "e2.csv", index=False)
    return tmp_path


def build(results: Path, resamples: int = 200) -> pd.DataFrame:
    rows = bq.build(results, [METHOD], ["davis_val"], resamples=resamples)
    return pd.DataFrame(rows).set_index("bin")


def test_bin_edges() -> None:
    assert bq.quality_bin(0.4999) == "lost"
    assert bq.quality_bin(0.5) == "partial"
    assert bq.quality_bin(0.7999) == "partial"
    assert bq.quality_bin(0.8) == "good"


def test_bins_counts_and_all_row(tmp_path: Path) -> None:
    table = build(make_results(tmp_path))
    assert list(table.index) == ["lost", "partial", "good", "all"]
    assert table["objects"].to_dict() == {"lost": 1, "partial": 2, "good": 2, "all": 5}
    assert table.loc["lost", "delta_jf"] == pytest.approx(0.2)
    assert table.loc["all", "delta_jf"] == pytest.approx(
        sum(FIX.values()) / 5 - sum(ALONE.values()) / 5
    )
    assert table.loc["all", "objects_improved"] == 2
    assert table.loc["all", "objects_worse"] == 2
    assert table.loc["all", "objects_unchanged"] == 1
    assert table.loc["good", "videos"] == 2


def test_all_row_must_match_the_published_table(tmp_path: Path) -> None:
    with pytest.raises(bq.E12Error, match="differs"):
        build(make_results(tmp_path, table_delta=0.5))


def test_bootstrap_is_deterministic(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    first, second = build(results), build(results)
    pd.testing.assert_frame_equal(first, second)
    assert first.loc["all", "ci_low"] <= first.loc["all", "delta_jf"] <= first.loc["all", "ci_high"]


def test_missing_input_and_object_mismatch_refused(tmp_path: Path) -> None:
    with pytest.raises(bq.E12Error, match="missing"):
        bq.build(tmp_path, [METHOD], ["davis_val"])
    results = make_results(tmp_path)
    write_scores(results / "fix_rules_csrt_tiny_davis_val.csv", {("a", 1): 0.3})
    with pytest.raises(bq.E12Error, match="different objects"):
        build(results)


def test_empty_bin_is_kept_and_optional_method_skipped(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    write_scores(results / "sam2_tiny_davis_val.csv", {("a", 1): 0.9})
    write_scores(results / "fix_rules_csrt_tiny_davis_val.csv", {("a", 1): 0.9})
    pd.DataFrame(
        [{"model": "tiny", "test_set": "davis_val", "method": METHOD.name, "delta_JF": 0.0}]
    ).to_csv(results / "e2.csv", index=False)
    table = build(results)
    assert table.loc["lost", "objects"] == 0
    optional = bq.Method("tiny", "fix_continuous_rules_csrt_tiny", False, "e8", optional=True)
    assert bq.build(results, [optional], ["davis_val"]) == []


def test_main_writes_csv(tmp_path: Path) -> None:
    results = make_results(tmp_path / "r")
    out = tmp_path / "out" / "q.csv"
    bq.write_rows(out, bq.build(results, [METHOD], ["davis_val"], resamples=50))
    assert list(pd.read_csv(out).columns) == list(bq.OUT_COLUMNS)
