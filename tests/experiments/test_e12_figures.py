"""Tests for the analysis figures: files are written, and missing inputs are refused."""

from pathlib import Path

import pandas as pd
import pytest

from experiments.e12 import figures

BINS = figures.BINS


def write_tables(results: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    rows = []
    for test_set in figures.TEST_SETS:
        for model, method, _, _ in figures.BY_QUALITY_SERIES:
            for name in BINS:
                rows.append(
                    {
                        "model": model,
                        "test_set": test_set,
                        "method": method,
                        "bin": name,
                        "objects": 3,
                        "delta_jf": 0.01,
                        "ci_low": -0.02,
                        "ci_high": 0.04,
                    }
                )
    rows[0].update({"objects": 0, "delta_jf": "", "ci_low": "", "ci_high": ""})
    pd.DataFrame(rows).to_csv(results / "e12_by_quality.csv", index=False)
    e7 = [
        {"arm": arm, "test_set": test_set, "J&F": 0.7 + i / 100, "delta_JF": i / 100 - 0.01}
        for test_set in figures.TEST_SETS
        for i, arm in enumerate([figures.E7_BASE_ARM, *(a[0] for a in figures.E7_ARMS)])
    ]
    pd.DataFrame(e7).to_csv(results / "e7.csv", index=False)
    split = [
        {"model": m, "class": c, "gain_sum": g}
        for m in ("tiny", "large")
        for c, g in (("real_failure", 5.0), ("false_alarm", -9.0), ("unscored", 0.0))
    ]
    pd.DataFrame(split).to_csv(results / "e12_false_alarms.csv", index=False)
    bound = [
        {"model": m, "kind": "strategy", "name": n, "gain_sum": g}
        for m in ("tiny", "large")
        for n, g in (("accept_all", -4.0), ("best_grid", -2.0), ("perfect_verifier", 6.0))
    ]
    pd.DataFrame(bound).to_csv(results / "e12_upper_bound.csv", index=False)


def test_figures_are_written(tmp_path: Path) -> None:
    write_tables(tmp_path / "r")
    figures.main(["--results-dir", str(tmp_path / "r"), "--figure-dir", str(tmp_path / "f")])
    for name in ("e12_by_quality.png", "e7_decomposition.png", "e12_false_alarms.png"):
        assert (tmp_path / "f" / name).stat().st_size > 1000


def test_missing_input_refused(tmp_path: Path) -> None:
    with pytest.raises(figures.FigureError, match="missing"):
        figures.make_figures(tmp_path, tmp_path / "f")


def test_missing_e7_arm_refused(tmp_path: Path) -> None:
    write_tables(tmp_path)
    e7 = pd.read_csv(tmp_path / "e7.csv")
    e7[e7["arm"] != "rules_oracle_box"].to_csv(tmp_path / "e7.csv", index=False)
    with pytest.raises(figures.FigureError, match="rules_oracle_box"):
        figures.make_figures(tmp_path, tmp_path / "f")
