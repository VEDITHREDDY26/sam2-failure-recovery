"""Tests for the false-alarm analysis: class thresholds, strategies, accept-all matching
the tuning table and AUC orientation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.e12 import false_alarms as fa
from experiments.e12 import upper_bound as ub
from experiments.e12.candidates import CandidateError

NAN = float("nan")
# iou_alone, iou_method, gain
CASES = [
    (0.2, 0.9, 0.7),
    (0.49, 0.3, -0.19),
    (0.5, 0.9, 0.4),
    (0.9, 0.2, -0.7),
    (0.8, 0.8, 0.0),
    (0.95, 0.4, -0.55),
    (NAN, NAN, NAN),
]


def candidates() -> pd.DataFrame:
    rows = []
    for i, (alone, method, gain) in enumerate(CASES):
        rows.append(
            {
                "video": f"v{i}",
                "object_id": 1,
                "frame_index": 10,
                "cand_score": 1.0 + i,
                "orig_score": 2.0,
                "area_ratio": 1.0,
                "box_iou": 0.5,
                "mask_iou_orig": 0.5,
                "iou_method": method,
                "iou_alone": alone,
                "gain": gain,
            }
        )
    return pd.DataFrame(rows)


def make_results(tmp_path: Path, accept_all: float | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidates().to_csv(tmp_path / "verify_candidates_tiny_sav_val.csv", index=False)
    total = float(np.nansum([c[2] for c in CASES])) if accept_all is None else accept_all
    grid = pd.DataFrame(
        [
            {"model": "tiny", "setting": "0", "gain_sum": total},
            {"model": "tiny", "setting": "1", "gain_sum": -0.3},
            {"model": "tiny", "setting": "accept_nothing", "gain_sum": 0.0},
        ]
    )
    grid.to_csv(tmp_path / "verify_sav_val.csv", index=False)
    return tmp_path


def test_class_threshold_and_unscored(tmp_path: Path) -> None:
    rows = {r["class"]: r for r in fa.build(make_results(tmp_path), ["tiny"])}
    assert rows["real_failure"]["candidates"] == 2
    assert rows["false_alarm"]["candidates"] == 4
    assert rows["unscored"]["candidates"] == 1
    assert rows["real_failure"]["gain_sum"] == pytest.approx(0.7 - 0.19)
    assert rows["false_alarm"]["helped"] == 1
    assert rows["false_alarm"]["hurt"] == 2
    assert rows["false_alarm"]["unchanged"] == 1
    assert rows["false_alarm"]["wrong_after"] == 2
    assert sum(float(r["share"]) for r in rows.values()) == pytest.approx(1.0)


def test_missing_candidates_refused(tmp_path: Path) -> None:
    with pytest.raises(CandidateError, match="missing input"):
        fa.build(tmp_path, ["tiny"])


def test_strategies(tmp_path: Path) -> None:
    rows = ub.build(make_results(tmp_path), ["tiny"])
    gains = {r["name"]: r["gain_sum"] for r in rows if r["kind"] == "strategy"}
    assert gains["accept_all"] == pytest.approx(-0.34)
    assert gains["perfect_verifier"] == pytest.approx(1.1)
    assert gains["frozen"] == 0.0
    assert gains["best_grid"] == pytest.approx(-0.3)


def test_accept_all_must_match_tuning_table(tmp_path: Path) -> None:
    with pytest.raises(CandidateError, match="differs"):
        ub.build(make_results(tmp_path, accept_all=5.0), ["tiny"])


def test_auc_orientation(tmp_path: Path) -> None:
    results = make_results(tmp_path)
    table = candidates()
    table["cand_score"] = [3.0, 1.0, 4.0, 0.0, 9.0, 0.5, 9.0]
    table["iou_alone"] = [0.1, 0.3, 0.2, 0.9, 0.8, 0.95, NAN]
    table["area_ratio"] = [1.0, 5.0, 1.0, 5.0, 1.0, np.inf, 1.0]
    table.to_csv(results / "verify_candidates_tiny_sav_val.csv", index=False)
    signals = {r["name"]: r for r in ub.build(results, ["tiny"]) if r["kind"] == "signal"}
    assert signals["cand_score"]["auc_higher_helped"] == 1.0
    assert signals["cand_score"]["direction"] == "higher_helped"
    assert signals["iou_alone"]["auc_higher_helped"] == 0.0
    assert signals["iou_alone"]["auc_best"] == 1.0
    assert signals["iou_alone"]["direction"] == "lower_helped"
    assert signals["iou_alone"]["ground_truth_reference"] is True
    assert signals["cand_score"]["helped"] == 2
    assert signals["cand_score"]["hurt"] == 3
    assert signals["area_ratio"]["hurt"] == 2


def test_cli_writes_csvs(tmp_path: Path) -> None:
    results = make_results(tmp_path / "r")
    candidates().to_csv(results / "verify_candidates_large_sav_val.csv", index=False)
    fa.main(["--results-dir", str(results), "--out", str(tmp_path / "o" / "fa.csv")])
    assert list(pd.read_csv(tmp_path / "o" / "fa.csv").columns) == list(fa.OUT_COLUMNS)
