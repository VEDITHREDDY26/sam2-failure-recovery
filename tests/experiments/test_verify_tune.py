import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.verify import tune
from experiments.verify.draw_subset import draw
from experiments.verify.label_candidates import LABEL_COLUMNS, labels_name
from src.recovery.verify import VerifyThresholds, load_verify_thresholds


def table(rows: list[tuple[float, float, float, float]], video: str = "v") -> pd.DataFrame:
    """rows are (cand_score, area_ratio, box_iou, gain)."""
    data = [
        (video, 1, i, s, 0.0, a, b, 0.5, True, "", i, 0.5, 0.5, g)
        for i, (s, a, b, g) in enumerate(rows)
    ]
    return pd.DataFrame(data, columns=list(LABEL_COLUMNS))


def test_grid_is_the_fixed_200_settings_in_order() -> None:
    settings = tune.grid()
    assert len(settings) == 200 == 10 * 5 * 4
    first, last = settings[0], settings[-1]
    assert (first.s_min, first.r_max, first.b_min) == (None, None, None)
    assert (last.s_min, last.r_max, last.b_min) == (12.0, 4.0, 0.7)
    assert settings[1].b_min == 0.3 and settings[4].r_max == 1.5


def test_accept_mask_matches_the_run_time_rule() -> None:
    rng = np.random.default_rng(0)
    rows = [
        (
            float(rng.uniform(-6, 14)),
            float(rng.choice([1.0, 2.5, 9.0, math.inf])),
            float(rng.random()),
            0.1,
        )
        for _ in range(120)
    ]
    data = table(rows)
    for setting in tune.grid():
        rule = setting.thresholds()
        mask = tune.accept_mask(data, rule)
        expected = [not rule.failed(s, a, b) for s, a, b, _ in rows]
        assert mask.tolist() == expected


def test_best_setting_maximises_the_gain_sum_and_is_marked() -> None:
    rows = [
        (10.0, 1.0, 0.9, 0.3),
        (9.0, 1.1, 0.9, 0.2),
        (1.0, 1.0, 0.9, -0.4),
        (-3.0, 5.0, 0.1, -0.5),
    ]
    result, rule = tune.tune_model(table(rows), "tiny")
    assert len(result) == 200 and result["chosen"].sum() == 1
    pick = result[result["chosen"]].iloc[0]
    assert pick["gain_sum"] == pytest.approx(0.5) and pick["accepted"] == 2
    assert pick["gain_positive"] == 2 and pick["gain_negative"] == 0
    assert rule == VerifyThresholds(8.0, 1.5, 0.7, False)


def test_ties_prefer_fewer_accepted_then_stricter() -> None:
    rows = [(10.0, 1.0, 0.9, 0.5), (4.0, 1.0, 0.9, 0.0), (0.0, 1.0, 0.9, 0.0)]
    result, rule = tune.tune_model(table(rows), "tiny")
    assert result[result["chosen"]].iloc[0]["accepted"] == 1
    assert rule == VerifyThresholds(10.0, 1.5, 0.7, False)


def test_accept_nothing_when_no_setting_has_positive_gain() -> None:
    rows = [(10.0, 1.0, 0.9, -0.2), (9.0, 1.0, 0.9, 0.0), (3.0, 1.0, 0.9, math.nan)]
    result, rule = tune.tune_model(table(rows), "large")
    assert rule.accept_nothing and len(result) == 201
    pick = result[result["chosen"]]
    assert pick["setting"].tolist() == ["accept_nothing"] and pick["accepted"].tolist() == [0]
    assert rule.failed(100.0, 1.0, 1.0) == ["accept_nothing"]


def test_unscored_candidates_add_no_gain_but_count_as_accepted() -> None:
    data = table([(5.0, 1.0, 0.9, 0.4), (5.0, 1.0, 0.9, math.nan)])
    row = tune.score_setting(data, "tiny", tune.grid()[0])
    assert row["accepted"] == 2 and row["accepted_scored"] == 1 and row["gain_sum"] == 0.4


def write_inputs(results: Path, subset: list[str]) -> None:
    good = table([(10.0, 1.0, 0.9, 0.3), (1.0, 1.0, 0.9, -0.3)])
    good.to_csv(results / labels_name("tiny"), index=False)
    table([(10.0, 1.0, 0.9, 0.3)], video=subset[0]).to_csv(
        results / labels_name("large"), index=False
    )


def run(tmp_path: Path, *extra: str) -> int:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    subset = [f"v{i}" for i in range(60)]
    subset_file = tmp_path / "subset.txt"
    subset_file.write_text("\n".join(subset) + "\n")
    if not (results / labels_name("tiny")).exists():
        write_inputs(results, subset)
    return tune.main(
        [
            "--results-dir",
            str(results),
            "--config-out",
            str(tmp_path / "verify.yaml"),
            "--results-out",
            str(results / "verify_sav_val.csv"),
            "--subset-file",
            str(subset_file),
            *extra,
        ]
    )


def test_end_to_end_writes_results_and_a_loadable_config(tmp_path: Path) -> None:
    assert run(tmp_path) == 0
    result = pd.read_csv(tmp_path / "results" / "verify_sav_val.csv")
    assert set(result["model"]) == {"tiny", "large"}
    assert (result.groupby("model")["chosen"].sum() == 1).all()
    tiny = load_verify_thresholds("tiny", tmp_path / "verify.yaml")
    assert tiny.failed(10.0, 1.0, 0.9) == [] and tiny.failed(1.0, 1.0, 0.9) != []


def test_existing_config_is_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / "verify.yaml").write_text("kind: verify\n")
    with pytest.raises(SystemExit, match="frozen once"):
        run(tmp_path)
    assert (tmp_path / "verify.yaml").read_text() == "kind: verify\n"


def test_test_splits_are_refused(tmp_path: Path) -> None:
    for split in ("davis_val", "sav_test"):
        with pytest.raises(SystemExit, match="refused"):
            run(tmp_path, "--split", split)
    assert not (tmp_path / "verify.yaml").exists()


def test_large_candidates_must_come_from_the_subset(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    write_inputs(results, ["outside_video"])
    with pytest.raises(SystemExit, match="outside the tuning subset"):
        run(tmp_path)
    assert not (tmp_path / "verify.yaml").exists()


def test_committed_subset_draw_is_what_the_script_would_draw() -> None:
    names = [f"n{i}" for i in range(155)]
    assert len(draw(names)) == 60
