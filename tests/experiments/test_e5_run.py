import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.e2.table import E2Error
from experiments.e5 import run_learned, table
from experiments.e5 import score as e5_score
from src.common.layout import events_path, signals_path
from src.detection.learned import FEATURE_NAMES, LearnedModel, save_model
from src.eval.rows import ScoreRow, write_scores_csv
from src.recovery.events import EVENT_COLUMNS
from tests.eval.sam2_trees import build_world, run_all
from tests.experiments.e2_trees import make_online_run
from tests.experiments.test_run_online import write_prompts
from tests.recovery.fakes import StatefulFake
from tests.recovery.online_helpers import SPLIT, VIDEO

CONFIG = "configs/recovery/learned_csrt_tiny.yaml"
RULES_CONFIG = "configs/recovery/rules_csrt_tiny.yaml"
N = len(FEATURE_NAMES)


def quiet_model() -> LearnedModel:
    return LearnedModel("tiny", (0.0,) * N, (1.0,) * N, (0.0,) * N, -20.0, 0.5)


def test_help_works_without_data_or_gpu(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        run_learned.main(["--help"])
    assert stop.value.code == 0 and "--config" in capsys.readouterr().out


def test_rules_config_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="learned"):
        run_learned.main(["--config", RULES_CONFIG, "--prompts", str(tmp_path / "p.json")])


def test_missing_or_modified_model_stops_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = write_prompts(tmp_path, monkeypatch)
    monkeypatch.setattr(run_learned, "build_predictor", lambda *a, **k: pytest.fail("built"))
    monkeypatch.setattr(run_learned, "LEARNED_DIR", tmp_path / "learned")
    with pytest.raises(SystemExit, match="learned detector refused"):
        run_learned.main(["--config", CONFIG, "--prompts", str(prompts)])
    path = tmp_path / "learned" / "logreg_tiny.json"
    save_model(path, quiet_model())
    path.write_text(path.read_text().replace('"intercept": -20.0', '"intercept": 20.0'))
    with pytest.raises(SystemExit, match="does not match"):
        run_learned.main(["--config", CONFIG, "--prompts", str(prompts)])
    assert not (tmp_path / "drive" / "outputs").exists()


def test_sav_val_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(run_learned, "LEARNED_DIR", tmp_path / "learned")
    save_model(tmp_path / "learned" / "logreg_tiny.json", quiet_model())
    prompts = tmp_path / "p.json"
    entry = {"frame_index": 0, "boxes": {"1": [0, 0, 5, 5]}}
    prompts.write_text(json.dumps({"split": "sav_val", "videos": {"v": entry}}))
    monkeypatch.setattr(run_learned, "build_predictor", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(SystemExit, match="refused"):
        run_learned.main(["--config", CONFIG, "--prompts", str(prompts)])


def test_learned_system_runs_end_to_end_with_a_fake_predictor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = write_prompts(tmp_path, monkeypatch)
    save_model(tmp_path / "learned" / "logreg_tiny.json", quiet_model())
    monkeypatch.setattr(run_learned, "LEARNED_DIR", tmp_path / "learned")
    monkeypatch.setattr(run_learned, "build_predictor", lambda *a, **k: StatefulFake())
    run_learned.main(["--config", CONFIG, "--prompts", str(prompts)])
    out = tmp_path / "drive" / "outputs"
    assert signals_path(out, "fix_learned_csrt_tiny", SPLIT, VIDEO).is_file()
    events = pd.read_csv(events_path(out, "fix_learned_csrt_tiny", SPLIT, VIDEO))
    assert list(events.columns) == list(EVENT_COLUMNS) and events.empty


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset, models=("tiny",), splits=("davis_val", "sav_test"))
    return root / "outputs", subset


def test_learned_run_is_scored_under_its_name(world: tuple[Path, Path], tmp_path: Path) -> None:
    out, subset = world
    make_online_run(out, "fix_learned_csrt_tiny")
    results = tmp_path / "results"
    code = e5_score.main(
        ["--config", CONFIG, "--subset-file", str(subset)]
        + ["--results-dir", str(results), "--splits", SPLIT]
    )
    assert code == 0
    assert (results / f"fix_learned_csrt_tiny_{SPLIT}.csv").is_file()


def test_scoring_refuses_the_rules_config(world: tuple[Path, Path], tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="learned"):
        e5_score.main(["--config", RULES_CONFIG, "--results-dir", str(tmp_path / "r")])


def write_results(results: Path, learned_objects: int = 2) -> None:
    results.mkdir(exist_ok=True)
    jf = {"sam2_tiny": 0.5, table.METHODS["rules"]: 0.7, table.METHODS["learned"]: 0.65}
    for method, value in jf.items():
        n = learned_objects if method == table.METHODS["learned"] else 2
        rows = [ScoreRow(f"v{i}", 1, value, value) for i in range(n)]
        write_scores_csv(results / f"{method}_davis_val.csv", rows)
        if method != "sam2_tiny":
            events = [("v0", 1, 3, "score", "reprompted", "", "", "", "", "", 0)] * (
                1 if "rules" in method else 3
            )
            pd.DataFrame(events, columns=list(EVENT_COLUMNS)).to_csv(
                results / f"events_{method}_davis_val.csv", index=False
            )
    det = [
        ("tiny", "logreg", 8, 2, 30, 10, 25, 9, 0.36, 0.9, 0.51, 0.77, 0.2, "f1_protocol", False),
        (
            "tiny",
            "logreg",
            8,
            2,
            30,
            10,
            12,
            8,
            0.67,
            0.8,
            0.73,
            0.77,
            0.4,
            "f1_flag_cap_0.10",
            True,
        ),
        (
            "tiny",
            "rules",
            8,
            2,
            30,
            10,
            14,
            7,
            0.5,
            0.7,
            0.58,
            np.nan,
            np.nan,
            "rules_frozen",
            True,
        ),
    ]
    pd.DataFrame(det, columns=list(table.DETECTOR_COLUMNS)).to_csv(
        results / "e5_detector.csv", index=False
    )
    weights = [("tiny", f, 0.1 * i, 0.0, 1.0) for i, f in enumerate(FEATURE_NAMES)]
    pd.DataFrame(weights, columns=list(table.WEIGHT_COLUMNS)).to_csv(
        results / "e5_weights.csv", index=False
    )


def test_table_joins_detector_scores_and_weights(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_results(results)
    out = tmp_path / "e5.csv"
    table.main(["--results-dir", str(results), "--out", str(out), "--splits", "davis_val"])
    frame = pd.read_csv(out, keep_default_na=False)
    assert list(frame.columns) == list(table.TABLE_COLUMNS)
    rules, learned = frame.iloc[0], frame.iloc[1]
    assert (rules["detector"], learned["detector"]) == ("rules", "learned")
    assert np.isclose(learned["delta_vs_alone"], 0.15) and np.isclose(
        learned["delta_vs_rules"], -0.05
    )
    assert rules["delta_vs_rules"] == 0.0
    assert (rules["reprompts"], learned["reprompts"]) == (1, 3)
    assert learned["precision"] == 0.67 and rules["tp"] == 7
    assert float(learned["w_gone"]) == pytest.approx(0.1 * FEATURE_NAMES.index("gone"))
    assert rules["w_gone"] == ""


def test_table_refuses_missing_inputs_and_mismatched_objects(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_results(results, learned_objects=1)
    with pytest.raises(table.E5Error, match="different videos"):
        table.build_table(results, ["davis_val"])
    with pytest.raises(SystemExit, match="e5 table"):
        table.main(["--results-dir", str(results), "--out", str(tmp_path / "x.csv")])
    (results / "e5_weights.csv").unlink()
    with pytest.raises(table.E5Error, match="missing input"):
        table.build_table(results, ["davis_val"])
    write_results(results)
    (results / "events_fix_learned_csrt_tiny_davis_val.csv").unlink()
    with pytest.raises(E2Error, match="missing input"):
        table.build_table(results, ["davis_val"])
