import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.e7 import arms, check_arms, run_oracle, table
from experiments.e7 import score as e7_score
from src.common.layout import events_path, signals_path
from src.eval.rows import ScoreRow, write_scores_csv
from src.recovery.events import EVENT_COLUMNS
from tests.eval.sam2_trees import build_world, run_all
from tests.experiments.e2_trees import make_online_run
from tests.experiments.test_run_online import QUIET, write_prompts
from tests.oracle.test_online import sample
from tests.recovery.fakes import StatefulFake
from tests.recovery.online_helpers import SPLIT, VIDEO

ORACLE_CONFIGS = [
    "configs/oracle/oracle_flags_csrt_tiny.yaml",
    "configs/oracle/oracle_flags_oracle_box_tiny.yaml",
    "configs/oracle/rules_oracle_box_tiny.yaml",
]
ENTRY = {"v": {"frame_index": 0, "boxes": {"1": [0, 0, 5, 5]}}}
REAL_CONFIG = "configs/recovery/rules_csrt_tiny.yaml"


def test_help_works_without_data_or_gpu(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        run_oracle.main(["--help"])
    assert stop.value.code == 0 and "--config" in capsys.readouterr().out


def test_real_system_config_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="oracle arms only"):
        run_oracle.main(["--config", REAL_CONFIG, "--prompts", str(tmp_path / "p.json")])


def test_sav_val_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(tmp_path))
    prompts = tmp_path / "p.json"
    prompts.write_text(json.dumps({"split": "sav_val", "videos": ENTRY}))
    monkeypatch.setattr(run_oracle, "build_predictor", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(SystemExit, match="refused"):
        run_oracle.main(["--config", ORACLE_CONFIGS[1], "--prompts", str(prompts)])


def test_rules_are_needed_only_by_the_arm_that_uses_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = write_prompts(tmp_path, monkeypatch)
    monkeypatch.setattr(run_oracle, "build_predictor", lambda *a, **k: StatefulFake())
    monkeypatch.setattr(run_oracle, "DEFAULT_RULES_PATH", tmp_path / "no.yaml")
    cfg = run_oracle.load_data_config()
    monkeypatch.setattr(run_oracle, "load_video", lambda *_: sample(cfg))
    with pytest.raises(SystemExit, match="detector not frozen yet"):
        run_oracle.main(["--config", ORACLE_CONFIGS[2], "--prompts", str(prompts)])
    run_oracle.main(["--config", ORACLE_CONFIGS[1], "--prompts", str(prompts)])
    out = tmp_path / "drive" / "outputs"
    method = "fix_oracle_flags_oracle_box_tiny"
    assert signals_path(out, method, SPLIT, VIDEO).is_file()
    assert list(pd.read_csv(events_path(out, method, SPLIT, VIDEO)).columns) == list(EVENT_COLUMNS)


def test_rules_oracle_box_arm_runs_with_frozen_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = write_prompts(tmp_path, monkeypatch)
    rules = tmp_path / "rules.yaml"
    rules.write_text(f"kind: rules\ntiny: {QUIET}\nlarge: {QUIET}\n")
    monkeypatch.setattr(run_oracle, "build_predictor", lambda *a, **k: StatefulFake())
    monkeypatch.setattr(run_oracle, "DEFAULT_RULES_PATH", rules)
    cfg = run_oracle.load_data_config()
    monkeypatch.setattr(run_oracle, "load_video", lambda *_: sample(cfg))
    run_oracle.main(["--config", ORACLE_CONFIGS[2], "--prompts", str(prompts)])
    out = tmp_path / "drive" / "outputs"
    assert signals_path(out, "fix_rules_oracle_box_tiny", SPLIT, VIDEO).is_file()


def test_shipped_oracle_configs_match_the_arms() -> None:
    names = {a.method for a in arms.ARMS if a.oracle}
    found = set()
    for path in ORACLE_CONFIGS:
        cfg = run_oracle.load_recovery_config(Path(path))
        run_oracle.check_config(cfg, Path(path))
        found.add(cfg.method)
    assert found == names and len(arms.ARMS) == 4


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset, models=("tiny",), splits=("davis_val", "sav_test"))
    return root / "outputs", subset


def test_oracle_arm_is_scored_under_its_oracle_name(
    world: tuple[Path, Path], tmp_path: Path
) -> None:
    out, subset = world
    method = "fix_oracle_flags_csrt_tiny"
    make_online_run(out, method)
    results = tmp_path / "results"
    code = e7_score.main(
        ["--config", ORACLE_CONFIGS[0], "--subset-file", str(subset)]
        + ["--results-dir", str(results), "--splits", SPLIT]
    )
    assert code == 0
    written = sorted(p.name for p in results.iterdir())
    assert written and all("oracle" in n for n in written)


def test_scoring_refuses_the_real_system_config(world: tuple[Path, Path], tmp_path: Path) -> None:
    _, subset = world
    with pytest.raises(SystemExit, match="oracle arms only"):
        e7_score.main(["--config", REAL_CONFIG, "--results-dir", str(tmp_path / "r")])


def make_results(results: Path, drop: tuple[str, str] | None = None) -> None:
    objects = {("a", 1): 0.5, ("a", 2): 0.6, ("b", 1): 0.7}
    gains = {arms.ALONE: 0.0}
    gains.update({a.method: g for a, g in zip(arms.ARMS, (0.05, 0.1, 0.2, 0.3), strict=True)})
    for split in arms.TEST_SETS:
        for method, gain in gains.items():
            rows = [ScoreRow(v, o, j + gain, j + gain) for (v, o), j in objects.items()]
            if drop and drop[0] == method and drop[1] == split:
                rows = rows[:-1]
            write_scores_csv(results / f"{method}_{split}.csv", rows)
            if method != arms.ALONE:
                events = [("a", 1, 3, "x", "reprompted", "", "", "", "", "", 0)]
                pd.DataFrame(events, columns=list(EVENT_COLUMNS)).to_csv(
                    results / f"events_{method}_{split}.csv", index=False
                )


def make_prompts(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    boxes = {"a": {"frame_index": 0, "boxes": {"1": [0, 0, 5, 5], "2": [1, 1, 6, 6]}}}
    boxes["b"] = {"frame_index": 0, "boxes": {"1": [2, 2, 8, 8]}}
    for split in arms.TEST_SETS:
        (directory / f"{split}.json").write_text(json.dumps({"split": split, "videos": boxes}))
    return {}


def run_check(tmp_path: Path, *extra: str) -> None:
    prompts = tmp_path / "prompts"
    make_prompts(prompts)
    check_arms.main(
        ["--results-dir", str(tmp_path), "--prompts-dir", str(prompts)]
        + ["--out", str(tmp_path / "e7_arms_check.csv"), *extra]
    )


def test_check_passes_and_records_one_hash_per_test_set(tmp_path: Path) -> None:
    make_results(tmp_path)
    run_check(tmp_path)
    got = pd.read_csv(tmp_path / "e7_arms_check.csv")
    assert list(got.columns) == list(check_arms.CHECK_COLUMNS)
    assert got.groupby("test_set")["prompts_sha256"].nunique().tolist() == [1, 1]
    assert len(got) == 2 * 5


def test_check_fails_when_one_arm_used_different_prompts(tmp_path: Path) -> None:
    make_results(tmp_path)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"split": "davis_val", "videos": {}}))
    with pytest.raises(check_arms.ArmsMismatch, match="different prompts"):
        run_check(tmp_path, "--arm-prompts", f"oracle_flags_csrt:davis_val={other}")


def test_check_fails_when_an_arm_scored_other_objects(tmp_path: Path) -> None:
    make_results(tmp_path, drop=("fix_rules_oracle_box_tiny", "sav_test"))
    with pytest.raises(check_arms.ArmsMismatch, match="differ"):
        run_check(tmp_path)


def test_table_labels_oracle_rows_and_reports_deltas(tmp_path: Path) -> None:
    make_results(tmp_path)
    run_check(tmp_path)
    out = tmp_path / "e7.csv"
    table.main(["--results-dir", str(tmp_path), "--out", str(out)])
    got = pd.read_csv(out)
    assert list(got.columns) == list(table.TABLE_COLUMNS)
    assert len(got) == 2 * 5 and set(got["test_set"]) == set(arms.TEST_SETS)
    assert all(got["arm"].str.contains("oracle") == got["oracle"])
    real = got[got["arm"] == "rules_csrt"].iloc[0]
    assert not real["oracle"] and real["delta_JF"] == pytest.approx(0.05)
    top = got[got["arm"] == "oracle_flags_oracle_box"].iloc[0]
    assert top["oracle"] and top["delta_JF"] == pytest.approx(0.3) and top["reprompts"] == 1


def test_table_needs_the_arms_check_first(tmp_path: Path) -> None:
    make_results(tmp_path)
    with pytest.raises(table.E7Error, match="check_arms"):
        table.build_rows(tmp_path)
