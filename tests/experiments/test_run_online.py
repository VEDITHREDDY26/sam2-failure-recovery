import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.e2 import run_online
from src.common.layout import events_path, signals_path
from src.recovery.events import EVENT_COLUMNS
from tests.recovery.fakes import StatefulFake
from tests.recovery.online_helpers import SPLIT, VIDEO, make_tree

REAL_CONFIG = "configs/recovery/rules_csrt_tiny.yaml"
ORACLE_CONFIG = "configs/oracle/oracle_flags_csrt_tiny.yaml"
QUIET = (
    "{score_threshold: -100.0, area_ratio: 100.0, jump_ratio: 100.0,"
    " min_area_fraction: 0.5, border_margin_px: 8}"
)


def write_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    make_tree(tmp_path, monkeypatch)
    entry = {"frame_index": 0, "boxes": {"1": [10, 20, 22, 32]}}
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"split": SPLIT, "videos": {VIDEO: entry}}))
    return path


def test_help_works_without_data_or_gpu(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        run_online.main(["--help"])
    assert stop.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_oracle_config_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="real system only"):
        run_online.main(["--config", ORACLE_CONFIG, "--prompts", str(tmp_path / "p.json")])


def test_missing_rules_file_stops_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = write_prompts(tmp_path, monkeypatch)
    monkeypatch.setattr(run_online, "build_predictor", lambda *a, **k: pytest.fail("built"))
    monkeypatch.setattr(run_online, "DEFAULT_RULES_PATH", tmp_path / "no.yaml")
    with pytest.raises(SystemExit, match="detector not frozen yet"):
        run_online.main(["--config", REAL_CONFIG, "--prompts", str(prompts)])
    assert not (tmp_path / "drive" / "outputs").exists()


def test_real_system_runs_end_to_end_with_a_fake_predictor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = write_prompts(tmp_path, monkeypatch)
    rules = tmp_path / "rules.yaml"
    rules.write_text(f"kind: rules\ntiny: {QUIET}\nlarge: {QUIET}\n")
    monkeypatch.setattr(run_online, "build_predictor", lambda *a, **k: StatefulFake())
    monkeypatch.setattr(run_online, "DEFAULT_RULES_PATH", rules)
    run_online.main(["--config", REAL_CONFIG, "--prompts", str(prompts)])
    out = tmp_path / "drive" / "outputs"
    assert signals_path(out, "fix_rules_csrt_tiny", SPLIT, VIDEO).is_file()
    events = pd.read_csv(events_path(out, "fix_rules_csrt_tiny", SPLIT, VIDEO))
    assert list(events.columns) == list(EVENT_COLUMNS) and events.empty
