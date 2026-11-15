import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.e2 import run_online
from src.common.layout import events_path, signals_path
from src.recovery.events import EVENT_COLUMNS
from src.recovery.verify import candidate_path
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


def test_continuous_variant_runs_with_the_tracker_mode_of_its_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = write_prompts(tmp_path, monkeypatch)
    rules = tmp_path / "rules.yaml"
    rules.write_text(f"kind: rules\ntiny: {QUIET}\nlarge: {QUIET}\n")
    seen: list[str] = []
    real = run_online.CSRTBoxSource

    def spy(mode: str) -> object:
        seen.append(mode)
        return real(mode)

    monkeypatch.setattr(run_online, "build_predictor", lambda *a, **k: StatefulFake())
    monkeypatch.setattr(run_online, "DEFAULT_RULES_PATH", rules)
    monkeypatch.setattr(run_online, "CSRTBoxSource", spy)
    run_online.main(
        ["--config", "configs/recovery/rules_csrt_continuous_tiny.yaml", "--prompts", str(prompts)]
    )
    assert seen == ["continuous"]
    out = tmp_path / "drive" / "outputs"
    assert signals_path(out, "fix_continuous_rules_csrt_tiny", SPLIT, VIDEO).is_file()


VERIFIED_CONFIG = "configs/recovery/verified_rules_csrt_tiny.yaml"
ACCEPT_ALL = (
    "kind: verify\n"
    "tiny: {s_min: -.inf, r_max: .inf, b_min: -.inf, accept_nothing: false}\n"
    "large: {s_min: -.inf, r_max: .inf, b_min: -.inf, accept_nothing: false}\n"
)


def verified_config(tmp_path: Path, verify_file: Path) -> Path:
    text = Path(VERIFIED_CONFIG).read_text().replace("configs/verify/verify.yaml", str(verify_file))
    path = tmp_path / "verified.yaml"
    path.write_text(text)
    return path


def quiet_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rules = tmp_path / "rules.yaml"
    rules.write_text(f"kind: rules\ntiny: {QUIET}\nlarge: {QUIET}\n")
    monkeypatch.setattr(run_online, "build_predictor", lambda *a, **k: StatefulFake())
    monkeypatch.setattr(run_online, "DEFAULT_RULES_PATH", rules)


def test_verified_run_refuses_until_the_verify_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = write_prompts(tmp_path, monkeypatch)
    quiet_rules(tmp_path, monkeypatch)
    config = verified_config(tmp_path, tmp_path / "missing.yaml")
    with pytest.raises(SystemExit, match="not frozen"):
        run_online.main(["--config", str(config), "--prompts", str(prompts)])
    assert not signals_path(
        tmp_path / "drive" / "outputs", "fix_verified_rules_csrt_tiny", SPLIT, VIDEO
    ).exists()


def test_verified_run_writes_the_candidate_log_next_to_the_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = write_prompts(tmp_path, monkeypatch)
    quiet_rules(tmp_path, monkeypatch)
    verify = tmp_path / "verify.yaml"
    verify.write_text(ACCEPT_ALL)
    config = verified_config(tmp_path, verify)
    run_online.main(["--config", str(config), "--prompts", str(prompts)])
    out = tmp_path / "drive" / "outputs"
    method = "fix_verified_rules_csrt_tiny"
    assert signals_path(out, method, SPLIT, VIDEO).is_file()
    assert candidate_path(out, method, SPLIT, VIDEO).is_file()


def test_verified_run_refuses_the_tuning_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_cfg = run_online.load_recovery_config(Path(VERIFIED_CONFIG))
    with pytest.raises(SystemExit, match="tuning split"):
        run_online.choose_verifier(run_cfg, "tiny", "sav_val", False)
    with pytest.raises(SystemExit, match="accept-all"):
        run_online.choose_verifier(run_cfg, "tiny", "sav_test", True)


def test_candidate_logging_is_for_sav_val_and_the_plain_method_only() -> None:
    plain = run_online.load_recovery_config(Path(REAL_CONFIG))
    assert run_online.choose_verifier(plain, "tiny", "sav_val", True) is None
    assert run_online.choose_verifier(plain, "tiny", "davis_val", False) is None
    for split in ("davis_val", "sav_test"):
        with pytest.raises(SystemExit, match="tuning run"):
            run_online.choose_verifier(plain, "tiny", split, True)
