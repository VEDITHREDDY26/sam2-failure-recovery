from pathlib import Path

import pytest

from experiments.e2 import run_online
from experiments.e13 import make_configs
from src.recovery import config as recovery_config
from src.recovery.config import GatedRunConfig, load_recovery_config

HEADER = "model,arm,s_min,a_max,reprompts,net,baseline_net,go\n"


def setup(
    tmp_path: Path, rows: str, gates: tuple[str, ...] = ("tiny", "large")
) -> tuple[Path, Path]:
    config_dir = tmp_path / "configs" / "recovery"
    config_dir.mkdir(parents=True)
    for model in gates:
        (config_dir / f"box_gate_{model}.yaml").write_text(
            "kind: box_gate\ns_min: 0.4\na_max: .inf\n"
        )
    choice = tmp_path / "e13_choice.csv"
    choice.write_text(HEADER + rows)
    return choice, config_dir


def run(choice: Path, config_dir: Path) -> int:
    return make_configs.main(["--choice", str(choice), "--config-dir", str(config_dir)])


def names(config_dir: Path) -> list[str]:
    return sorted(p.name for p in config_dir.glob("rules_*.yaml"))


def test_go_writes_gated_for_both_models_and_ungated_for_tiny(tmp_path: Path) -> None:
    choice, config_dir = setup(
        tmp_path, "tiny,vit_anchored,0.4,off,5,3,1,True\nlarge,vit_anchored,0.4,off,5,3,1,True\n"
    )
    assert run(choice, config_dir) == 0
    assert names(config_dir) == [
        "rules_gated_vit_anchored_large.yaml",
        "rules_gated_vit_anchored_tiny.yaml",
        "rules_vit_anchored_tiny.yaml",
    ]
    gated = load_recovery_config(config_dir / "rules_gated_vit_anchored_large.yaml")
    assert isinstance(gated, GatedRunConfig)
    assert gated.method == "fix_gated_rules_vit_anchored_large"
    assert gated.box_source == "vit_anchored" and gated.tracker_mode == ""
    assert gated.gate_config == "configs/recovery/box_gate_large.yaml"
    plain = load_recovery_config(config_dir / "rules_vit_anchored_tiny.yaml")
    assert not isinstance(plain, GatedRunConfig) and plain.method == "fix_rules_vit_anchored_tiny"


def test_no_go_or_missing_file_writes_nothing(tmp_path: Path) -> None:
    choice, config_dir = setup(tmp_path, "tiny,csrt_reinit,off,off,0,0,0,False\n")
    with pytest.raises(SystemExit, match="no model is a go"):
        run(choice, config_dir)
    with pytest.raises(SystemExit, match="is missing"):
        run(tmp_path / "none.csv", config_dir)
    assert names(config_dir) == []


def test_only_the_go_model_is_written(tmp_path: Path) -> None:
    choice, config_dir = setup(
        tmp_path, "tiny,motion,off,0.5,5,3,1,False\nlarge,motion,off,0.5,5,3,1,True\n"
    )
    assert run(choice, config_dir) == 0
    assert names(config_dir) == ["rules_gated_motion_large.yaml"]


def test_csrt_arm_reuses_the_existing_ungated_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    choice, config_dir = setup(tmp_path, "tiny,csrt_continuous,off,0.5,5,3,1,True\n")
    assert run(choice, config_dir) == 0
    assert names(config_dir) == ["rules_gated_csrt_continuous_tiny.yaml"]
    assert "rules_csrt_continuous_tiny.yaml" in capsys.readouterr().out
    gated = load_recovery_config(config_dir / "rules_gated_csrt_continuous_tiny.yaml")
    assert gated.box_source == "csrt" and gated.tracker_mode == "continuous"
    assert (
        make_configs.ungated_method("csrt_continuous", "tiny") == "fix_continuous_rules_csrt_tiny"
    )
    assert make_configs.ungated_method("csrt_reinit", "tiny") == "fix_rules_csrt_tiny"


def test_overwrite_with_different_content_is_refused_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    choice, config_dir = setup(tmp_path, "tiny,vit_reinit,0.4,off,5,3,1,True\n")
    (config_dir / "rules_vit_reinit_tiny.yaml").write_text("kind: recovery_run\n")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        run(choice, config_dir)
    assert not (config_dir / "rules_gated_vit_reinit_tiny.yaml").exists()


def test_rerun_with_same_content_is_fine(tmp_path: Path) -> None:
    choice, config_dir = setup(tmp_path, "tiny,motion,off,0.5,5,3,1,True\n")
    assert run(choice, config_dir) == 0
    assert run(choice, config_dir) == 0


def test_missing_gate_file_writes_nothing(tmp_path: Path) -> None:
    choice, config_dir = setup(tmp_path, "tiny,motion,off,0.5,5,3,1,True\n", gates=())
    with pytest.raises(SystemExit, match="box_gate_tiny.yaml"):
        run(choice, config_dir)
    assert names(config_dir) == []


@pytest.mark.parametrize("arm", make_configs.ARMS)
@pytest.mark.parametrize("model", ["tiny", "large"])
def test_every_written_config_loads_and_passes_the_run_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str, model: str
) -> None:
    choice, config_dir = setup(tmp_path, f"{model},{arm},0.4,off,5,3,1,True\n")
    monkeypatch.setattr(recovery_config, "ROOT", tmp_path)
    assert run(choice, config_dir) == 0
    for path in config_dir.glob("rules_*.yaml"):
        cfg = load_recovery_config(path)
        assert cfg.detector == "rules" and cfg.box_source != "oracle"
        assert run_online.model_of(cfg) == model
        run_online.check_online_method(cfg.method)
        gate = run_online.choose_gate(cfg)
        assert (gate is not None) == isinstance(cfg, GatedRunConfig)
        if gate is not None:
            assert gate.s_min == 0.4 and gate.a_max is None


def test_list_names_the_runs_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    choice, config_dir = setup(
        tmp_path, "tiny,csrt_reinit,off,0.5,5,3,1,True\nlarge,vit_reinit,0.4,off,5,3,1,True\n"
    )
    assert (
        make_configs.main(["--choice", str(choice), "--config-dir", str(config_dir), "--list"]) == 0
    )
    lines = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert [(m, v, Path(p).name, s) for m, v, p, s in lines] == [
        ("tiny", "a", "rules_csrt_tiny.yaml", "existing"),
        ("tiny", "b", "rules_gated_csrt_reinit_tiny.yaml", "new"),
        ("large", "b", "rules_gated_vit_reinit_large.yaml", "new"),
    ]
    assert names(config_dir) == []
