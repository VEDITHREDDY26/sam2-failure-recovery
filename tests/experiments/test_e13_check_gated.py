from pathlib import Path

import pandas as pd
import pytest

from experiments.e13 import check_gated
from tests.eval.sam2_trees import build_world, run_all
from tests.experiments.e2_trees import blank_object, make_online_run, write_events

SPLIT = "davis_val"
METHOD = "fix_gated_rules_vit_anchored_tiny"
CONFIG = (
    f"kind: gated_recovery_run\nmethod: {METHOD}\ndetector: rules\nbox_source: vit_anchored\n"
    'tracker_mode: ""\ndisabled_rules: []\ngate_config: configs/recovery/box_gate_tiny.yaml\n'
)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset, models=("tiny",), splits=("davis_val", "sav_test"))
    config = tmp_path / "gated.yaml"
    config.write_text(CONFIG)
    make_online_run(root / "outputs", method=METHOD)
    return root / "outputs", subset, config


def go(subset: Path, config: Path, results: Path, split: str = SPLIT) -> int:
    args = ["--config", str(config), "--subset-file", str(subset), "--results-dir", str(results)]
    return check_gated.main([*args, "--splits", split])


def table(results: Path, split: str = SPLIT) -> pd.DataFrame:
    return pd.read_csv(results / check_gated.check_name(METHOD, split))


def test_untouched_videos_are_identical_and_counted(
    world: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    out, subset, config = world
    write_events(out, METHOD, SPLIT, "zebra", [(1, 3, "gated"), (2, 9, "gated")])
    results = tmp_path / "results"
    assert go(subset, config, results) == 0
    raw = table(results)
    assert list(raw.columns) == list(check_gated.CHECK_COLUMNS)
    got = raw.set_index("video")
    assert got["gated"].to_dict() == {"bear": 0, "zebra": 2}
    assert got["reprompts"].tolist() == [0, 0] and got["identical"].tolist() == [True, True]
    assert got["compared_frames"].tolist() == [20, 20]


def test_a_changed_frame_in_a_video_without_reprompt_fails(
    world: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    out, subset, config = world
    blank_object(out, METHOD, SPLIT, "bear", "00007", 1)
    results = tmp_path / "results"
    assert go(subset, config, results) == 1
    got = table(results).set_index("video")
    assert not got.loc["bear", "identical"] and got.loc["zebra", "identical"]


def test_after_a_reprompt_only_the_earlier_frames_must_match(
    world: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    out, subset, config = world
    write_events(out, METHOD, SPLIT, "bear", [(1, 5, "reprompted")])
    blank_object(out, METHOD, SPLIT, "bear", "00007", 1)
    assert go(subset, config, tmp_path / "r1") == 0
    got = table(tmp_path / "r1").set_index("video")
    assert got.loc["bear", "reprompts"] == 1 and got.loc["bear", "compared_frames"] == 5
    blank_object(out, METHOD, SPLIT, "bear", "00002", 1)
    assert go(subset, config, tmp_path / "r2") == 1


def test_missing_events_are_refused(
    world: tuple[Path, Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, subset, config = world
    (out / METHOD / SPLIT / "events" / "zebra.csv").unlink()
    results = tmp_path / "results"
    assert go(subset, config, results) == 1
    assert "refused" in capsys.readouterr().out
    assert not results.exists()


def test_oracle_config_is_refused(world: tuple[Path, Path, Path], tmp_path: Path) -> None:
    _, subset, _ = world
    with pytest.raises(SystemExit, match="real system only"):
        go(subset, Path("configs/oracle/oracle_flags_csrt_tiny.yaml"), tmp_path / "r")
