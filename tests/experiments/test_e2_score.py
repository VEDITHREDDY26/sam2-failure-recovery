from pathlib import Path

import pandas as pd
import pytest

from experiments.e2 import score
from src.eval.rows import read_scores_csv
from tests.eval.sam2_trees import build_world, run_all
from tests.experiments.e2_trees import (
    METHOD,
    SPLIT,
    blank_object,
    make_online_run,
    write_events,
)

CONFIG = "configs/recovery/rules_csrt_tiny.yaml"


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset, models=("tiny",), splits=("davis_val", "sav_test"))
    return root / "outputs", subset


def _score(subset: Path, results: Path, *extra: str, config: str = CONFIG) -> int:
    args = ["--config", config, "--subset-file", str(subset), "--results-dir", str(results)]
    return score.main([*args, *extra])


def test_frames_recovered_and_broken_are_counted_per_object(
    world: tuple[Path, Path], tmp_path: Path
) -> None:
    out, subset = world
    make_online_run(out)
    blank_object(out, "sam2_tiny", SPLIT, "zebra", "00003", 2)
    blank_object(out, METHOD, SPLIT, "bear", "00005", 1)
    for method in ("sam2_tiny", METHOD):
        blank_object(out, method, SPLIT, "zebra", "00002", 1)
    write_events(out, METHOD, SPLIT, "zebra", [(2, 3, "reprompted"), (1, 9, "rejected")])
    results = tmp_path / "results"
    assert _score(subset, results, "--splits", "davis_val") == 0
    frames = pd.read_csv(results / f"{METHOD}_{SPLIT}_frames.csv")
    assert list(frames.columns) == list(score.FRAME_COLUMNS)
    expected = [
        ("bear", 1, 20, 0, 1, 0, 1),
        ("bear", 2, 20, 0, 0, 0, 0),
        ("zebra", 1, 20, 1, 1, 0, 0),
        ("zebra", 2, 20, 1, 0, 1, 0),
    ]
    assert [tuple(r) for r in frames.itertuples(index=False)] == expected
    events = pd.read_csv(results / f"events_{METHOD}_{SPLIT}.csv")
    assert list(events["action"]) == ["reprompted", "rejected"]
    timing = pd.read_csv(results / f"{METHOD}_timing_{SPLIT}.csv")
    assert {"restarts", "restart_s"} <= set(timing.columns) and len(timing) == 2
    rows = read_scores_csv(results / f"{METHOD}_{SPLIT}.csv")
    assert [(r.video, r.object) for r in rows] == [
        (v, o) for v in ("bear", "zebra") for o in (1, 2)
    ]
    assert not any(p.name.startswith(("sam2_", "baseline")) for p in results.iterdir())


def test_incomplete_run_is_refused_and_nothing_is_written(
    world: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, subset = world
    make_online_run(out)
    (out / METHOD / SPLIT / "events" / "zebra.csv").unlink()
    results = tmp_path / "results"
    assert _score(subset, results, "--splits", "davis_val") == 1
    assert "missing zebra" in capsys.readouterr().out
    assert not results.exists()


def test_sav_test_is_scored_on_the_subset(world: tuple[Path, Path], tmp_path: Path) -> None:
    out, subset = world
    make_online_run(out, split="sav_test")
    results = tmp_path / "results"
    assert _score(subset, results, "--splits", "sav_test") == 0
    videos = {r.video for r in read_scores_csv(results / f"{METHOD}_sav_test.csv")}
    assert videos == {"vid_a", "vid_b"}


def test_videos_option_scores_only_those_videos(world: tuple[Path, Path], tmp_path: Path) -> None:
    out, subset = world
    make_online_run(out)
    results = tmp_path / "results"
    assert _score(subset, results, "--splits", "davis_val", "--videos", "bear") == 0
    assert {r.video for r in read_scores_csv(results / f"{METHOD}_{SPLIT}.csv")} == {"bear"}


def test_sav_val_is_not_an_e2_split(world: tuple[Path, Path], tmp_path: Path) -> None:
    _, subset = world
    with pytest.raises(SystemExit) as stop:
        _score(subset, tmp_path / "results", "--splits", "sav_val")
    assert stop.value.code == 2


@pytest.mark.parametrize("method", ["sam2_tiny", "baseline_csrt_grabcut"])
def test_sam2_and_baseline_output_names_are_refused(method: str) -> None:
    with pytest.raises(ValueError, match="refusing"):
        score.output_names(method, SPLIT)


def test_sam2_config_is_refused(world: tuple[Path, Path], tmp_path: Path) -> None:
    _, subset = world
    config = tmp_path / "sam2_tiny.yaml"
    config.write_text("kind: recovery_run\nmethod: sam2_tiny\ndetector: rules\nbox_source: csrt\n")
    with pytest.raises(ValueError, match="SAM 2 alone"):
        _score(subset, tmp_path / "results", config=str(config))


def test_oracle_config_is_refused(world: tuple[Path, Path], tmp_path: Path) -> None:
    _, subset = world
    with pytest.raises(SystemExit, match="real system only"):
        _score(subset, tmp_path / "results", config="configs/oracle/oracle_flags_csrt_tiny.yaml")


def test_frame_counts_demand_the_same_labelled_frames() -> None:
    keys = {"video": ["v"], "object_id": [1], "frame_index": [0]}
    alone = pd.DataFrame({**keys, "failure": [True]})
    fix = pd.DataFrame({**keys, "frame_index": [1], "failure": [False]})
    with pytest.raises(ValueError, match="same frames"):
        score.frame_counts(alone, fix)
