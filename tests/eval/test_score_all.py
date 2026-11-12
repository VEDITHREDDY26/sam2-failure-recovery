from pathlib import Path

import pytest

from experiments.sam2_runs import score_all
from src.common.layout import mask_path
from src.eval.rows import read_scores_csv
from tests.eval.sam2_trees import build_world, run_all


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset, models=("tiny",))
    return root, subset


def _score(subset: Path, results: Path, *extra: str) -> int:
    args = ["--subset-file", str(subset), "--results-dir", str(results), "--models", "tiny", *extra]
    return score_all.main(args)


def test_complete_runs_are_scored_and_timing_collected(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, subset = world
    results = tmp_path / "results"
    assert _score(subset, results) == 0
    assert sorted(p.name for p in results.iterdir()) == sorted(
        f"sam2_tiny_{s}.csv" for s in ("davis_val", "sav_val", "sav_test")
    ) + sorted(f"sam2_tiny_timing_{s}.csv" for s in ("davis_val", "sav_val", "sav_test"))
    assert (
        results.joinpath("sam2_tiny_davis_val.csv").read_text().splitlines()[0]
        == "video,object,J,F,J&F"
    )
    rows = read_scores_csv(results / "sam2_tiny_davis_val.csv")
    assert [(r.video, r.object) for r in rows] == [
        (v, o) for v in ("bear", "zebra") for o in (1, 2)
    ]
    assert all(r.j == pytest.approx(1.0) and r.f == pytest.approx(1.0) for r in rows)
    timing = results / "sam2_tiny_timing_sav_val.csv"
    assert (
        timing.read_text().splitlines()[0].startswith("video,frames,objects,compute_s,wall_s,fps")
    )


def test_sav_test_is_restricted_to_the_subset(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, subset = world
    results = tmp_path / "results"
    assert _score(subset, results, "--splits", "sav_test") == 0
    videos = {r.video for r in read_scores_csv(results / "sam2_tiny_sav_test.csv")}
    assert videos == {"vid_a", "vid_b"}


def test_incomplete_run_is_refused_and_nothing_is_written(  # type: ignore[no-untyped-def]
    world, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, subset = world
    mask_path(root / "outputs", "sam2_tiny", "davis_val", "zebra", "00003").unlink()
    results = tmp_path / "results"
    assert _score(subset, results, "--splits", "davis_val", "sav_val") == 1
    out = capsys.readouterr().out
    assert "sam2_tiny/davis_val: refused, run is not complete" in out and "missing zebra" in out
    assert not (results / "sam2_tiny_davis_val.csv").exists()
    assert not (results / "sam2_tiny_timing_davis_val.csv").exists()
    assert (results / "sam2_tiny_sav_val.csv").is_file()
