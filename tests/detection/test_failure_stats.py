from pathlib import Path

import pandas as pd
import pytest

from experiments.sam2_runs import failure_stats
from src.common.layout import mask_path
from tests.eval.sam2_trees import build_world, run_all


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset, models=("tiny",))
    return root, subset


def _stats(subset: Path, out: Path, *extra: str) -> int:
    return failure_stats.main(
        ["--subset-file", str(subset), "--out", str(out), "--models", "tiny", *extra]
    )


def test_counts_on_the_fake_tree_match_hand_numbers(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, subset = world
    out = tmp_path / "results" / "failure_stats.csv"
    assert _stats(subset, out) == 0
    table = pd.read_csv(out).set_index("split")
    assert list(pd.read_csv(out).columns) == list(failure_stats.COLUMNS)
    # DAVIS: 2 videos x 2 objects x 20 frames = 80 labelled frames, the fake run is perfect
    davis = table.loc["davis_val"]
    assert (davis.videos, davis.objects, davis.labelled_frames) == (2, 4, 80)
    assert (davis.failure_frames, davis.objects_with_failure, davis.videos_with_failure) == (
        0,
        0,
        0,
    )
    # SA-V val: 2 videos x (5 + 5 + 3 annotated frames) = 26 labelled frames; the untracked
    # late object fails at its 3 annotated frames in each video -> 6 failures, 2 objects
    sav = table.loc["sav_val"]
    assert (sav.videos, sav.objects, sav.skipped_objects, sav.labelled_frames) == (2, 6, 0, 26)
    assert (sav.failure_frames, sav.objects_with_failure, sav.videos_with_failure) == (6, 2, 2)
    assert sav.failure_rate == pytest.approx(6 / 26)
    # SA-V test: only the 2 subset videos count (vid_c is left out): same numbers as val
    test = table.loc["sav_test"]
    assert (test.videos, test.labelled_frames, test.failure_frames) == (2, 26, 6)


def test_incomplete_run_is_refused_and_nothing_is_written(  # type: ignore[no-untyped-def]
    world, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, subset = world
    mask_path(root / "outputs", "sam2_tiny", "davis_val", "zebra", "00003").unlink()
    out = tmp_path / "results" / "failure_stats.csv"
    assert _stats(subset, out) == 1
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()
    text = capsys.readouterr().out
    assert "refused" in text and "zebra" in text
