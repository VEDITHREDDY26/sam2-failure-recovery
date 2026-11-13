import random
from pathlib import Path

import pandas as pd
import pytest

from experiments.e2 import rerun_check
from experiments.e2.rerun_check import SEEDED_VIDEOS
from experiments.e2.table import E2Error
from src.eval.rows import ScoreRow, write_scores_csv
from src.recovery.events import EVENT_COLUMNS

REPO = Path(__file__).resolve().parents[2]


def _events(counts: dict[str, int]) -> pd.DataFrame:
    rows = [
        (v, 1, f, "score", "reprompted", "", "", "", "", "", 0)
        for v, n in counts.items()
        for f in range(n)
    ]
    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


def write_run(results: Path, name: str, jf: dict[str, float], counts: dict[str, int]) -> None:
    rows = [ScoreRow(v, 1, x, x) for v, x in jf.items()]
    write_scores_csv(results / f"{name}.csv", rows)
    _events(counts).to_csv(results / f"events_{name}.csv", index=False)


BASE = dict(zip(SEEDED_VIDEOS, (0.5, 0.6, 0.7, 0.8, 0.9), strict=True))
COUNTS = dict(zip(SEEDED_VIDEOS, (1, 0, 2, 3, 1), strict=True))


def run_pair(
    results: Path, shift: dict[str, float] | None = None, counts: dict[str, int] | None = None
) -> list[rerun_check.VideoAgreement]:
    shifted = {v: x + (shift or {}).get(v, 0.0) for v, x in BASE.items()}
    write_run(results, rerun_check.E2_NAME, BASE, COUNTS)
    write_run(results, rerun_check.RERUN_NAME, shifted, counts or COUNTS)
    return rerun_check.compare(results)


def test_seeded_videos_are_rederived_from_the_davis_val_list() -> None:
    names = sorted(set(pd.read_csv(REPO / "results" / "sam2_tiny_davis_val.csv")["video"]))
    assert len(names) == 30
    assert tuple(
        sorted(random.Random(rerun_check.SEED).sample(names, rerun_check.SAMPLE_SIZE))
    ) == (SEEDED_VIDEOS)


def test_identical_runs_pass(tmp_path: Path) -> None:
    rows = run_pair(tmp_path)
    assert rerun_check.passes(rows) and rerun_check.mean_abs_delta(rows) == 0.0


def test_per_video_boundary_is_inclusive(tmp_path: Path) -> None:
    rows = run_pair(tmp_path, shift={"cows": 0.01})
    cows = next(r for r in rows if r.video == "cows")
    assert cows.within
    rows = run_pair(tmp_path, shift={"cows": 0.0101})
    assert not next(r for r in rows if r.video == "cows").within
    assert not rerun_check.passes(rows)


def test_mean_tolerance_fails_even_when_every_video_is_within(tmp_path: Path) -> None:
    shift = {v: 0.008 for v in SEEDED_VIDEOS}
    rows = run_pair(tmp_path, shift=shift)
    assert all(r.within for r in rows)
    assert rerun_check.mean_abs_delta(rows) == pytest.approx(0.008)
    assert not rerun_check.passes(rows)


def test_mean_boundary_passes(tmp_path: Path) -> None:
    rows = run_pair(tmp_path, shift={v: 0.005 for v in SEEDED_VIDEOS})
    assert rerun_check.passes(rows)


def test_different_reprompt_counts_fail(tmp_path: Path) -> None:
    rows = run_pair(tmp_path, counts={**COUNTS, "dog": 3})
    dog = next(r for r in rows if r.video == "dog")
    assert (dog.reprompts_e2, dog.reprompts_rerun, dog.within) == (2, 3, False)
    assert not rerun_check.passes(rows)


def test_main_writes_the_table_and_sets_the_exit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_pair(tmp_path)
    out = tmp_path / "check.csv"
    assert rerun_check.main(["--results-dir", str(tmp_path), "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert list(table.columns) == list(rerun_check.CHECK_COLUMNS) and len(table) == 5
    assert "within tolerance" in capsys.readouterr().out
    run_pair(tmp_path, shift={"dog": 0.05})
    assert rerun_check.main(["--results-dir", str(tmp_path), "--out", str(out)]) == 1
    assert "OUTSIDE" in capsys.readouterr().out


def test_missing_video_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path, rerun_check.E2_NAME, BASE, COUNTS)
    write_run(tmp_path, rerun_check.RERUN_NAME, {"cows": 0.6}, {})
    with pytest.raises(E2Error, match="missing"):
        rerun_check.compare(tmp_path)
