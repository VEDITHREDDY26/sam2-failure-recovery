"""CPU pilot of the whole chain on a fake data tree with the fake predictor (no SAM 2, no GPU).

pytest -q -s tests/eval/test_sam2_pilot.py
"""

from pathlib import Path

import pytest

from experiments.sam2_runs import make_prompts, score_all
from src.eval import check_complete
from src.sam2_runner import run
from src.sam2_runner.timing import timing_path
from tests.eval.sam2_trees import SPLITS, build_world
from tests.fixtures.synthetic_video import make_synthetic_video
from tests.sam2_runner.fakes import FakePredictor


def test_chain_prompts_run_rerun_check_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    source = make_synthetic_video(tmp_path / "source", "source")
    built: list[FakePredictor] = []

    def factory(*args: object, **kwargs: object) -> FakePredictor:
        built.append(FakePredictor(source))
        return built[-1]

    monkeypatch.setattr(run, "build_predictor", factory)
    prompts = root / "outputs" / "prompts"

    for split in SPLITS:
        make_prompts.main(["--split", split, "--subset-file", str(subset)])
    for model in ("tiny", "large"):
        for split in SPLITS:
            run.main(["--model", model, "--prompts", str(prompts / f"{split}.json")])
    first = capsys.readouterr().out
    assert first.count("done,") == 2 * 6 and "skipped (complete)" not in first
    assert len(built) == 6

    for model in ("tiny", "large"):
        for split in SPLITS:
            run.main(["--model", model, "--prompts", str(prompts / f"{split}.json")])
    second = capsys.readouterr().out
    assert second.count("skipped (complete)") == 12 and "done," not in second
    assert len(built) == 6

    assert check_complete.main(["--subset-file", str(subset)]) == 0
    results = tmp_path / "results"
    code = score_all.main(["--subset-file", str(subset), "--results-dir", str(results)])
    assert code == 0 and len(list(results.glob("sam2_*.csv"))) == 12
    assert timing_path(root / "outputs", "sam2_tiny", "sav_val", "vid_a").is_file()
    rest = capsys.readouterr().out
    with capsys.disabled():
        print("\n=== pilot output ===")
        print(first + second + rest)
