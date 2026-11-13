from pathlib import Path

import pandas as pd
import pytest

from experiments.e9 import time_components as e9
from src.common.boxes import Box
from src.recovery.box_check import BoxCheck, load_box_check_config
from src.recovery.settings import RecoverySettings
from src.sam2_runner.prompts import parse_batch_prompts
from src.sam2_runner.smoke import load_data_config
from tests.recovery.fakes import NUM_FRAMES, ScriptedBoxes, ScriptedDetector, StatefulFake, ok
from tests.recovery.online_helpers import VIDEO, make_tree

NEW_BOX = Box(30, 20, 42, 32)
PROMPTS = {
    "split": "davis_val",
    "videos": {VIDEO: {"frame_index": 0, "boxes": {"1": [10, 20, 22, 32]}}},
}


def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repeats: int, flags: bool) -> tuple:
    cfg, prompts = (
        make_tree(tmp_path, monkeypatch)
        if not (tmp_path / "drive").exists()
        else (load_data_config(), parse_batch_prompts(PROMPTS))
    )
    made = {"predictors": 0}
    fake = StatefulFake()

    def predictor() -> StatefulFake:
        made["predictors"] += 1
        return fake

    rows = e9.measure_model(
        "tiny",
        prompts,
        cfg,
        predictor,
        lambda: ScriptedDetector({(1, 5): ("low_score",)} if flags else {}),
        lambda: ScriptedBoxes({(1, 5): ok(NEW_BOX)}),
        BoxCheck(load_box_check_config()),
        RecoverySettings(12, "reinit"),
        [VIDEO],
        repeats,
    )
    return rows, fake, made


def test_components_counts_and_spread_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, _, made = run(tmp_path, monkeypatch, repeats=3, flags=True)
    table = pd.DataFrame(rows, columns=list(e9.COLUMNS))
    assert table["component"].tolist() == list(e9.COMPONENTS)
    by = table.set_index("component")
    assert made["predictors"] == 1
    assert (table["repeats"] == 3).all() and (table["frames"] == NUM_FRAMES).all()
    assert by.loc["sam2_propagate", "calls"] == 6 + (NUM_FRAMES - 5)
    assert by.loc["reprompt_restart", "calls"] == 1
    assert by.loc["init_state", "calls"] == 1
    assert by.loc["sam2_prompt", "calls"] == 3
    assert by.loc["detector", "calls"] > NUM_FRAMES
    assert by.loc["total_loop", "fps_mean"] > 0
    assert (table["ms_per_frame_std"] >= 0).all()
    assert by["reprompts"].iloc[0] == 1
    assert list(table.columns)[-3:] == ["gpu_name", "torch_version", "cuda_version"]


def test_warm_up_pass_is_not_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows, fake, _ = run(tmp_path, monkeypatch, repeats=2, flags=False)
    assert sum(c == ("init",) for c in fake.calls) == 3
    assert rows[0]["repeats"] == 2
    one, _, _ = run(tmp_path, monkeypatch, repeats=1, flags=False)
    assert one[0]["ms_per_frame_std"] == 0.0


def test_nothing_is_written_to_the_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(tmp_path, monkeypatch, repeats=1, flags=True)
    assert not (tmp_path / "drive" / "outputs").exists()


def test_sample_is_seeded_and_sorted() -> None:
    names = [f"v{i:02d}" for i in range(30)]
    assert e9.sample_videos(names) == e9.sample_videos(names[::-1])
    assert len(e9.sample_videos(names)) == 3
    with pytest.raises(ValueError):
        e9.sample_videos(names[:2])


def test_refuses_without_frozen_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.detection.rules as rules

    monkeypatch.setattr(rules, "DEFAULT_RULES_PATH", tmp_path / "none.yaml")
    monkeypatch.setattr(e9, "build_predictor", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(SystemExit, match="not frozen"):
        e9.main(["--prompts", str(tmp_path / "p.json"), "--out", str(tmp_path / "e9.csv")])
    assert not (tmp_path / "e9.csv").exists()
