import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.e2 import pilot_check
from src.common.layout import events_path, mask_path
from src.common.masks import read_label_map, write_label_map
from src.recovery.box_check import BoxCheck
from src.recovery.events import EVENT_COLUMNS
from src.recovery.online_runner import run_batch
from src.recovery.settings import RecoverySettings
from tests.recovery.fakes import NUM_FRAMES, ScriptedBoxes, ScriptedDetector, StatefulFake, ok
from tests.recovery.online_helpers import SPLIT, VIDEO, make_tree
from tests.recovery.test_online_runner import NEW_BOX

REAL_CONFIG = "configs/recovery/rules_csrt_tiny.yaml"
METHOD = "fix_rules_csrt_tiny"
REFERENCE = "sam2_tiny"
FLIP_AT = 5


def stem(index: int) -> str:
    return f"{index:05d}"


def run(cfg, prompts, method, flags=None) -> None:  # type: ignore[no-untyped-def]
    run_batch(
        prompts,
        cfg,
        method,
        StatefulFake,
        lambda video: ScriptedDetector(flags),
        lambda video: ScriptedBoxes({(1, FLIP_AT): ok(NEW_BOX)}),
        BoxCheck(),
        RecoverySettings(12, "reinit"),
    )


@pytest.fixture
def pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    cfg, prompts = make_tree(tmp_path, monkeypatch)
    out = cfg.data_root / cfg.output_root
    # the cached SAM 2 alone masks are what the same fake gives without any recovery
    run(cfg, prompts, "fix_reference")
    shutil.copytree(out / "fix_reference", out / REFERENCE)
    run(cfg, prompts, METHOD, {(1, FLIP_AT): ("low_score",)})
    path = tmp_path / "prompts.json"
    entry = {"frame_index": 0, "boxes": {"1": [10, 20, 22, 32]}}
    path.write_text(json.dumps({"split": SPLIT, "videos": {VIDEO: entry}}))
    return out, path, tmp_path / "results" / "events_pilot.csv"


def go(path: Path, out_csv: Path) -> int:
    return pilot_check.main(
        ["--config", REAL_CONFIG, "--prompts", str(path), "--out", str(out_csv)]
    )


def test_identical_frames_before_the_first_reprompt_pass(pilot, capsys) -> None:  # type: ignore[no-untyped-def]
    out, prompts, out_csv = pilot
    assert go(prompts, out_csv) == 0
    check = pd.read_csv(out_csv.with_name("events_pilot_check.csv"))
    row = check.iloc[0]
    assert (row["verdict"], row["first_reprompt"], row["compared_frames"]) == ("identical", 5, 5)
    events = pd.read_csv(out_csv)
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert events["action"].tolist() == ["reprompted"]


def test_a_differing_frame_before_the_first_reprompt_fails(pilot, capsys) -> None:  # type: ignore[no-untyped-def]
    out, prompts, out_csv = pilot
    target = mask_path(out, REFERENCE, SPLIT, VIDEO, stem(3))
    label = read_label_map(target)
    label[0, 0] = 1 - label[0, 0] if label[0, 0] in (0, 1) else 0
    write_label_map(target, label)
    assert go(prompts, out_csv) == 1
    row = pd.read_csv(out_csv.with_name("events_pilot_check.csv")).iloc[0]
    assert (row["verdict"], row["first_diff"]) == ("differs", 3)
    assert f"{VIDEO}: differs at frame 3" in capsys.readouterr().out


def test_difference_at_or_after_the_first_reprompt_is_allowed(pilot) -> None:  # type: ignore[no-untyped-def]
    out, prompts, out_csv = pilot
    target = mask_path(out, REFERENCE, SPLIT, VIDEO, stem(FLIP_AT))
    write_label_map(target, np.zeros_like(read_label_map(target)))
    assert go(prompts, out_csv) == 0


def test_run_without_reprompt_must_match_everywhere(pilot) -> None:  # type: ignore[no-untyped-def]
    out, prompts, out_csv = pilot
    events_path(out, METHOD, SPLIT, VIDEO).write_text(",".join(EVENT_COLUMNS) + "\n")
    assert go(prompts, out_csv) == 1
    row = pd.read_csv(out_csv.with_name("events_pilot_check.csv")).iloc[0]
    assert (row["verdict"], row["first_diff"]) == ("differs", FLIP_AT)
    assert row["compared_frames"] == FLIP_AT < NUM_FRAMES


def test_missing_online_run_is_an_error(pilot) -> None:  # type: ignore[no-untyped-def]
    out, prompts, out_csv = pilot
    events_path(out, METHOD, SPLIT, VIDEO).unlink()
    with pytest.raises(FileNotFoundError):
        go(prompts, out_csv)
