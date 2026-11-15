import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from experiments.verify import pilot_check
from src.common.layout import mask_path
from src.common.masks import read_label_map, write_label_map
from src.recovery.box_check import BoxCheck
from src.recovery.online_runner import run_batch
from src.recovery.settings import RecoverySettings
from src.recovery.verify import VerifyThresholds, candidate_path
from tests.recovery.fakes import ScriptedBoxes, ScriptedDetector, StatefulFake, ok
from tests.recovery.online_helpers import SPLIT, VIDEO, make_tree
from tests.recovery.test_online_runner import NEW_BOX

CONFIG = "configs/recovery/verified_rules_csrt_tiny.yaml"
METHOD = "fix_verified_rules_csrt_tiny"
STRICT = VerifyThresholds(5.0, math.inf, -math.inf, False)
LENIENT = VerifyThresholds(-math.inf, math.inf, -math.inf, False)


def run(cfg: Any, prompts: Any, method: str, flags: Any, verifier: Any) -> None:
    run_batch(
        prompts,
        cfg,
        method,
        StatefulFake,
        lambda video: ScriptedDetector(flags),
        lambda video: ScriptedBoxes({(1, 5): ok(NEW_BOX)}),
        BoxCheck(),
        RecoverySettings(12, "reinit"),
        verifier=verifier,
        log_candidates=verifier is not None,
    )


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verifier: Any) -> tuple[Path, Path]:
    cfg, prompts = make_tree(tmp_path, monkeypatch)
    out = cfg.data_root / cfg.output_root
    run(cfg, prompts, "fix_reference", {}, None)
    shutil.copytree(out / "fix_reference", out / "sam2_tiny")
    run(cfg, prompts, METHOD, {(1, 5): ("low_score",)}, verifier)
    path = tmp_path / "prompts.json"
    entry = {"frame_index": 0, "boxes": {"1": [10, 20, 22, 32]}}
    path.write_text(json.dumps({"split": SPLIT, "videos": {VIDEO: entry}}))
    return out, path


def go(prompts: Path, out_csv: Path) -> int:
    return pilot_check.main(["--config", CONFIG, "--prompts", str(prompts), "--out", str(out_csv)])


def test_all_rejected_video_equals_sam2_alone_on_every_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, prompts = setup(tmp_path, monkeypatch, STRICT)
    out_csv = tmp_path / "check.csv"
    assert go(prompts, out_csv) == 0
    row = pd.read_csv(out_csv).iloc[0]
    assert (row["accepted"], row["rejected"], bool(row["all_rejected"])) == (0, 1, True)
    assert row["compared_frames"] == row["frames"] == 30 and row["verdict"] == "identical"


def test_all_rejected_video_that_differs_anywhere_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, prompts = setup(tmp_path, monkeypatch, STRICT)
    target = mask_path(out, "sam2_tiny", SPLIT, VIDEO, "00029")
    label = read_label_map(target)
    label[0, 0] = 1 - int(label[0, 0]) if label[0, 0] in (0, 1) else 0
    write_label_map(target, label)
    out_csv = tmp_path / "check.csv"
    assert go(prompts, out_csv) == 1
    assert pd.read_csv(out_csv).iloc[0]["first_diff"] == 29


def test_accepted_video_is_compared_only_before_its_first_reprompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, prompts = setup(tmp_path, monkeypatch, LENIENT)
    out_csv = tmp_path / "check.csv"
    assert go(prompts, out_csv) == 0
    row = pd.read_csv(out_csv).iloc[0]
    assert (row["accepted"], row["rejected"], row["compared_frames"]) == (1, 0, 5)


def test_a_missing_candidate_log_fails_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, prompts = setup(tmp_path, monkeypatch, STRICT)
    candidate_path(out, METHOD, SPLIT, VIDEO).unlink()
    out_csv = tmp_path / "check.csv"
    assert go(prompts, out_csv) == 1
    assert pd.read_csv(out_csv).iloc[0]["verdict"] == "candidate_log_mismatch"
