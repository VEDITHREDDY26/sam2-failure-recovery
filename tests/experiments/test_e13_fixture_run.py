"""End to end on a tiny synthetic video: ViT through the online runner, then the identity check."""

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from experiments.e2 import run_online
from experiments.e13 import check_gated
from src.common.layout import events_path
from src.recovery.config import load_recovery_config
from src.recovery.vit_tracker import load_vit_config
from tests.recovery.fakes import NUM_FRAMES, H, StatefulFake, W
from tests.recovery.online_helpers import SPLIT, START_BOX, VIDEO, make_tree

MODEL_ROOT = os.environ.get("SAM2FR_DATA_ROOT", "")
QUIET = (
    "{score_threshold: -100.0, area_ratio: 100.0, jump_ratio: 100.0,"
    " min_area_fraction: 0.5, border_margin_px: 8}"
)
ALWAYS = (
    "{score_threshold: 100.0, area_ratio: 100.0, jump_ratio: 100.0,"
    " min_area_fraction: 0.5, border_margin_px: 8}"
)
PLAIN = (
    "kind: recovery_run\nmethod: {method}\ndetector: rules\nbox_source: vit_reinit\n"
    'tracker_mode: ""\ndisabled_rules: []\n'
)
GATED = (
    "kind: gated_recovery_run\nmethod: {method}\ndetector: rules\nbox_source: vit_anchored\n"
    'tracker_mode: ""\ndisabled_rules: []\ngate_config: {gate}\n'
)


def draw_frames(frames: Path) -> None:
    """The square the fake predictor segments, moving one pixel per frame, on a textured ground."""
    rng = np.random.default_rng(0)
    background = rng.integers(200, 230, size=(H, W, 3), dtype=np.uint8)
    for index in range(NUM_FRAMES):
        frame = background.copy()
        frame[START_BOX.y0 : START_BOX.y1, START_BOX.x0 + index : START_BOX.x1 + index] = (
            30,
            60,
            200,
        )
        cv2.imwrite(str(frames / f"{index:05d}.jpg"), frame)


def run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompts: Path, config: str, rules: str
) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(f"kind: rules\ntiny: {rules}\nlarge: {rules}\n")
    monkeypatch.setattr(run_online, "build_predictor", lambda *a, **k: StatefulFake())
    monkeypatch.setattr(run_online, "DEFAULT_RULES_PATH", path)
    run_online.main(["--config", config, "--prompts", str(prompts)])


def test_gated_vit_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = Path(MODEL_ROOT) / load_vit_config().model if MODEL_ROOT else None
    if model is None or not model.is_file():
        pytest.skip("ViT model file not present under SAM2FR_DATA_ROOT")
    cfg, _ = make_tree(tmp_path, monkeypatch)
    draw_frames(cfg.data_root / "DAVIS" / "JPEGImages" / "480p" / VIDEO)
    target = cfg.data_root / load_vit_config().model
    target.parent.mkdir(parents=True)
    shutil.copy(model, target)
    prompts = tmp_path / "prompts.json"
    entry = {"frame_index": 0, "boxes": {"1": [10, 20, 22, 32]}}
    prompts.write_text(json.dumps({"split": SPLIT, "videos": {VIDEO: entry}}))
    gate = tmp_path / "gate.yaml"
    gate.write_text("kind: box_gate\ns_min: .inf\na_max: 0.5\n")
    configs = {
        "sam2_tiny": PLAIN.format(method="fix_rules_csrt_tiny").replace("vit_reinit", "csrt"),
        "fix_rules_vit_reinit_tiny": PLAIN.format(method="fix_rules_vit_reinit_tiny"),
        "fix_gated_rules_vit_anchored_tiny": GATED.format(
            method="fix_gated_rules_vit_anchored_tiny", gate=gate
        ),
    }
    paths = {}
    for name, text in configs.items():
        paths[name] = tmp_path / f"{name}.yaml"
        paths[name].write_text(text)
    out = cfg.data_root / cfg.output_root
    run(tmp_path, monkeypatch, prompts, str(paths["sam2_tiny"]), QUIET)
    shutil.copytree(out / "fix_rules_csrt_tiny", out / "sam2_tiny")
    run(tmp_path, monkeypatch, prompts, str(paths["fix_rules_vit_reinit_tiny"]), ALWAYS)
    run(tmp_path, monkeypatch, prompts, str(paths["fix_gated_rules_vit_anchored_tiny"]), ALWAYS)

    plain = pd.read_csv(events_path(out, "fix_rules_vit_reinit_tiny", SPLIT, VIDEO))
    gated = pd.read_csv(events_path(out, "fix_gated_rules_vit_anchored_tiny", SPLIT, VIDEO))
    assert (plain["action"] == "reprompted").sum() >= 1
    assert (gated["action"] == "gated").sum() >= 1 and (gated["action"] == "reprompted").sum() == 0
    assert set(gated.loc[gated["action"] == "gated", "reason"]) == {"agrees_with_sam2"}

    stems = [f"{i:05d}" for i in range(NUM_FRAMES)]
    method = "fix_gated_rules_vit_anchored_tiny"
    got = check_gated.check_split(out, method, "sam2_tiny", SPLIT, [VIDEO], {VIDEO: stems})
    assert got["identical"].all() and got["reprompts"].tolist() == [0]
    assert got["compared_frames"].tolist() == [NUM_FRAMES]
    assert load_recovery_config(paths[method]).box_source == "vit_anchored"
