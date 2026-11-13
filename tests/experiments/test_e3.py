import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.e2 import run_online
from experiments.e2.table import E2Error
from experiments.e3 import detector_eval, figure, table
from experiments.tune_rules.tune import build_windows
from src.common.layout import signals_path
from src.detection.rules import ALL_RULES, RuleThresholds
from src.eval.rows import ScoreRow, write_scores_csv
from src.recovery.config import load_recovery_config
from src.recovery.events import EVENT_COLUMNS
from tests.experiments.test_run_online import QUIET, write_prompts
from tests.recovery.fakes import StatefulFake
from tests.recovery.online_helpers import SPLIT, VIDEO

CONFIG_DIR = Path("configs/recovery")
CFG = RuleThresholds(
    score_threshold=0.0, area_ratio=2.0, jump_ratio=1.0, min_area_fraction=0.01, border_margin_px=4
)


def frame_row(
    video: str, obj: int, frame: int, score: float, area: int, cx: float, box: tuple[int, ...]
) -> dict[str, object]:
    empty = area == 0
    corners = (None,) * 4 if empty else box
    return {
        "video": video,
        "object_id": obj,
        "frame_index": frame,
        "object_score": score,
        "area": area,
        "centre_x": float("nan") if empty else cx,
        "centre_y": float("nan") if empty else 50.0,
        "height": 100,
        "width": 100,
        **dict(zip(("box_x0", "box_y0", "box_x1", "box_y1"), corners, strict=True)),
    }


def synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two single-object videos annotated at every frame (DAVIS style), 4 frames each.

    a: frame 1 low score (low_score fires), frame 2 area x4 (area_change fires), frame 3 centre
       jump (centre_jump fires), all failures.
    b: frame 1 mask disappears from the middle (disappearance fires), failure; frames 2-3 fine.
    """
    box = (40, 40, 60, 60)
    a = [
        frame_row("a", 1, 0, 5.0, 400, 50.0, box),
        frame_row("a", 1, 1, -3.0, 400, 50.0, box),
        frame_row("a", 1, 2, 5.0, 1600, 50.0, (30, 30, 70, 70)),
        frame_row("a", 1, 3, 5.0, 1600, 95.0, (75, 30, 99, 70)),
    ]
    b = [
        frame_row("b", 1, 0, 5.0, 400, 50.0, box),
        frame_row("b", 1, 1, 5.0, 0, 0.0, ()),
        frame_row("b", 1, 2, 5.0, 400, 50.0, box),
        frame_row("b", 1, 3, 5.0, 400, 50.0, box),
    ]
    table_ = pd.DataFrame(a + b).astype({c: "float64" for c in ("box_x0", "box_y0")})
    fail_a = [False, True, True, True]
    fail_b = [False, True, False, False]
    labels = pd.DataFrame(
        [("a", 1, f, 0.0, x) for f, x in enumerate(fail_a)]
        + [("b", 1, f, 0.0, x) for f, x in enumerate(fail_b)],
        columns=["video", "object_id", "frame_index", "iou", "failure"],
    )
    return table_, labels


def test_davis_windows_are_single_frames_after_the_prompt_frame() -> None:
    table_, labels = synthetic()
    windows = build_windows(table_, labels)
    assert windows.frame_window.tolist() == [-1, 0, 1, 2, -1, 3, 4, 5]
    assert windows.truth.tolist() == [True, True, True, True, False, False]


def test_recall_per_removed_rule_on_a_tiny_set() -> None:
    table_, labels = synthetic()
    rows = {r["removed_rule"]: r for r in detector_eval.evaluate_table(table_, labels, CFG)}
    assert [r for r in rows] == list(detector_eval.REMOVED)
    # 6 windows, 4 positives (a1, a2, a3, b1); all four rules fire once each on a failure
    full = rows["none"]
    assert (full["windows"], full["positives"], full["flagged"], full["tp"]) == (6, 4, 4, 4)
    assert full["recall"] == 1.0 and full["precision"] == 1.0
    for rule in ALL_RULES:
        r = rows[rule]
        assert (r["flagged"], r["tp"]) == (3, 3), rule
        assert r["recall"] == 0.75 and r["precision"] == 1.0, rule


def test_a_false_alarm_lowers_precision_not_recall() -> None:
    table_, labels = synthetic()
    labels.loc[(labels["video"] == "b") & (labels["frame_index"] == 1), "failure"] = False
    full = detector_eval.evaluate_table(table_, labels, CFG)[0]
    assert (full["positives"], full["flagged"], full["tp"]) == (3, 4, 3)
    assert full["precision"] == 0.75 and full["recall"] == 1.0


def test_sav_val_is_refused(tmp_path: Path) -> None:
    with pytest.raises(detector_eval.DetectorEvalError, match="not a test set"):
        detector_eval.run("tiny", ["sav_val"], tmp_path, tmp_path / "s.txt")


def test_missing_rules_file_is_refused_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(tmp_path))
    code = detector_eval.main(
        ["--rules", str(tmp_path / "no.yaml"), "--results-dir", str(tmp_path)]
    )
    assert code == 1 and not list(tmp_path.glob("e3_detector_*"))


def test_signals_boxes_and_labels_come_from_a_run_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.eval.sam2_trees import build_world, run_all

    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset, models=("tiny",), splits=("davis_val",))
    frame, labels = detector_eval.split_inputs("tiny", "davis_val", subset)
    assert set(frame["video"]) == set(labels["video"]) and len(frame) > 0
    assert not frame["box_x0"].isna().all()
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        "kind: rules\n"
        "tiny: {score_threshold: 0.0, area_ratio: 2.0, jump_ratio: 1.0,"
        " min_area_fraction: 0.01, border_margin_px: 4}\n"
        "large: {score_threshold: 0.0, area_ratio: 2.0, jump_ratio: 1.0,"
        " min_area_fraction: 0.01, border_margin_px: 4}\n"
    )
    out = tmp_path / "results"
    assert (
        detector_eval.main(
            [
                "--splits",
                "davis_val",
                "--subset-file",
                str(subset),
                "--results-dir",
                str(out),
                "--rules",
                str(rules),
            ]
        )
        == 0
    )
    got = pd.read_csv(out / "e3_detector_davis_val.csv")
    assert list(got.columns) == list(detector_eval.RESULT_COLUMNS)
    assert got["removed_rule"].tolist() == list(detector_eval.REMOVED)


@pytest.mark.parametrize("rule", ALL_RULES)
def test_ablation_configs_switch_off_exactly_one_rule(rule: str) -> None:
    cfg = load_recovery_config(CONFIG_DIR / f"rules_no_{rule}_csrt_tiny.yaml")
    assert cfg.method == table.method_for(rule)
    assert cfg.disabled_rules == [rule] and cfg.tracker_mode == ""
    assert cfg.enabled_rules == tuple(r for r in ALL_RULES if r != rule)
    assert (cfg.detector, cfg.box_source) == ("rules", "csrt")


def test_run_online_runs_an_ablation_with_the_remaining_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = write_prompts(tmp_path, monkeypatch)
    rules = tmp_path / "rules.yaml"
    rules.write_text(f"kind: rules\ntiny: {QUIET}\nlarge: {QUIET}\n")
    seen: list[tuple[str, ...]] = []
    real = run_online.OnlineRuleDetector.from_config

    def spy(model: str, path: Path, enabled: tuple[str, ...]) -> object:
        seen.append(tuple(enabled))
        return real(model, path, enabled)

    monkeypatch.setattr(run_online, "build_predictor", lambda *a, **k: StatefulFake())
    monkeypatch.setattr(run_online, "DEFAULT_RULES_PATH", rules)
    monkeypatch.setattr(run_online.OnlineRuleDetector, "from_config", staticmethod(spy))
    run_online.main(
        [
            "--config",
            str(CONFIG_DIR / "rules_no_centre_jump_csrt_tiny.yaml"),
            "--prompts",
            str(prompts),
        ]
    )
    assert seen == [("low_score", "area_change", "disappearance")]
    out = tmp_path / "drive" / "outputs"
    assert signals_path(out, "fix_rules_no_centre_jump_csrt_tiny", SPLIT, VIDEO).is_file()


def write_inputs(results: Path) -> None:
    """Scores, events and detector rows with hand-picked numbers for one split."""
    results.mkdir()
    jf = {"sam2_tiny": 0.5, table.method_for("none"): 0.7}
    jf |= {table.method_for(r): 0.7 - 0.1 * i for i, r in enumerate(ALL_RULES, 1)}
    for method, value in jf.items():
        rows = [ScoreRow("v1", 1, value, value), ScoreRow("v2", 1, value, value)]
        write_scores_csv(results / f"{method}_davis_val.csv", rows)
        if method != "sam2_tiny":
            events = [("v1", 1, 3, "score", "reprompted", "", "", "", "", "", 0)]
            pd.DataFrame(events, columns=list(EVENT_COLUMNS)).to_csv(
                results / f"events_{method}_davis_val.csv", index=False
            )
    det = [
        ("tiny", "davis_val", r, 10, 4, 5, 3, 0.6, 0.75 - 0.25 * i)
        for i, r in enumerate(detector_eval.REMOVED)
    ]
    pd.DataFrame(det, columns=list(detector_eval.RESULT_COLUMNS)).to_csv(
        results / "e3_detector_davis_val.csv", index=False
    )


def test_e3_table_deltas_and_detector_columns(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_inputs(results)
    out = tmp_path / "e3.csv"
    table.main(["--results-dir", str(results), "--out", str(out), "--splits", "davis_val"])
    rows = list(csv.DictReader(out.open()))
    assert [r["removed_rule"] for r in rows] == ["none", *ALL_RULES]
    assert list(rows[0]) == list(table.TABLE_COLUMNS)
    full, no_score = rows[0], rows[1]
    assert float(full["delta_vs_full"]) == 0.0
    assert np.isclose(float(full["delta_vs_alone"]), 0.2)
    assert np.isclose(float(no_score["J&F"]), 0.6) and np.isclose(
        float(no_score["delta_vs_full"]), -0.1
    )
    assert np.isclose(float(rows[4]["delta_vs_full"]), -0.4)
    assert full["reprompts"] == "1" and full["windows"] == "10" and full["tp"] == "3"
    assert float(no_score["recall"]) == 0.5


def test_table_refuses_missing_run_and_mismatched_objects(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_inputs(results)
    (results / f"{table.method_for('low_score')}_davis_val.csv").unlink()
    with pytest.raises(E2Error, match="missing input"):
        table.build_table(results, ["davis_val"])
    write_scores_csv(
        results / f"{table.method_for('low_score')}_davis_val.csv", [ScoreRow("v1", 1, 0, 0)]
    )
    with pytest.raises(table.E3Error, match="different videos"):
        table.build_table(results, ["davis_val"])


def test_figure_is_built_from_the_table_only(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_inputs(results)
    out = tmp_path / "e3.csv"
    table.main(["--results-dir", str(results), "--out", str(out), "--splits", "davis_val"])
    path = figure.plot_e3(out, tmp_path / "figures")
    assert path.name == "e3_ablation.png" and path.stat().st_size > 0
