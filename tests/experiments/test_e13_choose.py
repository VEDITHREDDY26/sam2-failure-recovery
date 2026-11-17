import math
from pathlib import Path

import pandas as pd
import pytest

from experiments.e13 import choose, pilot
from src.common.boxes import Box
from src.recovery.gate import BoxGate, load_box_gate
from src.recovery.tracker import TrackResult


def cand(
    arm: str,
    cls: str,
    iou_gt: float | None = 0.8,
    iou_sam2: float | None = 0.2,
    score: float | None = 0.9,
    status: str = "ok",
    check: bool = True,
    first: bool = False,
) -> dict[str, object]:
    return {
        "video": "v",
        "arm": arm,
        "status": status,
        "cls": cls,
        "iou_gt": iou_gt,
        "iou_sam2": iou_sam2,
        "score": score,
        "box_check_ok": check if status == "ok" else "",
        "first_of_object": first,
    }


def table(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_outcomes_useful_harmful_neutral_and_exclusions() -> None:
    rows = [
        cand("a", "real_failure", 0.8),  # useful
        cand("a", "real_failure", 0.2),  # harmful: bad box
        cand("a", "not_visible", None),  # harmful
        cand("a", "false_alarm", 0.9),  # neutral
        cand("a", "false_alarm", 0.1),  # harmful: bad box
        cand("a", "unscored", None),  # left out
        cand("a", "real_failure", 0.9, check=False),  # not sent
        cand("a", "real_failure", None, status="lost"),  # not sent
    ]
    assert choose.outcomes(table(rows), None, None) == (1, 3, 1)


def test_gate_thresholds_and_nan_score() -> None:
    rows = [
        cand("a", "real_failure", 0.8, iou_sam2=0.6, score=0.9),
        cand("a", "real_failure", 0.8, iou_sam2=0.1, score=0.3),
        cand("a", "real_failure", 0.8, iou_sam2=0.1, score=None),
    ]
    t = table(rows)
    assert choose.reprompt_mask(t, 0.5, None).tolist() == [True, False, False]
    assert choose.reprompt_mask(t, None, 0.5).tolist() == [False, True, True]
    assert choose.reprompt_mask(t, None, None).tolist() == [True, True, True]


@pytest.mark.parametrize("s_min", [None, 0.4])
@pytest.mark.parametrize("a_max", [None, 0.5])
def test_vectorised_gate_matches_the_loop_gate(s_min: float | None, a_max: float | None) -> None:
    sam2 = Box(10, 10, 30, 30)
    boxes = [Box(10, 10, 30, 30), Box(25, 25, 45, 45), Box(60, 60, 80, 80)]
    for score in (None, 0.2, 0.6):
        for box in boxes:
            row = cand("a", "real_failure", iou_sam2=choose_iou(box, sam2), score=score)
            blocked = BoxGate(s_min, a_max).reason(TrackResult("ok", box, box, score), sam2)
            assert (not choose.reprompt_mask(table([row]), s_min, a_max).iloc[0]) == (
                blocked is not None
            )


def choose_iou(a: Box, b: Box) -> float:
    from src.recovery.verify import box_iou

    return box_iou(a, b)


def test_grid_shape_s_min_only_for_vit_arms() -> None:
    rows = [cand(arm, "real_failure") for arm in choose.ARM_ORDER]
    grid = pd.DataFrame(choose.grid_rows("tiny", table(rows)))
    assert len(grid) == 3 * 4 + 3 * 6 * 4
    assert set(grid[~grid["arm"].str.startswith("vit_")]["s_min"]) == {"off"}
    assert set(grid[grid["arm"] == "vit_reinit"]["s_min"].astype(str)) == {
        "off",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
    }
    assert list(grid.columns) == list(choose.GRID_COLUMNS)


def grid_of(entries: list[tuple[str, object, object, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "tiny",
                "arm": arm,
                "s_min": s,
                "a_max": a,
                "reprompts": reprompts,
                "net": net,
            }
            for arm, s, a, reprompts, net in entries
        ]
    )


def test_choice_ties_fewer_reprompts_then_simpler_arm_then_fewer_gates() -> None:
    base = ("csrt_reinit", "off", "off", 10, 2)
    fewer = choose.choose(
        "tiny",
        grid_of([base, ("vit_anchored", "off", "off", 8, 5), ("motion", "off", "off", 9, 5)]),
    )
    assert fewer["arm"] == "vit_anchored"
    simpler = choose.choose(
        "tiny",
        grid_of([base, ("vit_anchored", "off", "off", 8, 5), ("vit_reinit", "off", "off", 8, 5)]),
    )
    assert simpler["arm"] == "vit_reinit"
    open_gate = choose.choose(
        "tiny",
        grid_of([base, ("vit_reinit", 0.4, "off", 8, 5), ("vit_reinit", "off", "off", 8, 5)]),
    )
    assert open_gate["s_min"] == "off"


def test_go_needs_positive_net_above_the_ungated_csrt_reinit() -> None:
    base = ("csrt_reinit", "off", "off", 10, 3)
    assert not choose.choose("tiny", grid_of([base, ("motion", "off", "off", 5, 3)]))["go"]
    assert not choose.choose("tiny", grid_of([("csrt_reinit", "off", "off", 4, -2)]))["go"]
    assert choose.choose("tiny", grid_of([base, ("motion", "off", "off", 5, 4)]))["go"]
    assert not choose.choose(
        "tiny", grid_of([("csrt_reinit", "off", "off", 4, 0), ("motion", "off", "off", 4, 0)])
    )["go"]


def test_gate_yaml_uses_inf_for_off_and_loads(tmp_path: Path) -> None:
    choice = {"model": "tiny", "s_min": "off", "a_max": 0.5}
    path = choose.write_gate_file(tmp_path, choice)
    assert path.name == "box_gate_tiny.yaml"
    assert load_box_gate(path).gate() == BoxGate(None, 0.5)
    both = choose.write_gate_file(tmp_path, {"model": "large", "s_min": 0.3, "a_max": "off"})
    assert load_box_gate(both).gate() == BoxGate(0.3, None)


def write_merged(results: Path, model: str, rows: list[dict[str, object]]) -> None:
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results / f"e13_pilot_candidates_{model}_sav_val.csv", index=False)
    timing = [(("v", arm, 100, 0.5)) for arm in choose.ARM_ORDER]
    pd.DataFrame(timing, columns=list(pilot.TIMING_COLUMNS)).to_csv(
        results / f"e13_pilot_timing_{model}_sav_val.csv", index=False
    )


def good_world() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in choose.ARM_ORDER:
        rows.append(cand(arm, "false_alarm", 0.1, iou_sam2=0.9, first=True))  # harmful, copies
        rows.append(cand(arm, "real_failure", 0.9, iou_sam2=0.1, score=0.8))
    return rows


def test_main_writes_the_tables_and_the_gate_file_only_on_go(tmp_path: Path) -> None:
    results, configs = tmp_path / "results", tmp_path / "configs"
    configs.mkdir()
    write_merged(results, "tiny", good_world())
    assert choose.main(["--results-dir", str(results), "--config-dir", str(configs)]) == 0
    choice = pd.read_csv(results / "e13_choice.csv")
    assert list(choice.columns) == list(choose.CHOICE_COLUMNS)
    assert bool(choice["go"].iloc[0]) and (configs / "box_gate_tiny.yaml").is_file()
    assert not (configs / "box_gate_large.yaml").exists()
    assert (results / "e13_gate_grid.csv").is_file() and (results / "e13_pilot.csv").is_file()
    flat = [{**cand(a, "real_failure", 0.1)} for a in choose.ARM_ORDER]
    results2, configs2 = tmp_path / "r2", tmp_path / "c2"
    configs2.mkdir()
    write_merged(results2, "tiny", flat)
    choose.main(["--results-dir", str(results2), "--config-dir", str(configs2)])
    assert not pd.read_csv(results2 / "e13_choice.csv")["go"].iloc[0]
    assert not list(configs2.iterdir())


def test_pilot_table_shares_views_and_timing() -> None:
    rows = [
        cand("motion", "real_failure", 0.9, iou_sam2=0.8, first=True),
        cand("motion", "real_failure", 0.2, iou_sam2=0.1),
        cand("motion", "false_alarm", 0.9, iou_sam2=0.7),
        cand("motion", "not_visible", None, iou_sam2=0.1),
        cand("motion", "unscored", None, None),
    ]
    timing = pd.DataFrame([("v", "motion", 200, 0.1)], columns=list(pilot.TIMING_COLUMNS))
    out = pd.DataFrame(choose.pilot_rows("tiny", table(rows), timing))
    row = out[(out["arm"] == "motion") & (out["view"] == "all")].iloc[0]
    assert (row["candidates"], row["real_failure"], row["unscored"]) == (5, 2, 1)
    assert row["real_good_box_share"] == 0.5 and row["real_copies_sam2_share"] == 0.5
    assert row["false_alarm_copies_sam2_share"] == 1.0 and row["ms_per_frame"] == pytest.approx(0.5)
    first = out[(out["arm"] == "motion") & (out["view"] == "first_of_object")].iloc[0]
    assert first["candidates"] == 1 and first["real_good_box_share"] == 1.0
    empty = out[(out["arm"] == "csrt_reinit") & (out["view"] == "all")].iloc[0]
    assert empty["candidates"] == 0 and math.isnan(empty["real_good_box_share"])
