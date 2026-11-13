from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.tune_rules import tune
from src.common.signals import write_signals
from src.detection.rules import ALL_RULES, RuleDetector, RuleThresholds, load_rules_config
from src.detection.signals import join_signals_boxes, obs_from_row

H, W = 100, 120
BOX_COLUMNS = ["box_x0", "box_y0", "box_x1", "box_y1"]


def make_video(
    name: str, objects: dict[int, tuple[int, int]], seed: int, calm: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Signals, boxes, labels for one video; objects = {id: (prompt frame, last frame)}.

    Labels sit on every 4th frame from the prompt frame on (the prompt frame included).
    calm = steady score, area and position, so a few frames can be the only flagged ones.
    """
    rng = np.random.default_rng(seed)
    sig_rows, box_rows, label_rows = [], [], []
    for obj, (start, stop) in objects.items():
        for f in range(start, stop + 1):
            empty = False if calm else rng.random() < 0.15
            area = 0 if empty else int(rng.integers(50, 3000))
            cx, cy = (np.nan, np.nan) if empty else (rng.uniform(5, W - 5), rng.uniform(5, H - 5))
            score = rng.normal(0, 4)
            if calm:
                area, cx, cy, score = 1000 + int(rng.integers(0, 20)), W / 2, H / 2, 5.0
                if rng.random() < 0.02:
                    score = -20.0
            sig_rows.append((name, obj, f, score, area, cx, cy, int(empty)))
            if empty:
                box = (None,) * 4
            else:
                x0, y0 = int(rng.integers(0, 60)), int(rng.integers(0, 50))
                box = (x0, y0, x0 + int(rng.integers(5, 50)), y0 + int(rng.integers(5, 40)))
            box_rows.append((name, obj, f, H, W, *box))
            if (f - start) % 4 == 0:
                label_rows.append((name, obj, f, 0.0, bool(rng.random() < 0.4)))
    signals = pd.DataFrame(
        sig_rows,
        columns=[
            "video", "object_id", "frame_index", "object_score", "area",
            "centre_x", "centre_y", "empty",
        ],
    )  # fmt: skip
    boxes = pd.DataFrame(
        box_rows,
        columns=["video", "object_id", "frame_index", "height", "width", *BOX_COLUMNS],
    )
    labels = pd.DataFrame(
        label_rows, columns=["video", "object_id", "frame_index", "iou", "failure"]
    )
    return signals, boxes, labels


def write_world(
    root: Path, n_videos: int, models: tuple[str, ...] = ("tiny", "large"), calm: bool = True
) -> Path:
    for m_index, model in enumerate(models):
        parts = [
            make_video(f"vid{i}", {1: (0, 40), 2: (2, 33)}, seed=100 * m_index + i, calm=calm)
            for i in range(n_videos)
        ]
        folder = root / "signals" / f"sam2_{model}" / "sav_val" / "signals"
        folder.mkdir(parents=True, exist_ok=True)
        for i, (signals, _, _) in enumerate(parts):
            write_signals(folder / f"vid{i}.csv", signals)
        pd.concat([p[1] for p in parts]).to_csv(root / f"boxes_{model}_sav_val.csv", index=False)
        pd.concat([p[2] for p in parts]).to_csv(root / f"labels_{model}_sav_val.csv", index=False)
    return root


def joined(signals: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    return join_signals_boxes(signals, boxes)


def test_grids_are_fixed_constants() -> None:
    assert tuple(float(v) for v in range(-8, 3)) == tune.SCORE_GRID
    assert tune.AREA_RATIO_GRID == (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0)
    assert tune.JUMP_RATIO_GRID == (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    assert tune.MIN_AREA_FRACTION_GRID == (0.001, 0.003, 0.01, 0.03, 0.1)
    assert tune.BORDER_MARGIN_GRID == (0.0, 4.0, 8.0, 16.0, 32.0)
    for grid in (
        tune.SCORE_GRID, tune.AREA_RATIO_GRID, tune.JUMP_RATIO_GRID,
        tune.MIN_AREA_FRACTION_GRID, tune.BORDER_MARGIN_GRID,
    ):  # fmt: skip
        assert list(grid) == sorted(set(grid))
    for point in ((0,) * 5, tuple(n - 1 for n in (11, 8, 8, 5, 5))):
        tune.thresholds_at(point)  # type: ignore[arg-type]


def _two_object_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, boxes, labels = [], [], []
    for obj, (start, stop, ann) in {1: (0, 14, (0, 4, 8, 12)), 2: (2, 9, (2, 4, 8))}.items():
        for f in range(start, stop + 1):
            rows.append(("v", obj, f, 1.0, 10, 5.0, 5.0, 0))
            boxes.append(("v", obj, f, H, W, 0, 0, 5, 5))
        labels += [("v", obj, f, 0.0, f == 8) for f in ann]
    sig = pd.DataFrame(
        rows,
        columns=["video", "object_id", "frame_index", "object_score", "area",
                 "centre_x", "centre_y", "empty"],
    )  # fmt: skip
    box = pd.DataFrame(
        boxes, columns=["video", "object_id", "frame_index", "height", "width", *BOX_COLUMNS]
    )
    lab = pd.DataFrame(labels, columns=["video", "object_id", "frame_index", "iou", "failure"])
    return joined(sig, box), lab


def test_window_construction_by_hand() -> None:
    table, labels = _two_object_table()
    windows = tune.build_windows(table, labels)
    # object 1: prompt 0, windows (0,4] (4,8] (8,12]; frames 13, 14 after the last label: none.
    # object 2: prompt 2, windows (2,4] (4,8]; frame 9 after the last label: none.
    obj1 = [-1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1, -1]
    obj2 = [-1, 3, 3, 4, 4, 4, 4, -1]
    assert windows.frame_window.tolist() == obj1 + obj2
    assert len(windows.truth) == 5
    assert windows.truth.tolist() == [False, True, False, False, True]


def test_flag_on_unannotated_frame_counts_in_the_next_annotated_window() -> None:
    table, labels = _two_object_table()
    windows = tune.build_windows(table, labels)
    frame_flags = np.zeros(len(table), dtype=bool)
    frame_flags[table.index[(table.object_id == 1) & (table.frame_index == 6)][0]] = (
        True  # window 1
    )
    frame_flags[table.index[(table.object_id == 1) & (table.frame_index == 0)][0]] = True  # prompt
    frame_flags[table.index[(table.object_id == 1) & (table.frame_index == 14)][0]] = True  # none
    frame_flags[table.index[(table.object_id == 2) & (table.frame_index == 3)][0]] = (
        True  # window 3
    )
    assert tune.window_any(frame_flags, windows).tolist() == [False, True, False, True, False]


def test_precision_recall_f1_by_hand() -> None:
    choice = tune.Choice(
        RuleThresholds(-1.0, 2.0, 0.5, 0.05, 8.0), windows=20, positives=6, flagged=4, tp=3,
        per_rule_fires={r: 0 for r in ALL_RULES}, per_rule_tp={r: 0 for r in ALL_RULES},
    )  # fmt: skip
    assert choice.fp == 1 and choice.fn == 3
    assert choice.precision == pytest.approx(0.75)
    assert choice.recall == pytest.approx(0.5)
    assert choice.f1 == pytest.approx(0.6)
    empty = tune.Choice(
        choice.thresholds, 5, 0, 0, 0,
        {r: 0 for r in ALL_RULES}, {r: 0 for r in ALL_RULES},
    )  # fmt: skip
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)


def _vectors(n_windows: int) -> tuple[list, list, list, list]:  # type: ignore[type-arg]
    zero = np.zeros(n_windows, dtype=bool)
    return (
        [zero.copy() for _ in tune.SCORE_GRID],
        [zero.copy() for _ in tune.AREA_RATIO_GRID],
        [zero.copy() for _ in tune.JUMP_RATIO_GRID],
        [zero.copy() for _ in tune.MIN_AREA_FRACTION_GRID for _ in tune.BORDER_MARGIN_GRID],
    )


def test_search_picks_highest_f1() -> None:
    truth = np.array([True, False, True, False])
    low, area, move, gone = _vectors(4)
    low[5] = np.array([True, True, False, False])  # tp 1, flagged 2: F1 0.5
    area[3] = np.array([True, False, True, False])  # perfect
    point, flagged, tp = tune.search(low, area, move, gone, truth)
    assert point == (0, 3, 0, 0, 0) and (flagged, tp) == (2, 2)


def test_search_ties_fewer_flagged_then_earlier_grid_point() -> None:
    truth = np.array([True, True, False, False])
    low, area, move, gone = _vectors(4)
    one = np.array([True, False, False, False])  # tp 1, flagged 1: F1 2/3
    all_on = np.array([True, True, True, True])  # tp 2, flagged 4: F1 2/3
    low[9] = all_on
    low[6] = one
    low[2] = one
    point, flagged, tp = tune.search(low, area, move, gone, truth)
    assert point == (2, 0, 0, 0, 0) and (flagged, tp) == (1, 1)
    low[2] = np.zeros(4, dtype=bool)
    point, _, _ = tune.search(low, area, move, gone, truth)
    assert point == (6, 0, 0, 0, 0)


def test_flag_cap_excludes_points_over_the_budget() -> None:
    truth = np.array([True] * 4 + [False] * 6)
    low, area, move, gone = _vectors(10)
    area[2] = np.array([True, True] + [False] * 8)  # flags 2, tp 2: F1 2/3
    point, flagged, tp = tune.search(low, area, move, gone, truth)
    assert point == (0, 2, 0, 0, 0) and (flagged, tp) == (2, 2)
    low[3] = np.array([True] * 4 + [False] * 6)  # perfect, flags 4 of 10
    point, flagged, tp = tune.search(low, area, move, gone, truth)
    assert point == (3, 0, 0, 0, 0) and (flagged, tp) == (4, 4)
    point, flagged, tp = tune.search(low, area, move, gone, truth, max_flag_rate=0.2)
    assert point == (0, 2, 0, 0, 0) and (flagged, tp) == (2, 2)
    point, _, _ = tune.search(low, area, move, gone, truth, max_flag_rate=0.4)
    assert point == (3, 0, 0, 0, 0)


def test_flag_cap_keeps_the_tie_break() -> None:
    truth = np.array([True, True, False, False, False, False, False, False, False, False])
    low, area, move, gone = _vectors(10)
    one = np.array([True] + [False] * 9)
    low[6] = one
    low[2] = one
    low[9] = np.array([True, True, True] + [False] * 7)  # tp 2, flagged 3: F1 0.8
    low[8] = np.array([True, True] + [False] * 8)  # tp 2, flagged 2: F1 1.0, flags 20%
    point, _, _ = tune.search(low, area, move, gone, truth, max_flag_rate=0.2)
    assert point == (8, 0, 0, 0, 0)
    low[8] = np.array([True, False, True] + [False] * 7)  # tp 1 flagged 2: F1 0.5
    low[7] = np.array([True, False, False, True] + [False] * 6)  # tp 1, flagged 2
    point, flagged, _ = tune.search(low, area, move, gone, truth, max_flag_rate=0.2)
    assert flagged == 1 and point == (2, 0, 0, 0, 0)


def test_flag_cap_with_no_eligible_point_raises() -> None:
    truth = np.array([True, False])
    low, area, move, gone = _vectors(2)
    for vector in (*low, *area, *move, *gone):
        vector[:] = True
    with pytest.raises(tune.TuneInputError, match="no grid point"):
        tune.search(low, area, move, gone, truth, max_flag_rate=0.4)


def test_disappearance_grid_index_maps_to_fraction_and_border() -> None:
    truth = np.array([True, False])
    low, area, move, gone = _vectors(2)
    n_border = len(tune.BORDER_MARGIN_GRID)
    gone[3 * n_border + 2] = np.array([True, False])
    point, _, _ = tune.search(low, area, move, gone, truth)
    assert point == (0, 0, 0, 3, 2)
    cfg = tune.thresholds_at(point)
    assert (cfg.min_area_fraction, cfg.border_margin_px) == (0.03, 8.0)


def test_streaming_detector_reproduces_the_searched_window_flags() -> None:
    parts = [make_video(f"v{i}", {1: (0, 40), 2: (2, 33), 3: (5, 21)}, seed=i) for i in range(4)]
    table = joined(
        pd.concat([p[0] for p in parts], ignore_index=True),
        pd.concat([p[1] for p in parts], ignore_index=True),
    )
    labels = pd.concat([p[2] for p in parts], ignore_index=True)
    windows = tune.build_windows(table, labels)
    protocol = tune.search(*tune.rule_vectors(table, windows), windows.truth)
    choice = tune._choice_at(table, windows, *protocol)
    stream = np.zeros(len(table), dtype=bool)
    fired = {name: np.zeros(len(table), dtype=bool) for name in ALL_RULES}
    for _, rows in table.groupby(["video", "object_id"], sort=False):
        detector = RuleDetector(choice.thresholds)
        for index, row in rows.iterrows():
            decision = detector.update(obs_from_row(row))
            stream[index] = decision.flag
            for name in decision.fired:
                fired[name][index] = True
    flagged = tune.window_any(stream, windows)
    assert int(flagged.sum()) == choice.flagged
    assert int((flagged & windows.truth).sum()) == choice.tp
    for name in ALL_RULES:
        assert int(tune.window_any(fired[name], windows).sum()) == choice.per_rule_fires[name]


def _run(root: Path, tmp_path: Path, *extra: str) -> int:
    return tune.main(
        ["--data-dir", str(root), "--config-out", str(tmp_path / "out" / "rules.yaml"),
         "--results-out", str(tmp_path / "out" / "rules_sav_val.csv"), *extra]
    )  # fmt: skip


def test_end_to_end_writes_loadable_config_and_results(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root = write_world(tmp_path / "data", 3)
    assert _run(root, tmp_path, "--expect-videos", "3") == 0
    results = pd.read_csv(tmp_path / "out" / "rules_sav_val.csv")
    assert results["model"].tolist() == ["tiny", "tiny", "large", "large"]
    assert results["selection"].tolist() == ["f1_protocol", "f1_flag_cap_0.10"] * 2
    assert results["frozen"].tolist() == [False, True, False, True]
    capped = results[results["frozen"]]
    assert (capped["flagged"] <= 0.10 * capped["windows"]).all()
    assert (results["split"] == "sav_val").all()
    for column in ("precision", "recall", "f1", "tp", "fires_low_score", "tp_disappearance"):
        assert column in results.columns
    assert (results["videos"] == 3).all()
    assert (results["overlapped_rows"] == 0).all()
    for model in ("tiny", "large"):
        cfg = load_rules_config(model, tmp_path / "out" / "rules.yaml")
        row = capped[capped["model"] == model].iloc[0]
        assert cfg.score_threshold == row["score_threshold"]
        assert cfg.border_margin_px == row["border_margin_px"]
    assert "precision=" in capsys.readouterr().out
    assert not list((tmp_path / "out").glob("*.tmp"))


def _assert_refused(tmp_path: Path, root: Path, *extra: str) -> None:
    assert _run(root, tmp_path, "--expect-videos", "3", *extra) == 1
    assert not (tmp_path / "out").exists()


def test_refuses_forbidden_split_paths(tmp_path: Path) -> None:
    for bad in ("davis_val_data", "sav_test_data"):
        root = write_world(tmp_path / bad, 3)
        _assert_refused(tmp_path, root)


def test_refuses_input_not_named_sav_val(tmp_path: Path) -> None:
    root = write_world(tmp_path / "data", 3)
    (root / "labels_tiny_sav_val.csv").rename(root / "labels_tiny_other.csv")
    _assert_refused(tmp_path, root)
    assert tune.input_paths(root, "tiny")[0].name == "labels_tiny_sav_val.csv"
    with pytest.raises(tune.TuneInputError):
        tune.check_paths([root / "labels_tiny_val.csv"])


def test_refuses_mismatched_video_sets(tmp_path: Path) -> None:
    root = write_world(tmp_path / "data", 3)
    (root / "signals" / "sam2_large" / "sav_val" / "signals" / "vid2.csv").unlink()
    _assert_refused(tmp_path, root)
    root = write_world(tmp_path / "data2", 3)
    labels = pd.read_csv(root / "labels_tiny_sav_val.csv", dtype={"video": str})
    labels[labels["video"] != "vid1"].to_csv(root / "labels_tiny_sav_val.csv", index=False)
    _assert_refused(tmp_path, root)


def test_refuses_wrong_video_count(tmp_path: Path) -> None:
    root = write_world(tmp_path / "data", 3)
    assert _run(root, tmp_path) == 1  # default expects 155 videos
    assert _run(root, tmp_path, "--expect-videos", "4") == 1
    assert not (tmp_path / "out").exists()


def test_needs_both_models(tmp_path: Path) -> None:
    root = write_world(tmp_path / "data", 3)
    assert _run(root, tmp_path, "--models", "tiny", "--expect-videos", "3") == 1
    assert not (tmp_path / "out").exists()


def test_overlapped_rows_are_accepted_and_counted(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root = write_world(tmp_path / "data", 3)
    path = root / "boxes_tiny_sav_val.csv"
    boxes = pd.read_csv(path, dtype={"video": str})
    hide = boxes[BOX_COLUMNS[0]].notna().to_numpy().nonzero()[0][[3, 20, 40]]
    boxes.loc[boxes.index[hide], BOX_COLUMNS] = np.nan
    boxes.to_csv(path, index=False)
    assert _run(root, tmp_path, "--expect-videos", "3") == 0
    results = pd.read_csv(tmp_path / "out" / "rules_sav_val.csv")
    assert results.drop_duplicates("model").set_index("model")["overlapped_rows"].to_dict() == {
        "tiny": 3,
        "large": 0,
    }
    assert "overlapped rows" in capsys.readouterr().out


def test_config_holds_only_the_capped_thresholds(tmp_path: Path) -> None:
    root = write_world(tmp_path / "data", 3)
    assert _run(root, tmp_path, "--expect-videos", "3") == 0
    results = pd.read_csv(tmp_path / "out" / "rules_sav_val.csv")
    capped = results[results["selection"] == "f1_flag_cap_0.10"]
    protocol = results[results["selection"] == "f1_protocol"]
    assert capped["frozen"].all() and not protocol["frozen"].any()
    for model in ("tiny", "large"):
        cfg = load_rules_config(model, tmp_path / "out" / "rules.yaml")
        row = capped[capped["model"] == model].iloc[0]
        assert (cfg.score_threshold, cfg.area_ratio, cfg.jump_ratio) == (
            row["score_threshold"], row["area_ratio"], row["jump_ratio"]
        )  # fmt: skip
        assert row["flagged"] <= tune.FROZEN_MAX_FLAG_RATE * row["windows"]
