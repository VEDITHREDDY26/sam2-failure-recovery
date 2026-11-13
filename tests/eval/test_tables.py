import csv
from pathlib import Path

import pytest

from experiments.e2 import table as e2_table
from src.eval import tables
from src.eval.rows import ScoreRow, write_scores_csv
from src.eval.tables import TableError, build_table, main

# (video, object) -> J&F, with J = F = value so that J&F equals the value
BASELINE = {("a", 1): 0.2, ("a", 2): 0.4, ("b", 1): 0.6, ("c", 1): 0.1}
TINY = {("a", 1): 0.5, ("a", 2): 0.5, ("b", 1): 0.7, ("c", 1): 0.3}
TINY_FIX = {("a", 1): 0.5, ("a", 2): 0.5, ("b", 1): 0.6, ("c", 1): 0.5}
LARGE = {k: v + 0.1 for k, v in TINY.items()}
FRAMES = {"a": 10, "b": 20, "c": 10}
SECONDS = {"a": 5.0, "b": 5.0, "c": 10.0}


def write_scores(path: Path, values: dict[tuple[str, int], float]) -> None:
    write_scores_csv(path, [ScoreRow(v, o, x, x) for (v, o), x in values.items()])


def write_timing(path: Path, device: tuple[str, str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["video", "frames", "compute_s", device[0]])
        for video in FRAMES:
            writer.writerow([video, FRAMES[video], SECONDS[video], device[1]])


def make_results(root: Path) -> Path:
    runs = {
        "baseline": (BASELINE, ("cpu_model", "arm")),
        "sam2_tiny": (TINY, ("gpu_name", "H100")),
        "fix_rules_csrt_tiny": (TINY_FIX, ("gpu_name", "H100")),
        "sam2_large": (LARGE, ("gpu_name", "H100")),
        "fix_rules_csrt_large": (LARGE, ("gpu_name", "H100")),
    }
    for split in tables.TEST_SETS:
        for prefix, (values, device) in runs.items():
            write_scores(root / f"{prefix}_{split}.csv", values)
            write_timing(root / f"{prefix}_timing_{split}.csv", device)
    return root


def row_of(table: list[dict[str, object]], method: str, split: str) -> dict[str, object]:
    return next(r for r in table if r["method"] == method and r["test_set"] == split)


def test_numbers_match_hand_computation(tmp_path: Path) -> None:
    table = build_table(make_results(tmp_path))
    assert len(table) == 10
    base = row_of(table, "baseline_csrt_grabcut", "davis_val")
    assert base["JF"] == pytest.approx(1.3 / 4)
    assert (base["videos"], base["objects"]) == (3, 4)
    assert base["fps"] == pytest.approx(40 / 20)
    assert base["hardware"] == "CPU: arm"
    assert base["delta_JF"] == ""
    tiny = row_of(table, "sam2_tiny", "sav_test")
    assert tiny["JF"] == pytest.approx(0.5)
    assert tiny["hardware"] == "GPU: H100"
    fix = row_of(table, "fix_rules_csrt_tiny", "davis_val")
    assert fix["JF"] == pytest.approx(0.525)
    assert fix["delta_JF"] == pytest.approx(0.025)
    assert (fix["videos_improved"], fix["videos_worse"], fix["videos_unchanged"]) == (1, 1, 1)
    assert fix["delta_low"] <= fix["delta_JF"] <= fix["delta_high"]  # type: ignore[operator]
    same = row_of(table, "fix_rules_csrt_large", "davis_val")
    assert (same["delta_JF"], same["delta_low"], same["delta_high"]) == (0.0, 0.0, 0.0)
    assert (same["videos_improved"], same["videos_worse"], same["videos_unchanged"]) == (0, 0, 3)


def test_worse_videos_are_counted_not_dropped(tmp_path: Path) -> None:
    root = make_results(tmp_path)
    write_scores(root / "fix_rules_csrt_tiny_davis_val.csv", {k: v - 0.2 for k, v in TINY.items()})
    fix = row_of(build_table(root), "fix_rules_csrt_tiny", "davis_val")
    assert (fix["videos_improved"], fix["videos_worse"], fix["videos_unchanged"]) == (0, 3, 0)
    assert fix["delta_JF"] == pytest.approx(-0.2)


def test_missing_input_is_refused_and_nothing_written(tmp_path: Path) -> None:
    root = make_results(tmp_path / "results")
    (root / "fix_rules_csrt_large_timing_sav_test.csv").unlink()
    out = tmp_path / "out"
    with pytest.raises(TableError, match="fix_rules_csrt_large_timing_sav_test.csv"):
        main(
            [
                "--results-dir", str(root),
                "--csv-out", str(out / "t.csv"),
                "--tex-out", str(out / "t.tex"),
            ]
        )  # fmt: skip
    assert not out.exists()


def test_different_object_sets_are_refused(tmp_path: Path) -> None:
    root = make_results(tmp_path)
    write_scores(
        root / "sam2_large_sav_test.csv", {k: v for k, v in LARGE.items() if k != ("a", 2)}
    )
    with pytest.raises(TableError, match="different videos or objects"):
        build_table(root)


def test_timing_videos_must_match_scores(tmp_path: Path) -> None:
    root = make_results(tmp_path)
    write_scores(
        root / "baseline_davis_val.csv", {k: v for k, v in BASELINE.items() if k[0] != "c"}
    )
    with pytest.raises(TableError):
        build_table(root)


def test_tex_regenerates_from_csv_and_contains_every_number(tmp_path: Path) -> None:
    root = make_results(tmp_path / "results")
    csv_out, tex_out = tmp_path / "t.csv", tmp_path / "t.tex"
    args = ["--results-dir", str(root), "--csv-out", str(csv_out), "--tex-out", str(tex_out)]
    main(args)
    first = tex_out.read_text()
    main(args)
    assert tex_out.read_text() == first
    assert "\\toprule" in first and "\\bottomrule" in first
    for row in tables.read_table(csv_out):
        assert f"{100 * float(row['JF']):.1f}" in first
        assert f"{float(row['fps']):.2f}" in first
        if row["delta_JF"]:
            assert f"{100 * float(row['delta_JF']):+.1f}" in first
            counts = f"{row['videos_improved']} / {row['videos_worse']} / {row['videos_unchanged']}"
            assert counts in first
    assert "1 / 1 / 1" in first and "0 / 0 / 3" in first
    assert "+2.5" in first and "52.5" in first and "32.5" in first
    assert "CPU: arm" in first and "GPU: H100" in first
    assert "J\\&F" in first


def test_tex_is_built_from_the_csv_only(tmp_path: Path) -> None:
    root = make_results(tmp_path)
    csv_out = tmp_path / "t.csv"
    tables.write_csv(csv_out, build_table(root))
    rows = tables.read_table(csv_out)
    for row in rows:
        if row["method"] == "baseline_csrt_grabcut" and row["test_set"] == "davis_val":
            row["JF"] = "0.9876"
    assert "98.8" in tables.latex_table(rows)


def test_change_counting_agrees_with_experiment_e2() -> None:
    assert tables.UNCHANGED_TOL == e2_table.UNCHANGED_TOL
    rows = [ScoreRow(v, o, x, x) for (v, o), x in TINY.items()]
    rows_fix = [ScoreRow(v, o, x, x) for (v, o), x in TINY_FIX.items()]
    mine = tables.count_changes(tables.per_video_scores(rows), tables.per_video_scores(rows_fix))
    theirs = e2_table.count_changes(e2_table.per_video_jf(rows), e2_table.per_video_jf(rows_fix))
    assert mine == theirs == (1, 1, 1)
