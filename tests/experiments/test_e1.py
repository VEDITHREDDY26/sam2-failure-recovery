from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.e1 import table as t
from experiments.e1.figure import FIGURE_NAME
from src.eval.rows import SCORE_COLUMNS

# Two videos (a: 10 frames, b: 30 frames), three objects (a1, b1, b2).
# Per method J values (a1, b1, b2) and F values; FPS = 40 frames / compute seconds.
SCORES = {
    "baseline_csrt_grabcut": ([0.2, 0.4, 0.6], [0.4, 0.6, 0.8], (2.0, 6.0), "m4"),
    "sam2_tiny": ([0.6, 0.6, 0.9], [0.7, 0.5, 0.9], (1.0, 3.0), "T4"),
    "sam2_large": ([0.9, 0.9, 0.9], [0.9, 0.9, 0.9], (0.5, 1.5), "L4"),
}
OBJECTS = [("a", 1), ("b", 1), ("b", 2)]


def write(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def make_inputs(d: Path) -> None:
    for test_set in t.TEST_SETS:
        for method, (js, fs, secs, hw) in SCORES.items():
            rows = [
                [v, o, j, f, (j + f) / 2] for (v, o), j, f in zip(OBJECTS, js, fs, strict=False)
            ]
            write(d / t.SCORE_FILES[method].format(test_set=test_set), list(SCORE_COLUMNS), rows)
            col = "cpu_model" if method == "baseline_csrt_grabcut" else "gpu_name"
            timing = [["a", 10, secs[0], hw, "extra"], ["b", 30, secs[1], hw, "extra"]]
            write(
                d / t.TIMING_FILES[method].format(test_set=test_set),
                ["video", "frames", "compute_s", col, "other"],
                timing,
            )


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    make_inputs(tmp_path)
    return tmp_path


def read_out(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_hand_computed_values(inputs: Path) -> None:
    table = {(r["method"], r["test_set"]): r for r in t.build_table(inputs)}
    base = table[("baseline_csrt_grabcut", "davis_val")]
    # J = (0.2+0.4+0.6)/3 = 0.4 ; F = (0.4+0.6+0.8)/3 = 0.6 ; J&F = 0.5 ; fps = 40/8 = 5
    assert (base["videos"], base["objects"]) == (2, 3)
    assert base["J"] == pytest.approx(0.4)
    assert base["F"] == pytest.approx(0.6)
    assert base["J&F"] == pytest.approx(0.5)
    assert base["fps"] == pytest.approx(5.0)
    assert base["hardware"] == "m4"
    tiny = table[("sam2_tiny", "sav_test")]
    # J = 2.1/3 = 0.7 ; F = 2.1/3 = 0.7 ; fps = 40/4 = 10
    assert tiny["J"] == pytest.approx(0.7)
    assert tiny["J&F"] == pytest.approx(0.7)
    assert tiny["fps"] == pytest.approx(10.0)
    assert table[("sam2_large", "davis_val")]["fps"] == pytest.approx(20.0)


def test_hardware_joined_and_sorted(inputs: Path) -> None:
    path = inputs / "sam2_tiny_timing_davis_val.csv"
    write(
        path,
        ["video", "frames", "compute_s", "gpu_name"],
        [["a", 10, 1, "T4"], ["b", 30, 3, "A100"]],
    )
    row = {(r["method"], r["test_set"]): r for r in t.build_table(inputs)}[
        ("sam2_tiny", "davis_val")
    ]
    assert row["hardware"] == "A100;T4"


def test_csv_and_row_order(inputs: Path, tmp_path: Path) -> None:
    out = inputs / "out" / "e1.csv"
    t.write_table(out, t.build_table(inputs))
    with out.open() as handle:
        assert handle.readline().strip() == "method,test_set,videos,objects,J,F,J&F,fps,hardware"
    rows = read_out(out)
    assert [(r["test_set"], r["method"]) for r in rows] == [
        (s, m) for s in t.TEST_SETS for m in t.METHODS
    ]
    assert not list(out.parent.glob("*.tmp"))


def test_missing_file(inputs: Path) -> None:
    (inputs / "sam2_large_sav_test.csv").unlink()
    with pytest.raises(t.E1Error, match="missing input file"):
        t.build_table(inputs)


def test_missing_timing_file(inputs: Path) -> None:
    (inputs / "sam2_tiny_timing_davis_val.csv").unlink()
    with pytest.raises(t.E1Error, match="missing input file"):
        t.build_table(inputs)


def test_timing_video_set_differs(inputs: Path) -> None:
    write(
        inputs / "sam2_tiny_timing_davis_val.csv",
        ["video", "frames", "compute_s", "gpu_name"],
        [["a", 10, 1, "T4"]],
    )
    with pytest.raises(t.E1Error, match="differ from timing"):
        t.build_table(inputs)


def test_methods_score_different_objects(inputs: Path) -> None:
    rows = [["a", 1, 0.5, 0.5, 0.5], ["b", 1, 0.5, 0.5, 0.5]]
    write(inputs / "sam2_large_davis_val.csv", list(SCORE_COLUMNS), rows)
    with pytest.raises(t.E1Error, match="different objects"):
        t.build_table(inputs)


@pytest.mark.parametrize("bad", ["nan", "1.5", "-0.1"])
def test_bad_score_values(inputs: Path, bad: str) -> None:
    rows = [["a", 1, bad, 0.5, 0.5], ["b", 1, 0.5, 0.5, 0.5], ["b", 2, 0.5, 0.5, 0.5]]
    write(inputs / "sam2_tiny_davis_val.csv", list(SCORE_COLUMNS), rows)
    with pytest.raises(t.E1Error, match="not in"):
        t.build_table(inputs)


@pytest.mark.parametrize("frames,secs", [(0, 1), (10, 0), (10, -2), (-5, 1)])
def test_bad_timing_values(inputs: Path, frames: int, secs: float) -> None:
    write(
        inputs / "sam2_tiny_timing_davis_val.csv",
        ["video", "frames", "compute_s", "gpu_name"],
        [["a", frames, secs, "T4"], ["b", 30, 3, "T4"]],
    )
    with pytest.raises(t.E1Error, match="frames="):
        t.build_table(inputs)


def test_timing_without_hardware_column(inputs: Path) -> None:
    write(
        inputs / "sam2_tiny_timing_davis_val.csv",
        ["video", "frames", "compute_s"],
        [["a", 10, 1], ["b", 30, 3]],
    )
    with pytest.raises(t.E1Error, match="needs columns"):
        t.build_table(inputs)


def test_cli_end_to_end(inputs: Path, tmp_path: Path) -> None:
    out, figs = tmp_path / "o" / "e1.csv", tmp_path / "figs"
    t.main(["--results-dir", str(inputs), "--out", str(out), "--figure-dir", str(figs)])
    assert len(read_out(out)) == len(t.METHODS) * len(t.TEST_SETS)
    figure = figs / FIGURE_NAME
    assert figure.is_file() and figure.stat().st_size > 0
