from pathlib import Path

import pytest

from src.common.layout import mask_path, signals_path
from src.eval import check_complete
from src.sam2_runner.timing import timing_path
from tests.eval.sam2_trees import build_world, run_all


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset)
    return root, subset


def _check(subset: Path, *extra: str) -> int:
    return check_complete.main(["--subset-file", str(subset), *extra])


def test_all_complete_exits_zero(world, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    _, subset = world
    assert _check(subset) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"sam2_{m}/{s}: {n}/{n} complete, 0 missing"
        for m in ("tiny", "large")
        for s, n in (("davis_val", 2), ("sav_val", 2), ("sav_test", 2))
    ]


def test_models_and_splits_can_be_selected(world, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    _, subset = world
    assert _check(subset, "--models", "tiny", "--splits", "sav_val") == 0
    assert capsys.readouterr().out.splitlines() == ["sam2_tiny/sav_val: 2/2 complete, 0 missing"]


def test_unfinished_video_is_named_and_fails(world, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    root, subset = world
    out = root / "outputs"
    signals_path(out, "sam2_large", "davis_val", "bear").unlink()
    assert _check(subset, "--models", "large", "--splits", "davis_val") == 1
    text = capsys.readouterr().out
    assert "sam2_large/davis_val: 1/2 complete, 1 missing" in text and "missing bear" in text


def test_deleted_png_fails(world, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    root, subset = world
    mask_path(root / "outputs", "sam2_tiny", "sav_val", "vid_b", "00007").unlink()
    assert _check(subset, "--models", "tiny", "--splits", "sav_val") == 1
    text = capsys.readouterr().out
    assert "1/2 complete, 1 missing" in text and "missing vid_b" in text


def test_missing_timing_fails(world, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    root, subset = world
    timing_path(root / "outputs", "sam2_tiny", "sav_test", "vid_a").unlink()
    assert _check(subset, "--models", "tiny", "--splits", "sav_test") == 1
    assert "missing vid_a" in capsys.readouterr().out


def test_never_run_split_lists_every_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    assert _check(subset, "--models", "tiny", "--splits", "davis_val") == 1
    text = capsys.readouterr().out
    assert "0/2 complete, 2 missing" in text and "missing zebra" in text and "missing bear" in text


def test_expected_objects_come_from_the_loader_without_a_prompts_file(world) -> None:  # type: ignore[no-untyped-def]
    root, subset = world
    (root / "outputs" / "prompts" / "sav_val.json").unlink()
    assert _check(subset, "--models", "tiny", "--splits", "sav_val") == 0
    signals = signals_path(root / "outputs", "sam2_tiny", "sav_val", "vid_a")
    signals.write_text(signals.read_text().replace(",3,", ",9,"))
    assert _check(subset, "--models", "tiny", "--splits", "sav_val") == 1


def test_subset_video_absent_from_test_list_is_an_error(world) -> None:  # type: ignore[no-untyped-def]
    root, _ = world
    bad = root / "bad.txt"
    bad.write_text("vid_a\nvid_nope\n")
    with pytest.raises(ValueError, match="vid_nope"):
        _check(bad, "--models", "tiny", "--splits", "sav_test")
