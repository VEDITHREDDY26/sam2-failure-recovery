import json
from pathlib import Path

import pytest

from experiments.sam2_runs import make_prompts
from src.sam2_runner.prompts import load_batch_prompts
from tests.data.helpers import (
    LATE_BOX,
    LATE_FIRST_FRAME,
    SAV_ANNOTATED,
    SAV_NAME,
    build_davis_root,
    build_sav_root,
    make_config,
    write_binary_png,
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(tmp_path))
    return tmp_path


def _run(split: str, out: Path, *extra: str) -> dict:  # type: ignore[type-arg]
    make_prompts.main(["--split", split, "--out", str(out), *extra])
    return json.loads(out.read_text())


def test_davis_val_prompts_every_object_at_frame_zero(root: Path, tmp_path: Path) -> None:
    videos = build_davis_root(root)
    raw = _run("davis_val", tmp_path / "d.json")
    assert list(raw["videos"]) == ["zebra", "bear"] and raw["never_visible"] == {}
    batch = load_batch_prompts(tmp_path / "d.json")
    assert batch.split == "davis_val"
    prompts = batch.videos["zebra"]
    assert sorted(prompts.boxes) == [1, 2] and prompts.groups() == {0: [1, 2]}
    box1 = videos[0].facts[0].object1_box
    assert prompts.boxes[1] == box1 and prompts.boxes[2] == videos[0].facts[0].object2_box


def test_sav_object_that_appears_late_is_prompted_at_its_own_frame(
    root: Path, tmp_path: Path
) -> None:
    build_sav_root(root, "sav_val")
    _run("sav_val", tmp_path / "s.json")
    prompts = load_batch_prompts(tmp_path / "s.json").videos[SAV_NAME]
    assert sorted(prompts.boxes) == [1, 2, 3]
    assert prompts.start_frame(1) == 0 and prompts.start_frame(3) == LATE_FIRST_FRAME
    assert prompts.boxes[3] == LATE_BOX
    assert prompts.groups() == {0: [1, 2], LATE_FIRST_FRAME: [3]}


def test_never_visible_object_is_left_out_and_reported(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sav = build_sav_root(root, "sav_val")
    for i in SAV_ANNOTATED:
        write_binary_png(sav / "Annotations_6fps" / SAV_NAME / "003" / f"{i:05d}.png", None)
    raw = _run("sav_val", tmp_path / "s.json")
    assert raw["never_visible"] == {SAV_NAME: [4]}
    assert sorted(raw["videos"][SAV_NAME]["boxes"]) == ["1", "2", "3"]
    assert "never visible, left out: vid_a objects [4]" in capsys.readouterr().out


def test_video_without_any_visible_object_is_an_error_naming_it(root: Path, tmp_path: Path) -> None:
    sav = build_sav_root(root, "sav_val")
    for folder in ("000", "001", "002"):
        for i in SAV_ANNOTATED:
            write_binary_png(sav / "Annotations_6fps" / SAV_NAME / folder / f"{i:05d}.png", None)
    with pytest.raises(ValueError, match=SAV_NAME):
        _run("sav_val", tmp_path / "s.json")


def test_sav_test_uses_exactly_the_subset_file(root: Path, tmp_path: Path) -> None:
    build_sav_root(root, "sav_test", names=("vid_a", "vid_b", "vid_c"))
    subset = tmp_path / "subset.txt"
    subset.write_text("# header\nvid_c\nvid_a\n")
    raw = _run("sav_test", tmp_path / "t.json", "--subset-file", str(subset))
    assert list(raw["videos"]) == ["vid_c", "vid_a"] and raw["split"] == "sav_test"


def test_listed_subset_video_missing_from_the_test_list_is_an_error(
    root: Path, tmp_path: Path
) -> None:
    build_sav_root(root, "sav_test", names=("vid_a",))
    subset = tmp_path / "subset.txt"
    subset.write_text("vid_a\nvid_zzz\n")
    with pytest.raises(ValueError, match="vid_zzz"):
        _run("sav_test", tmp_path / "t.json", "--subset-file", str(subset))


def test_default_output_location_and_split_videos(root: Path) -> None:
    build_davis_root(root)
    make_prompts.main(["--split", "davis_val"])
    cfg = make_config(root)
    assert (root / "outputs" / "prompts" / "davis_val.json").is_file()
    assert make_prompts.split_videos(cfg, "davis_val") == ["zebra", "bear"]


def test_committed_subset_file_has_sixty_videos() -> None:
    from src.data.make_subset import load_subset

    assert len(set(load_subset(make_prompts.SUBSET_FILE))) == 60
