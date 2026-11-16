from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.tune_rules import export_sav_val
from src.common.layout import mask_path
from src.common.masks import read_label_map, write_label_map
from src.detection.labels import video_labels
from src.eval.check_complete import load_video
from src.eval.score import load_data_config
from tests.eval.sam2_trees import build_world, run_all


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset, models=("tiny",), splits=("sav_val",))
    return root, subset


def _export(subset: Path, out: Path) -> int:
    return export_sav_val.main(
        ["--subset-file", str(subset), "--out-dir", str(out), "--models", "tiny"]
    )


def test_labels_equal_video_labels_and_boxes_match_masks(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    root, subset = world
    out = tmp_path / "export"
    assert _export(subset, out) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "boxes_tiny_sav_val.csv",
        "labels_tiny_sav_val.csv",
    ]
    cfg = load_data_config()
    method_root = cfg.data_root / cfg.output_root
    expected = pd.concat(
        [
            video_labels(load_video(cfg, "sav_val", v), method_root, "sam2_tiny", "sav_val")
            for v in ("vid_a", "vid_b")
        ],
        ignore_index=True,
    )
    labels = pd.read_csv(out / "labels_tiny_sav_val.csv")
    pd.testing.assert_frame_equal(labels, expected, check_dtype=False)
    boxes = pd.read_csv(out / "boxes_tiny_sav_val.csv")
    assert list(boxes.columns) == list(export_sav_val.BOX_COLUMNS)
    video = load_video(cfg, "sav_val", "vid_a")
    row = boxes[(boxes.video == "vid_a") & boxes.box_x0.notna()].iloc[0]
    label = read_label_map(
        mask_path(
            method_root, "sam2_tiny", "sav_val", "vid_a", video.frame_stems[int(row.frame_index)]
        )
    )
    ys, xs = np.nonzero(label == row.object_id)
    assert (row.box_x0, row.box_y0, row.box_x1, row.box_y1) == (
        xs.min(),
        ys.min(),
        xs.max() + 1,
        ys.max() + 1,
    )
    assert (row.height, row.width) == label.shape
    assert not boxes.duplicated(["video", "object_id", "frame_index"]).any()
    assert not (out / ".parts_tiny_sav_val").exists()


def test_empty_mask_gives_empty_box_cells(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    root, subset = world
    cfg = load_data_config()
    output_root = cfg.data_root / cfg.output_root
    video = load_video(cfg, "sav_val", "vid_a")
    stem = video.frame_stems[2]
    path = mask_path(output_root, "sam2_tiny", "sav_val", "vid_a", stem)
    write_label_map(path, np.zeros_like(read_label_map(path)))
    out = tmp_path / "export"
    assert _export(subset, out) == 0
    boxes = pd.read_csv(out / "boxes_tiny_sav_val.csv")
    frame = boxes[(boxes.video == "vid_a") & (boxes.frame_index == 2)]
    assert len(frame) >= 1
    assert frame[["box_x0", "box_y0", "box_x1", "box_y1"]].isna().all().all()
    assert frame.height.notna().all()


def test_incomplete_run_is_refused(  # type: ignore[no-untyped-def]
    world, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, subset = world
    mask_path(root / "outputs", "sam2_tiny", "sav_val", "vid_b", "00003").unlink()
    out = tmp_path / "export"
    assert _export(subset, out) == 1
    assert not list(out.glob("*.csv"))
    assert "refused" in capsys.readouterr().out


def test_resume_reuses_part_files(  # type: ignore[no-untyped-def]
    world, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, subset = world
    out = tmp_path / "export"
    parts = out / ".parts_tiny_sav_val"
    parts.mkdir(parents=True)
    cfg = load_data_config()
    video = load_video(cfg, "sav_val", "vid_a")
    root = cfg.data_root / cfg.output_root
    export_sav_val.write_atomic(
        video_labels(video, root, "sam2_tiny", "sav_val"), parts / "labels_vid_a.csv"
    )
    export_sav_val.write_atomic(
        export_sav_val.video_boxes(video, root, "sam2_tiny"), parts / "boxes_vid_a.csv"
    )
    assert _export(subset, out) == 0
    assert "vid_a: already done" in capsys.readouterr().out


def test_only_sav_val_is_reachable() -> None:
    assert export_sav_val.SPLIT == "sav_val"
    with pytest.raises(SystemExit):
        export_sav_val.main(["--split", "davis_val"])
    with pytest.raises(SystemExit):
        export_sav_val.main(["--splits", "sav_test"])
