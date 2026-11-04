from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common.masks import LabelMap, write_label_map
from src.eval.completeness import IncompleteError, check_completeness
from tests.eval.trees import SAV_STEP, copy_annotations, make_sav_gt
from tests.fixtures.synthetic_video import NUM_FRAMES, SyntheticVideo


def _davis(video: SyntheticVideo, tmp_path: Path) -> tuple[Path, Path]:
    gt = video.annotations_dir.parent
    return gt, copy_annotations(video, tmp_path / "pred")


def test_complete_prediction_passes(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    gt, pred = _davis(synthetic_video, tmp_path)
    check_completeness(gt, pred, [synthetic_video.name], sav=False)


def test_missing_frame_is_listed(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    gt, pred = _davis(synthetic_video, tmp_path)
    (pred / synthetic_video.name / "00007.png").unlink()
    with pytest.raises(IncompleteError, match="00007.png"):
        check_completeness(gt, pred, [synthetic_video.name], sav=False)


def test_missing_video_is_listed(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    gt, _ = _davis(synthetic_video, tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(IncompleteError, match="missing prediction folder"):
        check_completeness(gt, tmp_path / "empty", [synthetic_video.name], sav=False)


def test_wrong_shape_and_mode_are_listed(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    gt, pred = _davis(synthetic_video, tmp_path)
    folder = pred / synthetic_video.name
    write_label_map(folder / "00001.png", np.zeros((10, 10), dtype=np.uint8))
    Image.new("RGB", (synthetic_video.width, synthetic_video.height)).save(folder / "00002.png")
    with pytest.raises(IncompleteError) as info:
        check_completeness(gt, pred, [synthetic_video.name], sav=False)
    text = str(info.value)
    assert "00001.png: size" in text
    assert "00002.png: not an indexed" in text


def test_extra_object_id_is_listed(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    gt, pred = _davis(synthetic_video, tmp_path)
    label: LabelMap = np.zeros((synthetic_video.height, synthetic_video.width), dtype=np.uint8)
    label[0, 0] = 3
    write_label_map(pred / synthetic_video.name / "00003.png", label)
    with pytest.raises(IncompleteError, match="object id above"):
        check_completeness(gt, pred, [synthetic_video.name], sav=False)


def test_sav_needs_only_gt_annotated_frames(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    gt = make_sav_gt(synthetic_video, tmp_path / "gt")
    pred = tmp_path / "pred"
    for index in range(0, NUM_FRAMES, SAV_STEP):
        src = synthetic_video.annotation_path(index)
        dst = pred / synthetic_video.name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    check_completeness(gt, pred, [synthetic_video.name], sav=True)
    (pred / synthetic_video.name / "00008.png").unlink()
    with pytest.raises(IncompleteError, match="00008.png"):
        check_completeness(gt, pred, [synthetic_video.name], sav=True)


def test_long_lists_are_cut() -> None:
    error = IncompleteError([f"problem {i}" for i in range(25)])
    text = str(error)
    assert "25 problem(s)" in text and "problem 19" in text
    assert "problem 20" not in text and "and 5 more" in text
