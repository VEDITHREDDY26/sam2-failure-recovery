from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common.masks import LabelMap
from src.eval.completeness import IncompleteError
from src.eval.rows import ScoringError
from src.eval.sav_scores import sav_object_id, score_sav
from tests.eval.trees import SAV_STEP, copy_annotations, make_sav_gt, rewrite_labels
from tests.fixtures.synthetic_video import NUM_FRAMES, OBJECT2_ID, SyntheticVideo


def _setup(video: SyntheticVideo, tmp_path: Path) -> tuple[Path, Path]:
    gt = make_sav_gt(video, tmp_path / "gt")
    return gt, copy_annotations(video, tmp_path / "pred")


def test_object_folder_maps_to_id_plus_one() -> None:
    assert sav_object_id("000") == 1
    assert sav_object_id("007") == 8


def test_ground_truth_as_prediction_scores_one(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    gt, pred = _setup(synthetic_video, tmp_path)
    rows = score_sav(gt, pred, [synthetic_video.name])
    assert [(r.video, r.object) for r in rows] == [("synth", 1), ("synth", 2)]
    for row in rows:
        assert row.j == 1.0 and row.f == 1.0 and row.jf == 1.0


def test_empty_object_scores_zero_and_other_object_is_untouched(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    gt, pred = _setup(synthetic_video, tmp_path)

    def drop_object2(label: LabelMap) -> LabelMap:
        return np.where(label == OBJECT2_ID, 0, label).astype(np.uint8)

    rewrite_labels(pred, synthetic_video, drop_object2)
    first, second = score_sav(gt, pred, [synthetic_video.name])
    assert (first.object, first.j) == (1, 1.0)
    assert (second.object, second.j) == (2, 0.0)


def test_only_gt_annotated_frames_are_required(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    gt, pred = _setup(synthetic_video, tmp_path)
    for index in range(NUM_FRAMES):
        if index % SAV_STEP:
            (pred / synthetic_video.name / f"{index:05d}.png").unlink()
    assert all(r.j == 1.0 for r in score_sav(gt, pred, [synthetic_video.name]))


def test_missing_annotated_frame_is_an_error_not_an_exit(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    gt, pred = _setup(synthetic_video, tmp_path)
    (pred / synthetic_video.name / "00008.png").unlink()
    with pytest.raises(IncompleteError, match="00008.png"):
        score_sav(gt, pred, [synthetic_video.name])


def test_object_without_gt_pixels_in_scored_frames_is_a_clear_error(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    gt, pred = _setup(synthetic_video, tmp_path)
    blank = np.zeros((synthetic_video.height, synthetic_video.width), dtype=np.uint8)
    for index in range(0, NUM_FRAMES, SAV_STEP):
        path = gt / synthetic_video.name / "002" / f"{index:05d}.png"
        path.parent.mkdir(exist_ok=True)
        Image.fromarray(blank).save(path)
    with pytest.raises(ScoringError, match="rejected"):
        score_sav(gt, pred, [synthetic_video.name])
