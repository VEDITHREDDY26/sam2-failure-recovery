from pathlib import Path

import numpy as np
import pytest

from src.common.masks import LabelMap
from src.eval.completeness import IncompleteError
from src.eval.davis_scores import rows_from_metrics, score_davis, split_key
from src.eval.rows import ScoringError
from tests.eval.trees import copy_annotations, make_davis_root, rewrite_labels
from tests.fixtures.synthetic_video import OBJECT1_ID, OBJECT2_ID, SyntheticVideo


def _setup(video: SyntheticVideo, tmp_path: Path) -> tuple[Path, Path]:
    root = make_davis_root(video, tmp_path / "DAVIS")
    return root, copy_annotations(video, tmp_path / "pred")


def test_ground_truth_as_prediction_scores_one(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    root, pred = _setup(synthetic_video, tmp_path)
    rows = score_davis(root, pred, [synthetic_video.name])
    assert [(r.video, r.object) for r in rows] == [("synth", 1), ("synth", 2)]
    for row in rows:
        assert row.j == 1.0 and row.f == 1.0 and row.jf == 1.0


def test_empty_object_scores_zero_and_other_object_is_untouched(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    root, pred = _setup(synthetic_video, tmp_path)

    def drop_object2(label: LabelMap) -> LabelMap:
        return np.where(label == OBJECT2_ID, 0, label).astype(np.uint8)

    rewrite_labels(pred, synthetic_video, drop_object2)
    first, second = score_davis(root, pred, [synthetic_video.name])
    assert (first.object, first.j) == (OBJECT1_ID, 1.0)
    assert (second.object, second.j) == (OBJECT2_ID, 0.0)
    assert second.jf < 1.0


def test_missing_frame_is_an_error_not_an_exit(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    root, pred = _setup(synthetic_video, tmp_path)
    (pred / synthetic_video.name / "00005.png").unlink()
    with pytest.raises(IncompleteError, match="00005.png"):
        score_davis(root, pred, [synthetic_video.name])


def test_other_resolution_is_refused(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    root, pred = _setup(synthetic_video, tmp_path)
    with pytest.raises(ScoringError, match="480p"):
        score_davis(root, pred, [synthetic_video.name], resolution="Full-Resolution")


def test_keys_with_underscores_split_on_the_last_one() -> None:
    assert split_key("bike_packing_3") == ("bike_packing", 3)
    metrics = {
        "J": {"M_per_object": {"a_b_2": 0.5}},
        "F": {"M_per_object": {"a_b_2": 0.25}},
    }
    (row,) = rows_from_metrics(metrics)
    assert (row.video, row.object, row.j, row.f, row.jf) == ("a_b", 2, 0.5, 0.25, 0.375)
