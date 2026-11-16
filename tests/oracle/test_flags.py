import numpy as np
import pytest

from src.common.boxes import Box
from src.detection import labels
from src.oracle.flags import ORACLE_FAILURE_IOU, oracle_box, oracle_flag, oracle_iou


def _mask(*cells: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros((10, 10), dtype=bool)
    for y0, y1, x0, x1 in cells:
        out[y0:y1, x0:x1] = True
    return out


CASES = {
    "both_empty": (_mask(), _mask()),
    "gt_empty_pred_not": (_mask((0, 2, 0, 2)), _mask()),
    "pred_empty_gt_not": (_mask(), _mask((0, 2, 0, 2))),
    "identical": (_mask((1, 5, 1, 5)), _mask((1, 5, 1, 5))),
    "exactly_half": (_mask((0, 1, 0, 2)), _mask((0, 1, 0, 4))),
    "just_below_half": (_mask((0, 1, 0, 2)), _mask((0, 1, 0, 5))),
    "disjoint": (_mask((0, 2, 0, 2)), _mask((5, 7, 5, 7))),
}


def test_constant_matches_label_definition() -> None:
    assert ORACLE_FAILURE_IOU == labels.FAILURE_IOU


@pytest.mark.parametrize("name", CASES)
def test_agrees_with_label_definition(name: str) -> None:
    pred, gt = CASES[name]
    assert oracle_iou(pred, gt) == labels.mask_iou(pred, gt)
    assert oracle_flag(pred, gt) == labels.is_failure(labels.mask_iou(pred, gt))


def test_specific_flags() -> None:
    assert oracle_flag(*CASES["both_empty"]) is False
    assert oracle_flag(*CASES["gt_empty_pred_not"]) is True
    assert oracle_flag(*CASES["exactly_half"]) is False
    assert oracle_flag(*CASES["just_below_half"]) is True


def test_unannotated_frame_gives_none() -> None:
    assert oracle_flag(_mask((0, 2, 0, 2)), None) is None
    assert oracle_flag(_mask(), None) is None
    assert oracle_box(None) is None


def test_oracle_box_is_tight_and_none_for_empty() -> None:
    assert oracle_box(_mask((2, 5, 3, 8))) == Box(3, 2, 8, 5)
    assert oracle_box(_mask()) is None
