import dataclasses
from pathlib import Path

import pytest

from src.common.boxes import Box
from src.common.config import ConfigError
from src.recovery.box_check import BoxCheck, load_box_check_config

SHAPE = (100, 200)  # height, width
LAST = Box(80, 40, 120, 80)  # 40 x 40, centre (100, 60), diagonal about 56.6
CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "recovery" / "box_check.yaml"


def reason(box: Box | None, last: Box | None = LAST) -> str | None:
    result = BoxCheck().check(box, SHAPE, last)
    assert result.ok == (result.reason is None)
    return result.reason


def test_accepts_a_box_close_to_the_last_reliable_one() -> None:
    assert reason(Box(85, 42, 127, 84)) is None


def test_no_box_is_rejected_as_none() -> None:
    assert reason(None) == "none"


def test_out_of_frame() -> None:
    assert reason(Box(180, 40, 240, 80)) == "out_of_frame"  # 1/3 visible
    assert reason(Box(300, 40, 340, 80)) == "out_of_frame"  # fully outside


def test_too_small() -> None:
    assert reason(Box(100, 50, 103, 80), last=None) == "too_small"


def test_too_large() -> None:
    assert reason(Box(0, 0, 200, 99), last=None) == "too_large"


def test_bad_aspect_absolute() -> None:
    assert reason(Box(10, 10, 190, 20), last=None) == "bad_aspect"  # 18:1


def test_bad_aspect_change_against_last_reliable_box() -> None:
    assert reason(Box(60, 50, 140, 70)) == "bad_aspect"  # 4:1 vs 1:1, area ratio 1


def test_size_jump() -> None:
    assert reason(Box(70, 30, 130, 90)) is None  # area ratio 2.25
    assert reason(Box(57, 17, 143, 103)) == "size_jump"  # area ratio 4.6
    assert reason(Box(98, 58, 102, 62)) == "size_jump"  # shrinks 100x


def test_centre_jump() -> None:
    last = Box(10, 10, 50, 50)  # centre (30, 30), diagonal about 56.6, limit about 85 px
    assert reason(Box(90, 10, 130, 50), last) is None  # moved 80 px
    assert reason(Box(150, 50, 190, 90), last) == "centre_jump"  # moved about 161 px


def test_relative_checks_are_skipped_without_a_last_reliable_box() -> None:
    assert reason(Box(0, 0, 100, 10), last=None) is None  # 10:1 is still inside the range


def test_rules_run_in_the_documented_order() -> None:
    assert reason(Box(190, 0, 260, 3)) == "out_of_frame"  # also small and far away


def test_shipped_config_loads() -> None:
    cfg = load_box_check_config(CFG_PATH)
    assert cfg.max_centre_jump_diagonals == 1.5 and cfg.min_side_px == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_visible_fraction", 0.0),
        ("max_area_fraction", 1.5),
        ("min_side_px", 0),
        ("min_aspect", 20.0),
        ("max_aspect_change", 0.5),
        ("max_size_ratio", 0.9),
        ("max_centre_jump_diagonals", 0.0),
    ],
)
def test_invalid_values_raise(field: str, value: float) -> None:
    cfg = load_box_check_config(CFG_PATH)
    with pytest.raises(ConfigError):
        dataclasses.replace(cfg, **{field: value})


def test_unknown_key_in_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bc.yaml"
    path.write_text(CFG_PATH.read_text() + "extra: 1\n")
    with pytest.raises(ConfigError):
        load_box_check_config(path)
