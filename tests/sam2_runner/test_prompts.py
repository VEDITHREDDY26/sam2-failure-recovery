import json
from pathlib import Path

import pytest

from src.common.boxes import Box
from src.sam2_runner.prompts import load_batch_prompts, parse_batch_prompts, parse_prompts


def test_plain_form_uses_video_frame() -> None:
    p = parse_prompts({"video": "v", "frame_index": 2, "boxes": {"1": [0, 0, 4, 4]}})
    assert p.frame_index == 2 and p.start_frame(1) == 2 and p.object_frames == {}
    assert p.groups() == {2: [1]}


def test_per_object_form_and_mixed_form() -> None:
    raw = {
        "video": "v",
        "frame_index": 0,
        "boxes": {
            "3": {"frame_index": 8, "box": [1, 2, 5, 6]},
            "1": [0, 0, 4, 4],
            "2": {"frame_index": 8, "box": [2, 2, 6, 6]},
        },
    }
    p = parse_prompts(raw)
    assert p.boxes[3] == Box(1, 2, 5, 6)
    assert [p.start_frame(i) for i in (1, 2, 3)] == [0, 8, 8]
    assert p.groups() == {0: [1], 8: [2, 3]}


def test_video_frame_may_be_left_out_when_every_entry_has_its_own() -> None:
    entry = {"frame_index": 5, "box": [0, 0, 2, 2]}
    p = parse_prompts({"video": "v", "boxes": {"1": entry, "2": {**entry, "frame_index": 3}}})
    assert p.frame_index == 3 and p.groups() == {3: [2], 5: [1]}


@pytest.mark.parametrize(
    "boxes",
    [
        {},
        {"1": [0, 0, 2]},
        {"1": {"box": [0, 0, 2, 2]}},
        {"1": {"frame_index": 1}},
        {"1": {"frame_index": -1, "box": [0, 0, 2, 2]}},
        {"1": {"frame_index": 1, "box": [0, 0, 2]}},
    ],
)
def test_bad_boxes_are_rejected(boxes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_prompts({"video": "v", "frame_index": 0, "boxes": boxes})


def test_plain_boxes_without_video_frame_are_rejected() -> None:
    with pytest.raises(ValueError):
        parse_prompts({"video": "v", "boxes": {"1": [0, 0, 2, 2]}})
    with pytest.raises(ValueError):
        parse_prompts({"video": "v", "frame_index": -3, "boxes": {"1": [0, 0, 2, 2]}})


def test_batch_file_round_trip(tmp_path: Path) -> None:
    raw = {
        "split": "sav_val",
        "videos": {"a": {"boxes": {"1": {"frame_index": 4, "box": [0, 0, 2, 2]}}}},
    }
    path = tmp_path / "p.json"
    path.write_text(json.dumps(raw))
    assert load_batch_prompts(path).videos["a"].start_frame(1) == 4
    assert parse_batch_prompts(raw).split == "sav_val"
