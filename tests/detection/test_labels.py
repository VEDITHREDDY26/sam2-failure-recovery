from pathlib import Path

import numpy as np
import pytest

from src.common.layout import mask_path
from src.common.masks import BoolMask, read_label_map, write_label_map
from src.data.davis import load_davis_video
from src.data.sav import load_sav_video
from src.detection.labels import (
    FAILURE_IOU,
    LABEL_COLUMNS,
    count_skipped_objects,
    is_failure,
    mask_iou,
    video_labels,
)
from src.sam2_runner.smoke import load_data_config
from tests.data.helpers import write_binary_png
from tests.eval.sam2_trees import build_world, run_split
from tests.fixtures.synthetic_video import make_synthetic_video


def _mask(*flat_on: int) -> BoolMask:
    mask = np.zeros(6, dtype=bool)
    mask[list(flat_on)] = True
    return mask.reshape(2, 3)


def test_iou_hand_computed_cases() -> None:
    # pred {0,1,2}, gt {1,2,3,4}: intersection {1,2} = 2, union {0,1,2,3,4} = 5 -> 2/5
    assert mask_iou(_mask(0, 1, 2), _mask(1, 2, 3, 4)) == pytest.approx(0.4)
    # pred {0,1}, gt {0,1,2,3,4,5}: intersection 2, union 6 -> 1/3
    assert mask_iou(_mask(0, 1), _mask(0, 1, 2, 3, 4, 5)) == pytest.approx(1 / 3)
    # identical masks -> 1
    assert mask_iou(_mask(2, 3), _mask(2, 3)) == 1.0
    # disjoint masks -> 0
    assert mask_iou(_mask(0), _mask(5)) == 0.0


def test_empty_mask_cases() -> None:
    assert mask_iou(_mask(), _mask()) == 1.0
    assert mask_iou(_mask(0, 1), _mask()) == 0.0
    assert mask_iou(_mask(), _mask(0, 1)) == 0.0


def test_failure_is_strictly_below_half() -> None:
    assert FAILURE_IOU == 0.5
    # pred {0,1}, gt {0,1,2,3}: intersection 2, union 4 -> exactly 0.5
    assert mask_iou(_mask(0, 1), _mask(0, 1, 2, 3)) == 0.5
    assert not is_failure(0.5)
    # pred {0,1}, gt {0,1,2,3,4}: 2/5 -> failure
    assert is_failure(mask_iou(_mask(0, 1), _mask(0, 1, 2, 3, 4)))
    assert is_failure(0.4999) and not is_failure(0.5001) and not is_failure(1.0)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    return root, subset


def _run(root: Path, subset: Path, split: str) -> Path:
    source = make_synthetic_video(root / "fake_source", "source")
    run_split(root, "tiny", split, subset, source)
    cfg = load_data_config()
    return cfg.data_root / cfg.output_root


def test_davis_rows_cover_every_frame_of_every_object(world) -> None:  # type: ignore[no-untyped-def]
    root, subset = world
    out = _run(root, subset, "davis_val")
    video = load_davis_video(load_data_config(), "zebra")
    labels = video_labels(video, out, "sam2_tiny", "davis_val")
    assert list(labels.columns) == list(LABEL_COLUMNS)
    assert labels.failure.dtype == bool
    # DAVIS annotates all 20 frames of both objects, both prompted at frame 0: 40 rows
    assert len(labels) == 40 and not labels.failure.any()
    assert set(labels.object_id) == {1, 2} and set(labels.video) == {"zebra"}


def test_sav_rows_only_for_annotated_frames_after_the_prompt_frame(world) -> None:  # type: ignore[no-untyped-def]
    root, subset = world
    out = _run(root, subset, "sav_val")
    video = load_sav_video(load_data_config(), "val", "vid_a")
    labels = video_labels(video, out, "sam2_tiny", "sav_val")
    # objects 1 and 2: annotated 0, 4, 8, 12, 16 -> 5 rows each; object 3 appears at frame 8
    # (frames 0 and 4 hold an empty annotation before its prompt frame): rows at 8, 12, 16
    by_object = {o: list(g.frame_index) for o, g in labels.groupby("object_id")}
    assert by_object == {1: [0, 4, 8, 12, 16], 2: [0, 4, 8, 12, 16], 3: [8, 12, 16]}
    assert len(labels) == 13
    # the fake run never tracks object 3: GT non-empty, prediction empty -> IoU 0, 3 failures
    late = labels[labels.object_id == 3]
    assert late.iou.tolist() == [0.0, 0.0, 0.0] and late.failure.all()
    assert labels[labels.object_id != 3].failure.sum() == 0


def test_sav_threshold_edges_on_stored_predictions(world) -> None:  # type: ignore[no-untyped-def]
    root, subset = world
    out = _run(root, subset, "sav_val")
    cfg = load_data_config()
    video = load_sav_video(cfg, "val", "vid_a")
    # object 2 is a static 8x8 box (64 pixels); remove k of its pixels from the prediction:
    # intersection 64 - k, union 64 -> IoU (64 - k) / 64
    for frame, removed in ((4, 32), (8, 33), (12, 16)):
        path = mask_path(out, "sam2_tiny", "sav_val", "vid_a", f"{frame:05d}")
        label = read_label_map(path)
        gt = video.gt_mask(2, frame)
        assert gt is not None and int(gt.sum()) == 64
        label[label == 2] = 0
        keep = gt.copy()
        keep.flat[np.flatnonzero(gt)[:removed]] = False
        label[keep] = 2
        write_label_map(path, label)
    labels = video_labels(video, out, "sam2_tiny", "sav_val")
    two = labels[labels.object_id == 2].set_index("frame_index")
    assert two.loc[4, "iou"] == 0.5 and not two.loc[4, "failure"]
    assert two.loc[8, "iou"] == pytest.approx(31 / 64) and two.loc[8, "failure"]
    assert two.loc[12, "iou"] == pytest.approx(48 / 64) and not two.loc[12, "failure"]
    assert two.loc[[0, 16], "failure"].sum() == 0


def test_never_visible_object_is_skipped_and_counted(world) -> None:  # type: ignore[no-untyped-def]
    root, subset = world
    ann = root / "SA-V" / "sav_val" / "Annotations_6fps" / "vid_a" / "003"
    for frame in (0, 4, 8, 12, 16):
        write_binary_png(ann / f"{frame:05d}.png", None)
    out = _run(root, subset, "sav_val")
    video = load_sav_video(load_data_config(), "val", "vid_a")
    assert video.object_ids == (1, 2, 3, 4)
    assert count_skipped_objects(video) == 1
    labels = video_labels(video, out, "sam2_tiny", "sav_val")
    assert set(labels.object_id) == {1, 2, 3} and len(labels) == 13
