"""Objects with different prompt frames: one SAM 2 pass per start frame."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.common.boxes import Box
from src.common.layout import mask_path, masks_dir, signals_path
from src.common.masks import masks_to_label_map, read_label_map, write_label_map
from src.common.signals import read_signals, signals_from_masks, write_signals
from src.sam2_runner import run
from src.sam2_runner.predictor import to_numpy
from src.sam2_runner.prompts import Prompts
from src.sam2_runner.runner import (
    clear_video,
    frame_stems,
    is_complete,
    run_video,
    sam2_box,
    signal_row,
)
from src.sam2_runner.scores import object_score
from tests.fixtures.synthetic_video import OBJECT2_BOX, SyntheticVideo
from tests.sam2_runner.fakes import FakePredictor

METHOD = "sam2_tiny"
SPLIT = "davis_val"
LATE = 17
IDS = [1, 2]


def legacy_run_video(predictor, frames_dir, prompts, output_root, split, video):  # type: ignore[no-untyped-def]
    """The single-pass runner as it was before per-object start frames (reference)."""
    stems = frame_stems(frames_dir)
    clear_video(output_root, METHOD, split, video)
    state = predictor.init_state(str(frames_dir))
    for obj_id, box in prompts.boxes.items():
        predictor.add_new_points_or_box(state, prompts.frame_index, obj_id, box=sam2_box(box))
    per_object: dict[int, list[Any]] = {i: [] for i in prompts.boxes}
    scores: dict[int, list[float]] = {i: [] for i in prompts.boxes}
    for frame_idx, obj_ids, logits in predictor.propagate_in_video(state):
        arrays = {o: to_numpy(logits[r, 0]) for r, o in enumerate(obj_ids)}
        first = next(iter(arrays.values()))
        label = masks_to_label_map(
            {i: a > 0 for i, a in arrays.items()}, (first.shape[0], first.shape[1]), scores=arrays
        )
        write_label_map(mask_path(output_root, METHOD, split, video, stems[frame_idx]), label)
        for row, obj_id in enumerate(obj_ids):
            obj_logits = to_numpy(logits[row, 0])
            per_object[obj_id].append(obj_logits > 0)
            scores[obj_id].append(object_score(state, obj_id, frame_idx, obj_logits)[0])
    table = pd.concat(
        [signals_from_masks(video, i, per_object[i], scores[i]) for i in sorted(per_object)],
        ignore_index=True,
    )
    write_signals(signals_path(output_root, METHOD, split, video), table)


def _box1(video: SyntheticVideo, frame: int) -> Box:
    box = video.facts[frame].object1_box
    assert box is not None
    return box


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "timing" not in p.parts
    }


@pytest.mark.parametrize(
    "fake_args",
    [{}, {"overlap": True}, {"overlap": True, "positives": {2: 6.0}}, {"stored_scores": {}}],
)
def test_single_group_output_is_unchanged(
    synthetic_video: SyntheticVideo, tmp_path: Path, fake_args: dict[str, Any]
) -> None:
    prompts = Prompts("v", 0, {2: OBJECT2_BOX, 1: _box1(synthetic_video, 0)})
    old, new = tmp_path / "old", tmp_path / "new"
    legacy_run_video(
        FakePredictor(synthetic_video, **fake_args),
        synthetic_video.frames_dir,
        prompts,
        old,
        SPLIT,
        "v",
    )
    run_video(
        FakePredictor(synthetic_video, **fake_args),
        synthetic_video.frames_dir,
        prompts,
        new,
        METHOD,
        SPLIT,
        "v",
    )
    assert _tree_bytes(new) == _tree_bytes(old)


def test_signal_row_matches_signals_from_masks() -> None:
    rng = np.random.default_rng(3)
    masks = [rng.random((17, 23)) > p for p in (0.2, 0.7, 0.95, 2.0)]
    assert not masks[-1].any()
    scores = [1.5, -2.0, 0.25, float("nan")]
    expected = signals_from_masks("v", 4, masks, scores)
    rows = [signal_row("v", 4, i, m, s) for i, (m, s) in enumerate(zip(masks, scores, strict=True))]
    got = pd.DataFrame(rows, columns=list(expected.columns))
    pd.testing.assert_frame_equal(got, expected, check_exact=True)


def test_later_object_is_empty_with_nan_score_before_its_start(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    prompts = Prompts("v", 0, {1: _box1(synthetic_video, LATE), 2: OBJECT2_BOX}, {1: LATE})
    fake = FakePredictor(synthetic_video)
    result = run_video(fake, synthetic_video.frames_dir, prompts, tmp_path, METHOD, SPLIT, "v")
    assert fake.starts == [0, LATE] and fake.resets == 1
    assert [c[:2] for c in fake.calls] == [(0, 2), (LATE, 1)]
    stems = frame_stems(synthetic_video.frames_dir)
    assert is_complete(tmp_path, METHOD, SPLIT, "v", stems, IDS)
    df = read_signals(signals_path(tmp_path, METHOD, SPLIT, "v"))
    assert len(df) == 2 * synthetic_video.num_frames == len(result.table)
    one = df[df.object_id == 1].set_index("frame_index")
    assert (one.loc[: LATE - 1, "area"] == 0).all() and one.loc[: LATE - 1, "empty"].all()
    assert one.loc[: LATE - 1, "object_score"].isna().all()
    assert one.loc[LATE:, "object_score"].notna().all()
    for f in range(LATE, synthetic_video.num_frames):
        assert one.loc[f, "area"] == synthetic_video.facts[f].object1_area
    two = df[df.object_id == 2]
    assert two["object_score"].notna().all()
    assert (two["area"] == [x.object2_area for x in synthetic_video.facts]).all()
    for f, stem in enumerate(stems):
        label = read_label_map(mask_path(tmp_path, METHOD, SPLIT, "v", stem))
        assert (label == 2).sum() == synthetic_video.facts[f].object2_area
        assert (label == 1).sum() == (synthetic_video.facts[f].object1_area if f >= LATE else 0)
    assert "NaN before the prompt frame for 17 rows" in result.source_summary


@pytest.mark.parametrize("positives", [{}, {2: 6.0}, {1: 6.0}])
def test_overlap_across_groups_follows_highest_logit_then_smaller_id(
    synthetic_video: SyntheticVideo, tmp_path: Path, positives: dict[int, float]
) -> None:
    start = 3
    boxes = {2: OBJECT2_BOX, 1: _box1(synthetic_video, start)}
    split_prompts = Prompts("v", 0, boxes, {1: start})
    shared = Prompts("v", 0, boxes)
    grouped, reference = tmp_path / "grouped", tmp_path / "reference"
    kwargs: dict[str, Any] = {"overlap": True, "positives": positives}
    run_video(
        FakePredictor(synthetic_video, **kwargs),
        synthetic_video.frames_dir,
        split_prompts,
        grouped,
        METHOD,
        SPLIT,
        "v",
    )
    run_video(
        FakePredictor(synthetic_video, **kwargs),
        synthetic_video.frames_dir,
        shared,
        reference,
        METHOD,
        SPLIT,
        "v",
    )
    stems = frame_stems(synthetic_video.frames_dir)
    overlapping = 0
    for f, stem in enumerate(stems):
        got = read_label_map(mask_path(grouped, METHOD, SPLIT, "v", stem))
        want = read_label_map(mask_path(reference, METHOD, SPLIT, "v", stem))
        if f >= start:
            np.testing.assert_array_equal(got, want)
            overlapping += int(synthetic_video.facts[f].object1_area > 0)
        else:
            assert not (got == 1).any()
    assert overlapping > 5
    winner = 2 if positives.get(2, 0) > positives.get(1, 0) else 1
    assert (read_label_map(mask_path(grouped, METHOD, SPLIT, "v", stems[start])) == winner).any()


def test_all_objects_starting_late_share_one_pass(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    prompts = Prompts("v", 5, {2: OBJECT2_BOX})
    fake = FakePredictor(synthetic_video)
    run_video(fake, synthetic_video.frames_dir, prompts, tmp_path, METHOD, SPLIT, "v")
    assert fake.starts == [5] and fake.resets == 0
    pngs = sorted(masks_dir(tmp_path, METHOD, SPLIT, "v").glob("*.png"))
    assert len(pngs) == synthetic_video.num_frames
    assert not read_label_map(pngs[4]).any() and read_label_map(pngs[5]).any()


def test_prompt_frame_beyond_video_is_an_error(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    prompts = Prompts("v", 0, {2: OBJECT2_BOX}, {2: synthetic_video.num_frames})
    with pytest.raises(ValueError):
        run_video(
            FakePredictor(synthetic_video),
            synthetic_video.frames_dir,
            prompts,
            tmp_path,
            METHOD,
            SPLIT,
            "v",
        )


def test_offload_flag_reaches_init_state(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    prompts = Prompts("v", 0, {2: OBJECT2_BOX})
    for flag in (False, True):
        fake = FakePredictor(synthetic_video)
        run_video(fake, synthetic_video.frames_dir, prompts, tmp_path, METHOD, SPLIT, "v", flag)
        assert fake.offload_video is flag


@pytest.fixture
def batch(synthetic_video: SyntheticVideo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    data_root = tmp_path / "drive"
    frames = data_root / "DAVIS" / "JPEGImages" / "480p" / "synth"
    frames.parent.mkdir(parents=True)
    shutil.copytree(synthetic_video.frames_dir, frames)
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(data_root))
    made: list[FakePredictor] = []

    def factory(*a: object, **k: object) -> FakePredictor:
        made.append(FakePredictor(synthetic_video))
        return made[-1]

    monkeypatch.setattr(run, "build_predictor", factory)
    b = _box1(synthetic_video, LATE)
    raw = {
        "split": SPLIT,
        "videos": {
            "synth": {
                "frame_index": 0,
                "boxes": {
                    "1": {"frame_index": LATE, "box": [b.x0, b.y0, b.x1, b.y1]},
                    "2": [8, 44, 16, 52],
                },
            }
        },
    }
    path = tmp_path / "p.json"
    path.write_text(json.dumps(raw))
    return path, data_root / "outputs", made


def test_rerun_skips_and_partial_video_is_redone(
    synthetic_video: SyntheticVideo, batch, capsys: pytest.CaptureFixture[str]  # type: ignore[no-untyped-def]
) -> None:
    prompts, out, made = batch
    args = ["--model", "tiny", "--prompts", str(prompts), "--offload-video"]
    run.main(args)
    assert len(made) == 1 and made[0].offload_video is True
    stems = frame_stems(synthetic_video.frames_dir)
    assert is_complete(out, METHOD, SPLIT, "synth", stems, IDS)
    capsys.readouterr()
    run.main(args)
    assert len(made) == 1
    assert capsys.readouterr().out.splitlines() == ["synth: skipped (complete)", "0 run, 1 skipped"]
    good = signals_path(out, METHOD, SPLIT, "synth").read_bytes()
    mask_path(out, METHOD, SPLIT, "synth", stems[9]).unlink()
    stale = mask_path(out, METHOD, SPLIT, "synth", "99999")
    stale.write_bytes(b"x")
    run.main(args)
    assert len(made) == 2 and not stale.exists()
    assert is_complete(out, METHOD, SPLIT, "synth", stems, IDS)
    assert signals_path(out, METHOD, SPLIT, "synth").read_bytes() == good
