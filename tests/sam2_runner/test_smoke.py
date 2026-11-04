import json
from pathlib import Path

import numpy as np
import pytest

from src.common.layout import mask_path, signals_path
from src.common.masks import label_map_to_masks, read_label_map
from src.common.signals import SIGNAL_COLUMNS, read_signals
from src.sam2_runner.smoke import (
    SPLIT,
    Prompts,
    frame_stems,
    load_prompts,
    parse_prompts,
    run_smoke,
    sam2_box,
)
from tests.fixtures.synthetic_video import OBJECT2_BOX, SyntheticVideo
from tests.sam2_runner.fakes import FakePredictor

METHOD = "sam2_tiny_smoke"


def _prompts(video: SyntheticVideo) -> Prompts:
    f0 = video.facts[0]
    assert f0.object1_box is not None
    return Prompts(video.name, 0, {1: f0.object1_box, 2: OBJECT2_BOX})


def _run(video: SyntheticVideo, out: Path, overlap: bool = False) -> FakePredictor:
    fake = FakePredictor(video, overlap=overlap)
    run_smoke(fake, video.frames_dir, _prompts(video), out, METHOD)
    return fake


def test_one_png_per_frame_with_prompt_ids(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    _run(synthetic_video, tmp_path)
    for i in range(synthetic_video.num_frames):
        path = mask_path(tmp_path, METHOD, SPLIT, synthetic_video.name, f"{i:05d}")
        ids = set(label_map_to_masks(read_label_map(path)))
        facts = synthetic_video.facts[i]
        assert ids == ({1, 2} if facts.object1_area else {2})
    assert len(list(path.parent.glob("*.png"))) == synthetic_video.num_frames


def test_boxes_are_passed_as_corners(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    fake = _run(synthetic_video, tmp_path)
    b = OBJECT2_BOX
    assert (0, 2, (float(b.x0), float(b.y0), float(b.x1), float(b.y1))) in fake.calls
    assert fake.init_path == str(synthetic_video.frames_dir)


def test_signal_csv_matches_masks(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    _run(synthetic_video, tmp_path)
    path = signals_path(tmp_path, METHOD, SPLIT, synthetic_video.name)
    assert list(path.read_text().splitlines()[0].split(",")) == list(SIGNAL_COLUMNS)
    df = read_signals(path)
    assert len(df) == 2 * synthetic_video.num_frames
    assert df["object_score"].isna().all()
    for i, facts in enumerate(synthetic_video.facts):
        row1 = df[(df.object_id == 1) & (df.frame_index == i)].iloc[0]
        assert row1.area == facts.object1_area
        if facts.object1_box is not None:
            assert row1.centre_x == pytest.approx(
                (facts.object1_box.x0 + facts.object1_box.x1 - 1) / 2
            )
        else:
            assert row1["empty"] and np.isnan(row1.centre_x)
        assert df[(df.object_id == 2) & (df.frame_index == i)].iloc[0].area == facts.object2_area


def test_overlap_goes_to_smaller_id_on_tie(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    _run(synthetic_video, tmp_path, overlap=True)
    label = read_label_map(mask_path(tmp_path, METHOD, SPLIT, synthetic_video.name, "00000"))
    masks = label_map_to_masks(label)
    assert int(masks[1].sum()) == synthetic_video.facts[0].object1_area
    assert not (masks[1] & masks[2]).any()
    df = read_signals(signals_path(tmp_path, METHOD, SPLIT, synthetic_video.name))
    area2 = df[(df.object_id == 2) & (df.frame_index == 0)].iloc[0].area
    assert area2 == synthetic_video.facts[0].object2_area + synthetic_video.facts[0].object1_area


def test_prompts_file_and_frame_order(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    raw = {
        "video": "synth",
        "frame_index": 0,
        "boxes": {"1": [3, 20, 15, 32], "2": [8, 44, 16, 52]},
    }
    path = tmp_path / "p.json"
    path.write_text(json.dumps(raw))
    prompts = load_prompts(path)
    assert prompts.boxes[2] == OBJECT2_BOX and sorted(prompts.boxes) == [1, 2]
    assert sam2_box(prompts.boxes[1]) == [3.0, 20.0, 15.0, 32.0]
    assert frame_stems(synthetic_video.frames_dir)[:3] == ["00000", "00001", "00002"]
    with pytest.raises(ValueError):
        parse_prompts({"video": "x", "frame_index": 0, "boxes": {}})


def test_cli_end_to_end_with_fake(
    synthetic_video: SyntheticVideo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from src.sam2_runner import smoke

    data_root = tmp_path / "drive"
    frames = data_root / "DAVIS" / "JPEGImages" / "480p" / "synth"
    frames.parent.mkdir(parents=True)
    shutil.copytree(synthetic_video.frames_dir, frames)
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(data_root))
    monkeypatch.setattr(smoke, "build_predictor", lambda *a, **k: FakePredictor(synthetic_video))
    prompts = tmp_path / "p.json"
    prompts.write_text(
        json.dumps({"video": "synth", "frame_index": 0, "boxes": {"1": [3, 20, 15, 32]}})
    )
    smoke.main(["--model", "tiny", "--video", "synth", "--prompts", str(prompts)])
    out = data_root / "outputs" / "sam2_tiny_smoke" / "davis_val"
    assert len(list((out / "masks" / "synth").glob("*.png"))) == synthetic_video.num_frames
    assert (out / "signals" / "synth.csv").is_file()
