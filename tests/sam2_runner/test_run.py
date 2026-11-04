import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common.layout import mask_path, masks_dir, signals_path
from src.common.signals import SIGNAL_COLUMNS, read_signals
from src.sam2_runner import run
from src.sam2_runner.predictor import REPO_ROOT
from src.sam2_runner.prompts import load_batch_prompts, parse_batch_prompts
from src.sam2_runner.runner import frame_stems, is_complete, run_video
from tests.fixtures.synthetic_video import OBJECT2_BOX, SyntheticVideo
from tests.sam2_runner.fakes import FakePredictor

METHOD = "sam2_tiny_pilot"
SPLIT = "davis_val"
OBJ_IDS = [1, 2]


class CountingFake(FakePredictor):
    inits = 0

    def init_state(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        type(self).inits += 1
        return super().init_state(*args, **kwargs)  # type: ignore[arg-type]


def _raw(video: SyntheticVideo, name: str) -> dict[str, object]:
    f0 = video.facts[0].object1_box
    assert f0 is not None
    boxes = {"1": [f0.x0, f0.y0, f0.x1, f0.y1], "2": [8, 44, 16, 52]}
    return {"split": SPLIT, "videos": {name: {"frame_index": 0, "boxes": boxes}}}


@pytest.fixture
def setup(synthetic_video: SyntheticVideo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = tmp_path / "drive"
    frames = data_root / "DAVIS" / "JPEGImages" / "480p" / "synth"
    frames.parent.mkdir(parents=True)
    shutil.copytree(synthetic_video.frames_dir, frames)
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(data_root))
    CountingFake.inits = 0
    monkeypatch.setattr(run, "build_predictor", lambda *a, **k: CountingFake(synthetic_video))
    prompts = tmp_path / "p.json"
    prompts.write_text(json.dumps(_raw(synthetic_video, "synth")))
    return data_root, prompts, data_root / "outputs"


def _main(prompts: Path) -> None:
    run.main(["--model", "tiny", "--prompts", str(prompts), "--method", METHOD])


def _stems(video: SyntheticVideo) -> list[str]:
    return frame_stems(video.frames_dir)


def test_output_format_matches_contract(synthetic_video: SyntheticVideo, setup) -> None:  # type: ignore[no-untyped-def]
    _, prompts, out = setup
    _main(prompts)
    csv = signals_path(out, METHOD, SPLIT, "synth")
    assert csv.read_text().splitlines()[0] == ",".join(SIGNAL_COLUMNS)
    df = read_signals(csv)
    n = synthetic_video.num_frames
    assert len(df) == 2 * n and set(df["video"]) == {"synth"}
    assert not csv.with_name(csv.name + ".tmp").exists()
    for i, facts in enumerate(synthetic_video.facts):
        r1 = df[(df.object_id == 1) & (df.frame_index == i)].iloc[0]
        r2 = df[(df.object_id == 2) & (df.frame_index == i)].iloc[0]
        assert (r1.area, r2.area) == (facts.object1_area, facts.object2_area)
        assert bool(r1["empty"]) == (facts.object1_area == 0)
        assert np.isnan(r1.centre_x) == (facts.object1_area == 0)
        assert r2.centre_x == pytest.approx((OBJECT2_BOX.x0 + OBJECT2_BOX.x1 - 1) / 2)
        assert r2.centre_y == pytest.approx((OBJECT2_BOX.y0 + OBJECT2_BOX.y1 - 1) / 2)
    pngs = sorted(masks_dir(out, METHOD, SPLIT, "synth").iterdir())
    assert [p.stem for p in pngs] == _stems(synthetic_video)
    for i, path in enumerate(pngs):
        with Image.open(path) as img:
            assert img.mode == "P"
            values = set(np.unique(np.asarray(img)).tolist()) - {0}
        assert values == ({1, 2} if synthetic_video.facts[i].object1_area else {2})


def test_rerun_skips_complete_videos(
    synthetic_video: SyntheticVideo, setup, capsys: pytest.CaptureFixture[str]  # type: ignore[no-untyped-def]
) -> None:
    _, prompts, out = setup
    _main(prompts)
    assert CountingFake.inits == 1
    files = sorted(p for p in out.rglob("*") if p.is_file())
    before = [(p, p.stat().st_mtime_ns, p.read_bytes()) for p in files]
    capsys.readouterr()
    _main(prompts)
    assert CountingFake.inits == 1
    assert capsys.readouterr().out.splitlines() == ["synth: skipped (complete)", "0 run, 1 skipped"]
    assert [(p, p.stat().st_mtime_ns, p.read_bytes()) for p in files] == before
    assert sorted(p for p in out.rglob("*") if p.is_file()) == files


def test_rerun_does_not_build_predictor(setup, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    _, prompts, _ = setup
    _main(prompts)

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("predictor built on a full rerun")

    monkeypatch.setattr(run, "build_predictor", boom)
    _main(prompts)


def _damage_delete_png(out: Path) -> None:
    mask_path(out, METHOD, SPLIT, "synth", "00007").unlink()


def _damage_truncate_csv(out: Path) -> None:
    path = signals_path(out, METHOD, SPLIT, "synth")
    path.write_text(path.read_text()[:120])


def _damage_remove_csv(out: Path) -> None:
    signals_path(out, METHOD, SPLIT, "synth").unlink()


@pytest.mark.parametrize("damage", [_damage_delete_png, _damage_truncate_csv, _damage_remove_csv])
def test_incomplete_output_is_detected_and_redone(
    synthetic_video: SyntheticVideo, setup, damage  # type: ignore[no-untyped-def]
) -> None:
    _, prompts, out = setup
    _main(prompts)
    stems = _stems(synthetic_video)
    assert is_complete(out, METHOD, SPLIT, "synth", stems, OBJ_IDS)
    damage(out)
    assert not is_complete(out, METHOD, SPLIT, "synth", stems, OBJ_IDS)
    _main(prompts)
    assert CountingFake.inits == 2
    assert is_complete(out, METHOD, SPLIT, "synth", stems, OBJ_IDS)
    assert len(list(masks_dir(out, METHOD, SPLIT, "synth").glob("*.png"))) == len(stems)


def test_wrong_objects_or_missing_rows_are_incomplete(
    synthetic_video: SyntheticVideo, setup  # type: ignore[no-untyped-def]
) -> None:
    _, prompts, out = setup
    _main(prompts)
    stems = _stems(synthetic_video)
    assert not is_complete(out, METHOD, SPLIT, "synth", stems, [1, 2, 3])
    assert not is_complete(out, METHOD, SPLIT, "synth", stems, [1])
    assert not is_complete(out, METHOD, SPLIT, "synth", stems[:-1], OBJ_IDS)


def test_redo_removes_only_this_videos_files(
    synthetic_video: SyntheticVideo, setup, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    _, prompts, out = setup
    other = mask_path(out, METHOD, SPLIT, "other", "00000")
    other.parent.mkdir(parents=True)
    other.write_bytes(b"keep")
    stale = mask_path(out, METHOD, SPLIT, "synth", "99999")
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    _main(prompts)
    assert other.read_bytes() == b"keep"
    assert not stale.exists()


def test_run_video_direct(synthetic_video: SyntheticVideo, tmp_path: Path) -> None:
    prompts = parse_batch_prompts(_raw(synthetic_video, "v1")).videos["v1"]
    fake = FakePredictor(synthetic_video)
    result = run_video(fake, synthetic_video.frames_dir, prompts, tmp_path, METHOD, SPLIT, "v1")
    assert len(result.table) == 2 * synthetic_video.num_frames
    assert is_complete(tmp_path, METHOD, SPLIT, "v1", _stems(synthetic_video), OBJ_IDS)


def test_videos_filter_and_unknown_video(setup) -> None:  # type: ignore[no-untyped-def]
    _, prompts, out = setup
    with pytest.raises(ValueError):
        run.main(["--model", "tiny", "--prompts", str(prompts), "--videos", "nope"])
    assert not out.exists()


def test_sav_frame_folders(setup) -> None:  # type: ignore[no-untyped-def]
    from src.sam2_runner.smoke import load_data_config

    cfg = load_data_config()
    assert run.frames_dir_for(cfg, "sav_val", "sav_000001") == (
        cfg.data_root / "SA-V" / "sav_val" / "JPEGImages_24fps" / "sav_000001"
    )
    assert run.frames_dir_for(cfg, "sav_test", "v").parts[-3:] == (
        "sav_test",
        "JPEGImages_24fps",
        "v",
    )


def test_batch_prompts_validation(tmp_path: Path) -> None:
    good = {"split": "sav_val", "videos": {"a": {"frame_index": 0, "boxes": {"1": [0, 0, 2, 2]}}}}
    assert parse_batch_prompts(good).split == "sav_val"
    with pytest.raises(ValueError):
        parse_batch_prompts({**good, "split": "train"})
    with pytest.raises(ValueError):
        parse_batch_prompts({"split": "sav_val", "videos": {"a/b": good["videos"]["a"]}})
    with pytest.raises(ValueError):
        parse_batch_prompts({"split": "sav_val", "videos": {}})
    path = tmp_path / "p.json"
    path.write_text(json.dumps(good))
    assert load_batch_prompts(path).videos["a"].boxes[1].x1 == 2


def test_runner_never_reads_ground_truth() -> None:
    forbidden = re.compile(r"src\.oracle|src\.data|Annotations|src\.detection\.labels")
    for path in (REPO_ROOT / "src" / "sam2_runner").glob("*.py"):
        text = path.read_text()
        for line in text.splitlines():
            if line.lstrip().startswith(("import ", "from ")):
                assert not forbidden.search(line), f"{path.name}: {line}"
        assert "Annotations" not in text, path.name
