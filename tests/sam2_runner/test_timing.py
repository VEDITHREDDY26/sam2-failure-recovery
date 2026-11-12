import sys
from pathlib import Path

import pandas as pd
import pytest

from src.common.layout import signals_path
from src.sam2_runner import timing
from src.sam2_runner.prompts import Prompts
from src.sam2_runner.runner import frame_stems, is_complete, run_video
from src.sam2_runner.timing import (
    CORE_COLUMNS,
    TIMING_COLUMNS,
    collect_timing,
    device_fields,
    timing_path,
)
from tests.fixtures.synthetic_video import OBJECT2_BOX, SyntheticVideo
from tests.sam2_runner.fakes import FakePredictor

METHOD = "sam2_tiny"
SPLIT = "davis_val"


def _run(video: SyntheticVideo, root: Path, name: str = "v") -> None:
    prompts = Prompts(name, 0, {2: OBJECT2_BOX})
    run_video(FakePredictor(video), video.frames_dir, prompts, root, METHOD, SPLIT, name)


def test_timing_file_has_the_agreed_columns(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    _run(synthetic_video, tmp_path)
    table = pd.read_csv(timing_path(tmp_path, METHOD, SPLIT, "v"), keep_default_na=False)
    assert tuple(table.columns) == TIMING_COLUMNS
    assert TIMING_COLUMNS[:6] == ("video", "frames", "objects", "compute_s", "wall_s", "fps")
    row = table.iloc[0]
    assert (row.video, row.frames, row.objects) == ("v", synthetic_video.num_frames, 1)
    assert 0 < row.compute_s <= row.wall_s
    assert row.fps == pytest.approx(row.frames / row.compute_s)


def test_timing_is_written_before_the_signal_csv(
    synthetic_video: SyntheticVideo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    from src.sam2_runner import runner

    original = runner._write_signals_atomic

    def spy(path: Path, table: pd.DataFrame) -> None:
        seen.append(timing_path(tmp_path, METHOD, SPLIT, "v").is_file())
        original(path, table)

    monkeypatch.setattr(runner, "_write_signals_atomic", spy)
    _run(synthetic_video, tmp_path)
    assert seen == [True]


def test_missing_timing_makes_a_video_incomplete_and_it_is_redone(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    _run(synthetic_video, tmp_path)
    stems = frame_stems(synthetic_video.frames_dir)
    assert is_complete(tmp_path, METHOD, SPLIT, "v", stems, [2])
    timing_path(tmp_path, METHOD, SPLIT, "v").unlink()
    assert not is_complete(tmp_path, METHOD, SPLIT, "v", stems, [2])
    timing_path(tmp_path, METHOD, SPLIT, "v").write_text("garbage")
    assert not is_complete(tmp_path, METHOD, SPLIT, "v", stems, [2])
    _run(synthetic_video, tmp_path)
    assert is_complete(tmp_path, METHOD, SPLIT, "v", stems, [2])


def test_device_fields_are_empty_strings_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    fields = device_fields(True)
    assert fields == {"gpu_name": "", "torch_version": "", "cuda_version": "", "offload_video": 1}


def test_collect_fails_naming_a_video_without_timing(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    _run(synthetic_video, tmp_path, "a")
    _run(synthetic_video, tmp_path, "b")
    table = collect_timing(tmp_path, METHOD, SPLIT, ["b", "a"])
    assert table["video"].tolist() == ["b", "a"] and tuple(table.columns) == TIMING_COLUMNS
    assert set(CORE_COLUMNS) <= set(table.columns)
    with pytest.raises(FileNotFoundError, match="'c'"):
        collect_timing(tmp_path, METHOD, SPLIT, ["a", "c"])
    assert signals_path(tmp_path, METHOD, SPLIT, "a").is_file()


def test_collector_cli(
    synthetic_video: SyntheticVideo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(tmp_path))
    _run(synthetic_video, tmp_path / "outputs", "a")
    listing = tmp_path / "videos.txt"
    listing.write_text("a\n")
    out = tmp_path / "res" / "t.csv"
    timing.main(
        ["--model", "tiny", "--split", SPLIT, "--videos-file", str(listing), "--out", str(out)]
    )
    assert len(pd.read_csv(out)) == 1 and "1 videos" in capsys.readouterr().out
    listing.write_text("a\nzzz\n")
    with pytest.raises(FileNotFoundError, match="zzz"):
        timing.main(
            ["--model", "tiny", "--split", SPLIT, "--videos-file", str(listing), "--out", str(out)]
        )
