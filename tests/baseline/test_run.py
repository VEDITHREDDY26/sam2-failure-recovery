from pathlib import Path

import pandas as pd
import pytest

from src.baseline.config import METHOD, BaselineConfig
from src.baseline.run import run_batch, split_videos
from src.baseline.runner import timing_path
from src.baseline.timing import collect_timing
from src.baseline.timing import main as timing_main
from src.common.layout import mask_path
from tests.data.helpers import SAV_NAME, build_davis_root, build_sav_root, make_config

CFG = BaselineConfig(grabcut_iterations=2, roi_margin_frac=0.25, min_box_side=4)


def test_davis_batch_runs_then_skips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build_davis_root(tmp_path)
    cfg = make_config(tmp_path)
    assert run_batch(cfg, CFG, "davis_val") == (2, 0)
    assert run_batch(cfg, CFG, "davis_val") == (0, 2)
    assert "zebra: skipped (complete)" in capsys.readouterr().out
    out = tmp_path / cfg.output_root
    assert mask_path(out, METHOD, "davis_val", "bear", "00019").is_file()


def test_incomplete_video_is_redone(tmp_path: Path) -> None:
    build_davis_root(tmp_path, ("zebra",))
    cfg = make_config(tmp_path)
    run_batch(cfg, CFG, "davis_val")
    mask_path(tmp_path / cfg.output_root, METHOD, "davis_val", "zebra", "00004").unlink()
    assert run_batch(cfg, CFG, "davis_val") == (1, 0)


def test_unknown_video_rejected(tmp_path: Path) -> None:
    build_davis_root(tmp_path)
    with pytest.raises(ValueError, match="nope"):
        run_batch(make_config(tmp_path), CFG, "davis_val", videos=["nope"])


def test_sav_test_uses_only_the_subset_file(tmp_path: Path) -> None:
    build_sav_root(tmp_path, "sav_test", names=(SAV_NAME, "vid_b"))
    subset = tmp_path / "subset.txt"
    subset.write_text(f"# header\n{SAV_NAME}\n")
    cfg = make_config(tmp_path)
    assert split_videos(cfg, "sav_test", subset) == (SAV_NAME,)
    assert run_batch(cfg, CFG, "sav_test", subset_file=subset) == (1, 0)
    with pytest.raises(ValueError, match="vid_b"):
        run_batch(cfg, CFG, "sav_test", videos=["vid_b"], subset_file=subset)


def test_timing_collection_needs_every_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_davis_root(tmp_path)
    cfg = make_config(tmp_path)
    run_batch(cfg, CFG, "davis_val", videos=["zebra"])
    with pytest.raises(FileNotFoundError, match="bear"):
        collect_timing(cfg, "davis_val")
    run_batch(cfg, CFG, "davis_val")
    table = collect_timing(cfg, "davis_val")
    assert list(table["video"]) == ["zebra", "bear"]
    assert (table["frames"] == 20).all() and table["cpu_model"].notna().all()
    assert timing_path(tmp_path / cfg.output_root, METHOD, "davis_val", "bear").is_file()


def test_timing_cli_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_davis_root(tmp_path)
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(tmp_path))
    from src.baseline.run import main as run_main

    run_main(["--split", "davis_val"])
    out = tmp_path / "res" / "t.csv"
    timing_main(["--split", "davis_val", "--out", str(out)])
    assert len(pd.read_csv(out)) == 2
