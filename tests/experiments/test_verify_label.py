from pathlib import Path

import pandas as pd
import pytest

from experiments.verify import label_candidates
from src.recovery.verify import CANDIDATE_COLUMNS, candidate_path
from tests.eval.sam2_trees import build_world, run_all
from tests.experiments.e2_trees import blank_object, make_online_run

CONFIG = "configs/recovery/rules_csrt_tiny.yaml"
METHOD = "fix_rules_csrt_tiny"
SPLIT = "sav_val"


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset, models=("tiny",), splits=(SPLIT,))
    out = root / "outputs"
    make_online_run(out, METHOD, SPLIT)
    return out


def write_log(out: Path, video: str, frames: list[int]) -> None:
    rows = [(video, 1, f, 4.0, 3.0, 1.0, 0.9, 0.5, True, "") for f in frames]
    path = candidate_path(out, METHOD, SPLIT, video)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS)).to_csv(path, index=False)


def run(results: Path, *extra: str) -> int:
    return label_candidates.main(["--config", CONFIG, "--results-dir", str(results), *extra])


def test_gain_is_method_iou_minus_alone_iou_at_the_first_annotated_frame(
    world: Path, tmp_path: Path
) -> None:
    out = world
    video = "vid_a"
    stems = sorted(p.stem for p in (out / "sam2_tiny" / SPLIT / "masks" / video).glob("*.png"))
    for name in ("vid_a", "vid_b"):
        write_log(out, name, [3] if name == video else [])
    blank_object(out, METHOD, SPLIT, video, stems[4], 1)
    results = tmp_path / "results"
    assert run(results) == 0
    got = pd.read_csv(results / "verify_candidates_tiny_sav_val.csv")
    assert list(got.columns) == list(label_candidates.LABEL_COLUMNS)
    row = got.iloc[0]
    assert row["gt_frame_index"] == 4 and row["iou_method"] == 0.0
    assert row["iou_alone"] > 0.9 and row["gain"] == pytest.approx(-row["iou_alone"])


def test_candidate_after_the_last_annotated_frame_is_unscored(world: Path, tmp_path: Path) -> None:
    write_log(world, "vid_a", [19])
    write_log(world, "vid_b", [])
    results = tmp_path / "results"
    run(results)
    got = pd.read_csv(results / "verify_candidates_tiny_sav_val.csv")
    assert len(got) == 1 and pd.isna(got["gain"].iloc[0]) and pd.isna(got["gt_frame_index"].iloc[0])


def test_unchanged_masks_give_zero_gain(world: Path, tmp_path: Path) -> None:
    write_log(world, "vid_a", [0, 8])
    write_log(world, "vid_b", [])
    results = tmp_path / "results"
    run(results)
    got = pd.read_csv(results / "verify_candidates_tiny_sav_val.csv")
    assert got["gain"].tolist() == [0.0, 0.0] and got["gt_frame_index"].tolist() == [0, 8]


def test_missing_candidate_log_stops(world: Path, tmp_path: Path) -> None:
    write_log(world, "vid_a", [3])
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "results")


def test_videos_file_limits_the_videos(world: Path, tmp_path: Path) -> None:
    write_log(world, "vid_a", [3])
    names = tmp_path / "names.txt"
    names.write_text("vid_a\n")
    results = tmp_path / "results"
    run(results, "--videos-file", str(names))
    assert len(pd.read_csv(results / "verify_candidates_tiny_sav_val.csv")) == 1


def test_test_splits_and_verified_configs_are_refused(world: Path, tmp_path: Path) -> None:
    for split in ("davis_val", "sav_test"):
        with pytest.raises(SystemExit, match="refused"):
            run(tmp_path / "results", "--split", split)
    with pytest.raises(SystemExit, match="accept-all"):
        label_candidates.main(
            [
                "--config",
                "configs/recovery/verified_rules_csrt_tiny.yaml",
                "--results-dir",
                str(tmp_path / "r"),
            ]
        )
    assert not (tmp_path / "results").exists()
