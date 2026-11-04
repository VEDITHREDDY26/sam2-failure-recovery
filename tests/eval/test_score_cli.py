from pathlib import Path

import pytest

from src.common.layout import mask_path
from src.eval.completeness import IncompleteError
from src.eval.rows import SCORE_COLUMNS, read_scores_csv
from src.eval.score import main
from tests.eval.trees import make_davis_root, make_sav_gt
from tests.fixtures.synthetic_video import NUM_FRAMES, SyntheticVideo


@pytest.fixture
def data_root(
    tmp_path: Path, synthetic_video: SyntheticVideo, monkeypatch: pytest.MonkeyPatch
) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    make_davis_root(synthetic_video, root / "DAVIS")
    make_sav_gt(synthetic_video, root / "SA-V" / "sav_val" / "Annotations_6fps")
    return root


def _store_predictions(video: SyntheticVideo, root: Path, split: str, method: str) -> None:
    for index in range(NUM_FRAMES):
        target = mask_path(root / "outputs", method, split, video.name, f"{index:05d}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(video.annotation_path(index).read_bytes())


def test_davis_from_stored_method_predictions(
    data_root: Path, synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    _store_predictions(synthetic_video, data_root, "davis_val", "sam2_tiny")
    out = tmp_path / "r" / "davis.csv"
    main(["--dataset", "davis", "--split", "davis_val", "--method", "sam2_tiny", "--out", str(out)])
    assert out.read_text().splitlines()[0] == "video,object,J,F,J&F"
    assert out.read_text().splitlines()[0] == ",".join(SCORE_COLUMNS)
    rows = read_scores_csv(out)
    assert [(r.video, r.object, r.j, r.f, r.jf) for r in rows] == [
        ("synth", 1, 1.0, 1.0, 1.0),
        ("synth", 2, 1.0, 1.0, 1.0),
    ]


def test_sav_from_explicit_prediction_dir(
    data_root: Path, synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    out = tmp_path / "sav.csv"
    main(
        [
            "--dataset", "sav", "--split", "sav_val", "--out", str(out),
            "--pred-dir", str(synthetic_video.annotations_dir.parent),
        ]
    )  # fmt: skip
    assert [(r.object, r.j, r.f) for r in read_scores_csv(out)] == [(1, 1.0, 1.0), (2, 1.0, 1.0)]


def test_incomplete_predictions_stop_the_run_and_write_nothing(
    data_root: Path, synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    _store_predictions(synthetic_video, data_root, "davis_val", "sam2_tiny")
    mask_path(data_root / "outputs", "sam2_tiny", "davis_val", "synth", "00003").unlink()
    out = tmp_path / "davis.csv"
    with pytest.raises(IncompleteError):
        main(
            [
                "--dataset",
                "davis",
                "--split",
                "davis_val",
                "--method",
                "sam2_tiny",
                "--out",
                str(out),
            ]
        )
    assert not out.exists()


def test_method_and_pred_dir_are_exclusive(data_root: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--dataset", "davis", "--split", "davis_val", "--out", str(tmp_path / "x.csv")])


def test_split_must_match_dataset(data_root: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not belong"):
        main(
            ["--dataset", "davis", "--split", "sav_val", "--pred-dir", str(tmp_path),
             "--out", str(tmp_path / "x.csv")]
        )  # fmt: skip
