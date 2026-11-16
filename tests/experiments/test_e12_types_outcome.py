from pathlib import Path

import pandas as pd
import pytest

from experiments.e10_types.count import LabelError
from experiments.e12 import types_outcome as to
from experiments.e12.candidates import CandidateError

SAMPLE = [
    ("s0", "davis_val", "bear", 1, 10, "reprompted"),
    ("s1", "davis_val", "bear", 1, 20, "reprompted"),
    ("s2", "sav_test", "sav_000001", 2, 30, "rejected"),
    ("s3", "sav_test", "sav_000002", 1, 40, "reprompted"),
]
TYPES = ["occlusion", "occlusion", "fast_motion", "other"]


def write_inputs(results: Path, types: list[str]) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    sample = pd.DataFrame(
        SAMPLE, columns=["sample_id", "split", "video", "object_id", "frame_index", "action"]
    )
    sample.to_csv(results / "e10_types_sample.csv", index=False)
    pd.DataFrame(
        {
            "video": ["bear", "bear"],
            "object_id": [1, 1],
            "frame_index": [10, 20],
            "gt_frame_index": [10, 20],
            "iou_after": [0.9, 0.2],
            "wrong": [False, True],
        }
    ).to_csv(results / "fix_rules_csrt_tiny_davis_val_reprompts.csv", index=False)
    pd.DataFrame(
        {
            "video": ["sav_000002"],
            "object_id": [1],
            "frame_index": [40],
            "gt_frame_index": [41],
            "iou_after": [0.5],
            "wrong": [False],
        }
    ).to_csv(results / "fix_rules_csrt_tiny_sav_test_reprompts.csv", index=False)
    labels = results / "labels.csv"
    pd.DataFrame({"sample_id": sample["sample_id"], "type": types, "notes": ""}).to_csv(
        labels, index=False
    )
    return labels


def test_counts_per_type(tmp_path: Path) -> None:
    labels = write_inputs(tmp_path, TYPES)
    rows = {r[0]: r[1:] for r in to.build(labels, tmp_path)}
    assert rows["occlusion"] == [2, 2, 1]
    assert rows["fast_motion"] == [1, 0, 0]
    assert rows["other"] == [1, 1, 0]
    assert rows["distractor"] == [0, 0, 0]


def test_invalid_labels_refused(tmp_path: Path) -> None:
    labels = write_inputs(tmp_path, ["occlusion", "", "other", "other"])
    with pytest.raises(LabelError):
        to.build(labels, tmp_path)
    labels = write_inputs(tmp_path, ["occlusion", "weird", "other", "other"])
    with pytest.raises(LabelError, match="not allowed"):
        to.build(labels, tmp_path)


def test_missing_inputs_refused(tmp_path: Path) -> None:
    with pytest.raises(CandidateError, match="missing"):
        to.build(tmp_path / "none.csv", tmp_path)
    labels = write_inputs(tmp_path, TYPES)
    (tmp_path / "fix_rules_csrt_tiny_sav_test_reprompts.csv").unlink()
    with pytest.raises(CandidateError, match="missing"):
        to.build(labels, tmp_path)


def test_cli_writes_csv(tmp_path: Path) -> None:
    labels = write_inputs(tmp_path / "r", TYPES)
    out = tmp_path / "o" / "t.csv"
    to.main(["--labels", str(labels), "--results-dir", str(tmp_path / "r"), "--out", str(out)])
    assert list(pd.read_csv(out).columns) == list(to.OUT_COLUMNS)
