from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.e5 import train
from experiments.tune_rules.tune import TuneInputError, build_windows
from src.detection.learned import FEATURE_NAMES, load_pinned, sha256_of
from tests.experiments.test_e4 import write_rules
from tests.experiments.test_tune_rules import joined, make_video, write_world


def test_split_is_disjoint_complete_and_deterministic() -> None:
    names = [f"v{i:03d}" for i in range(155)]
    train_set, held = train.split_videos(names)
    assert not set(train_set) & set(held)
    assert sorted(train_set + held) == names
    assert len(held) == 31
    assert train.split_videos(list(reversed(names))) == (train_set, held)
    small_train, small_held = train.split_videos(["a", "b", "c"])
    assert len(small_held) == 1 and len(small_train) == 2


def test_split_never_puts_a_video_in_both_parts_for_any_size() -> None:
    for n in range(2, 40):
        names = [f"v{i}" for i in range(n)]
        a, b = train.split_videos(names)
        assert set(a).isdisjoint(b) and len(a) + len(b) == n


def separable_world() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [make_video(f"v{i}", {1: (0, 40), 2: (2, 33)}, seed=i) for i in range(10)]
    table = joined(
        pd.concat([p[0] for p in parts], ignore_index=True),
        pd.concat([p[1] for p in parts], ignore_index=True),
    )
    labels = pd.concat([p[2] for p in parts], ignore_index=True)
    # make failures a clean function of a low object score at the annotated frame
    key = ["video", "object_id", "frame_index"]
    low = table.merge(labels[key], on=key)["object_score"] < -1.0
    labels = labels.merge(table[key + ["object_score"]], on=key)
    labels["failure"] = labels["object_score"] < -1.0
    assert low.any() and labels["failure"].any()
    return table, labels.drop(columns="object_score")


def test_training_rows_are_annotated_frames_after_the_prompt_frame() -> None:
    table, labels = separable_world()
    features, failure = train.training_rows(table, labels)
    prompt = table.groupby(["video", "object_id"])["frame_index"].transform("min")
    merged = labels.merge(
        table[["video", "object_id", "frame_index"]].assign(prompt=prompt.to_numpy()),
        on=["video", "object_id", "frame_index"],
    )
    expected = int((merged["frame_index"] > merged["prompt"]).sum())
    assert len(failure) == expected == len(features)
    assert features.shape[1] == len(FEATURE_NAMES)


def test_fit_learns_a_low_score_signal_with_class_weights() -> None:
    table, labels = separable_world()
    features, failure = train.training_rows(table, labels)
    model = train.fit(features, failure, "tiny")
    assert model.coef[FEATURE_NAMES.index("score")] < 0
    assert train.fit(features, failure, "tiny") == model  # fixed seed


def test_window_max_and_threshold_by_hand() -> None:
    table, labels = separable_world()
    windows = build_windows(table, labels)
    probability = np.where(table["object_score"] < -1.0, 0.9, 0.1)
    best = train.window_max(probability, windows)
    assert best.shape == windows.truth.shape and (best >= 0).all()
    truth = np.array([True, True, False, False])
    scores = np.array([0.9, 0.6, 0.55, 0.1])
    assert train.choose_threshold(scores, truth) == 0.6
    assert train.pr_auc(scores, truth) == pytest.approx(1.0)
    assert np.isnan(train.pr_auc(scores, np.zeros(4, dtype=bool)))


def test_capped_threshold_respects_the_flag_rate_on_train_windows() -> None:
    truth = np.array([True] * 4 + [False] * 16)
    best = np.array([0.9, 0.9, 0.5, 0.5, 0.5] + [0.1] * 15)
    assert train.choose_threshold(best, truth) == 0.5  # flags 5/20, F1 8/9
    capped = train.choose_threshold(best, truth, 0.10)
    assert capped == 0.9 and (best >= capped).sum() <= 0.10 * len(truth)
    with pytest.raises(ValueError, match="no threshold"):
        train.choose_threshold(np.full(20, 0.99), truth, 0.10)


def run(tmp_path: Path, **kw: object) -> dict[str, str]:
    root = write_world(tmp_path / "data", 10, models=("tiny",))
    return train.run(  # type: ignore[arg-type]
        ("tiny",), root, write_rules(tmp_path / "rules.yaml"), tmp_path / "learned",
        tmp_path / "results", expect_videos=10, **kw,
    )  # fmt: skip


def test_run_writes_hashed_artifact_and_tables(tmp_path: Path) -> None:
    hashes = run(tmp_path)
    path = tmp_path / "learned" / "logreg_tiny.json"
    assert hashes == {"tiny": sha256_of(path)}
    assert load_pinned("tiny", tmp_path / "learned").sam2_model == "tiny"
    detector = pd.read_csv(tmp_path / "results" / "e5_detector.csv")
    assert list(detector.columns) == list(train.DETECTOR_COLUMNS)
    assert detector["method"].tolist() == ["logreg", "logreg", "rules"]
    assert detector["selection"].tolist() == ["f1_protocol", "f1_flag_cap_0.10", "rules_frozen"]
    assert detector["frozen"].tolist() == [False, True, True]
    frozen_row = detector[detector["frozen"] & (detector["method"] == "logreg")].iloc[0]
    assert load_pinned("tiny", tmp_path / "learned").threshold == pytest.approx(
        frozen_row["threshold"]
    )
    assert frozen_row["threshold"] >= detector["threshold"].iloc[0]
    assert (detector["train_videos"] == 8).all() and (detector["heldout_videos"] == 2).all()
    assert detector["windows"].nunique() == 1
    weights = pd.read_csv(tmp_path / "results" / "e5_weights.csv")
    assert weights["feature"].tolist() == list(FEATURE_NAMES)


def test_run_refuses_overwrite_missing_rules_and_non_val_input(tmp_path: Path) -> None:
    run(tmp_path)
    with pytest.raises(TuneInputError, match="exists"):
        run(tmp_path)
    run(tmp_path, overwrite=True)
    root = write_world(tmp_path / "data", 10, models=("tiny",))
    with pytest.raises(TuneInputError, match="not frozen"):
        train.run(("tiny",), root, tmp_path / "no.yaml", tmp_path / "l2", tmp_path / "r2", 10)
    bad = write_world(tmp_path / "davis_val_data", 10, models=("tiny",))
    with pytest.raises(TuneInputError, match="only SA-V val"):
        train.run(("tiny",), bad, tmp_path / "rules.yaml", tmp_path / "l3", tmp_path / "r3", 10)
    assert not (tmp_path / "l3").exists() and not (tmp_path / "l2").exists()
