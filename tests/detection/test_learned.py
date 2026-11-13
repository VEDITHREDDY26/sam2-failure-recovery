from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from src.detection.learned import (
    FEATURE_NAMES,
    LearnedModel,
    ModelFileError,
    OnlineLearnedDetector,
    load_model,
    load_pinned,
    model_path,
    save_model,
    sha256_of,
    table_features,
)
from src.detection.signals import TABLE_COLUMNS, obs_from_mask

H, W = 40, 50
N = len(FEATURE_NAMES)


def make_masks(seed: int, n: int = 14) -> list[npt.NDArray[np.bool_]]:
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(n):
        mask = np.zeros((H, W), dtype=bool)
        if rng.random() > 0.2:
            h, w = rng.integers(2, 20, 2)
            y, x = rng.integers(0, H - h), rng.integers(0, W - w)
            mask[y : y + h, x : x + w] = True
        masks.append(mask)
    return masks


Streams = dict[tuple[str, int], list[tuple[npt.NDArray[np.bool_], float]]]


def make_table(videos: dict[str, dict[int, int]]) -> tuple[pd.DataFrame, Streams]:
    rows: list[tuple[object, ...]] = []
    streams: Streams = {}
    for v, (name, objects) in enumerate(videos.items()):
        for obj, seed in objects.items():
            masks = make_masks(seed + 10 * v)
            scores = np.random.default_rng(seed).normal(0, 3, len(masks))
            streams[(name, obj)] = [(m, float(x)) for m, x in zip(masks, scores, strict=True)]
            for f, (mask, score) in enumerate(zip(masks, scores, strict=True)):
                o = obs_from_mask(mask, float(score))
                box = (o.box.x0, o.box.y0, o.box.x1, o.box.y1) if o.box else (np.nan,) * 4
                rows.append(
                    (name, obj, f, o.object_score, o.area, o.centre_x, o.centre_y, H, W, *box)
                )
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS)), streams


def model(threshold: float = 0.5, seed: int = 0) -> LearnedModel:
    rng = np.random.default_rng(seed)
    return LearnedModel(
        "tiny",
        tuple(rng.normal(0, 1, N)),
        tuple(rng.uniform(0.5, 2, N)),
        tuple(rng.normal(0, 1, N)),
        0.1,
        threshold,
    )


def test_streaming_features_and_probabilities_equal_batch() -> None:
    table, streams = make_table({"a": {1: 1, 2: 2}, "b": {1: 3}})
    m = model()
    batch = m.table_probabilities(table)
    features, _ = table_features(table)
    assert features.shape == (len(table), N) and np.isfinite(features).all()
    for (name, obj), stream in streams.items():
        detector = OnlineLearnedDetector(m)
        index = table.index[(table.video == name) & (table.object_id == obj)]
        for i, (mask, score) in zip(index, stream, strict=True):
            probability = detector.probability(obj, obs_from_mask(mask, float(score)))
            decision = detector.update(obj, 0, mask, float(score), (H, W))
            expected = batch[i]
            if probability is None:
                assert expected == 0.0 and not decision.flag
            else:
                assert probability == pytest.approx(expected, abs=1e-12)
                assert decision.flag == (expected >= m.threshold)


def test_future_frames_do_not_change_the_output_at_t() -> None:
    table, streams = make_table({"a": {1: 4}})
    m = model(threshold=0.4)
    stream = streams[("a", 1)]
    t = 7
    full = m.table_probabilities(table)
    cut = m.table_probabilities(table[table.frame_index <= t].reset_index(drop=True))
    assert np.array_equal(full[: t + 1], cut)
    other = list(stream[: t + 1]) + [(np.ones((H, W), dtype=bool), 9.0)] * 3
    outputs = []
    for frames in (stream, other):
        detector = OnlineLearnedDetector(m)
        flags = [
            detector.update(1, i, mask, float(s), (H, W)).flag
            for i, (mask, s) in enumerate(frames[: t + 1])
        ]
        outputs.append(flags)
    assert outputs[0] == outputs[1]


def test_observe_sets_history_without_flagging_and_reset_clears() -> None:
    m = model(threshold=1e-9)
    masks = make_masks(5)
    detector = OnlineLearnedDetector(m)
    detector.observe(1, 0, masks[0], 1.0, (H, W))
    assert detector.update(1, 1, masks[1], 1.0, (H, W)).flag
    detector.reset()
    assert not detector.update(1, 0, masks[0], 1.0, (H, W)).flag  # first frame again
    with pytest.raises(ValueError):
        detector.update(1, 1, np.zeros((3, 3), dtype=bool), 0.0, (H, W))


def test_artifact_round_trip_and_hash(tmp_path: Path) -> None:
    m = model()
    path = tmp_path / "logreg_tiny.json"
    digest = save_model(path, m)
    assert digest == sha256_of(path) and (tmp_path / "logreg_tiny.sha256").read_text().startswith(
        digest
    )
    assert load_model(path, digest) == m
    assert load_pinned("tiny", tmp_path) == m
    assert model_path("tiny", tmp_path) == path
    features = np.random.default_rng(0).normal(size=(5, N))
    assert np.array_equal(load_model(path).probabilities(features), m.probabilities(features))


def test_modified_artifact_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "logreg_tiny.json"
    digest = save_model(path, model())
    path.write_text(path.read_text().replace('"threshold": 0.5', '"threshold": 0.6'))
    with pytest.raises(ModelFileError, match="does not match"):
        load_model(path, digest)
    with pytest.raises(ModelFileError, match="does not match"):
        load_pinned("tiny", tmp_path)
    (tmp_path / "logreg_tiny.sha256").unlink()
    with pytest.raises(ModelFileError, match="no pinned hash"):
        load_pinned("tiny", tmp_path)
    with pytest.raises(ModelFileError, match="missing"):
        load_model(tmp_path / "none.json")


def test_malformed_models_are_refused(tmp_path: Path) -> None:
    good = model()
    with pytest.raises(ModelFileError):
        LearnedModel("tiny", good.mean[:-1], good.std, good.coef, 0.0, 0.5)
    with pytest.raises(ModelFileError):
        LearnedModel("tiny", good.mean, (0.0,) * N, good.coef, 0.0, 0.5)
    with pytest.raises(ModelFileError):
        LearnedModel("tiny", good.mean, good.std, good.coef, 0.0, 1.0)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(ModelFileError, match="not a learned-detector file"):
        load_model(bad)
