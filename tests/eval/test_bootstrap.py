from itertools import product

import numpy as np
import pytest

from src.eval.bootstrap import bootstrap_delta, bootstrap_mean


def test_constant_data_has_zero_width_interval() -> None:
    interval = bootstrap_mean({"a": [0.5, 0.5], "b": [0.5], "c": [0.5]}, resamples=500)
    assert interval.estimate == interval.low == interval.high == 0.5


def test_two_video_case_matches_exact_enumeration() -> None:
    scores = {"a": [0.0], "b": [1.0, 1.0, 1.0]}
    draws = [(x, y) for x, y in product("ab", repeat=2)]
    exact = sorted(sum(sum(scores[v]) for v in d) / sum(len(scores[v]) for v in d) for d in draws)
    assert exact == [0.0, 0.75, 0.75, 1.0]
    interval = bootstrap_mean(scores, resamples=20000)
    assert interval.estimate == pytest.approx(0.75)
    assert interval.low == pytest.approx(0.0)
    assert interval.high == pytest.approx(1.0)
    wide = bootstrap_mean(scores, resamples=20000, level=0.4)
    assert wide.low == pytest.approx(0.75)
    assert wide.high == pytest.approx(0.75)


def test_symmetric_data_is_centred() -> None:
    rng = np.random.default_rng(1)
    half = rng.uniform(0.2, 0.8, size=40)
    scores = {f"v{i}": [float(x)] for i, x in enumerate(half)}
    scores |= {f"w{i}": [float(1.0 - x)] for i, x in enumerate(half)}
    interval = bootstrap_mean(scores)
    assert interval.estimate == pytest.approx(0.5)
    assert (interval.low + interval.high) / 2 == pytest.approx(0.5, abs=0.01)
    assert interval.low < 0.5 < interval.high


def test_same_seed_reproduces_and_other_seed_differs() -> None:
    scores = {f"v{i}": [i / 10.0] for i in range(10)}
    first = bootstrap_mean(scores, seed=1, resamples=30)
    assert first == bootstrap_mean(scores, seed=1, resamples=30)
    assert first != bootstrap_mean(scores, seed=2, resamples=30)


def test_paired_delta_of_identical_runs_is_zero() -> None:
    scores = {"a": [0.1, 0.9], "b": [0.4], "c": [0.7]}
    interval = bootstrap_delta(scores, scores)
    assert interval.estimate == interval.low == interval.high == 0.0


def test_paired_delta_constant_gain() -> None:
    old = {"a": [0.1, 0.3], "b": [0.5]}
    new = {"a": [0.2, 0.4], "b": [0.6]}
    interval = bootstrap_delta(new, old)
    assert interval.estimate == pytest.approx(0.1)
    assert interval.low == pytest.approx(0.1)
    assert interval.high == pytest.approx(0.1)


def test_paired_delta_refuses_different_objects() -> None:
    with pytest.raises(ValueError):
        bootstrap_delta({"a": [0.1]}, {"b": [0.1]})
    with pytest.raises(ValueError):
        bootstrap_delta({"a": [0.1, 0.2]}, {"a": [0.1]})
