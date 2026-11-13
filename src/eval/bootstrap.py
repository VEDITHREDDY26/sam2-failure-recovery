"""Cluster bootstrap over videos for mean-over-objects scores.

The point estimate is the mean over all objects. A resample draws videos with replacement and
keeps all their objects, so the statistic is the mean over the objects of the drawn videos.
The paired version resamples the same videos for two runs that score the same objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DEFAULT_SEED = 5415
DEFAULT_RESAMPLES = 10000
DEFAULT_LEVEL = 0.95


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def _percentile_interval(
    samples: npt.NDArray[np.float64], estimate: float, level: float
) -> Interval:
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return Interval(estimate, float(low), float(high))


def _ratio_bootstrap(
    sums: npt.NDArray[np.float64],
    counts: npt.NDArray[np.float64],
    seed: int,
    resamples: int,
    level: float,
) -> Interval:
    if len(sums) == 0 or counts.sum() <= 0:
        raise ValueError("bootstrap needs at least one video with objects")
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(sums), size=(resamples, len(sums)))
    stats = sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    return _percentile_interval(stats, float(sums.sum() / counts.sum()), level)


def _per_video(
    scores: Mapping[str, Sequence[float]],
) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    videos = sorted(scores)
    sums = np.array([float(np.sum(scores[v])) for v in videos])
    counts = np.array([float(len(scores[v])) for v in videos])
    return videos, sums, counts


def bootstrap_mean(
    scores: Mapping[str, Sequence[float]],
    seed: int = DEFAULT_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
) -> Interval:
    """Mean over objects with a percentile interval from resampling videos."""
    _, sums, counts = _per_video(scores)
    return _ratio_bootstrap(sums, counts, seed, resamples, level)


def bootstrap_delta(
    new: Mapping[str, Sequence[float]],
    old: Mapping[str, Sequence[float]],
    seed: int = DEFAULT_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = DEFAULT_LEVEL,
) -> Interval:
    """Mean(new) - mean(old) over objects, resampling the same videos for both runs."""
    if set(new) != set(old) or any(len(new[v]) != len(old[v]) for v in new):
        raise ValueError("paired bootstrap needs the same videos and objects in both runs")
    videos, new_sums, counts = _per_video(new)
    old_sums = np.array([float(np.sum(old[v])) for v in videos])
    return _ratio_bootstrap(new_sums - old_sums, counts, seed, resamples, level)
