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
    """A point estimate with its percentile confidence interval.

    Attributes:
        estimate: Statistic computed on the original data (not the resample average).
        low: Lower percentile bound of the resampled statistic.
        high: Upper percentile bound of the resampled statistic.
    """

    estimate: float
    low: float
    high: float


def _percentile_interval(
    samples: npt.NDArray[np.float64], estimate: float, level: float
) -> Interval:
    """Take the central `level` percentile range of the resampled statistics.

    Args:
        samples: Statistic from each resample.
        estimate: Point estimate on the original data.
        level: Confidence level, e.g. 0.95 uses the 2.5th and 97.5th percentiles.
    """
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
    """Bootstrap a ratio of sums over videos (a mean over objects, or a mean difference).

    Args:
        sums: Per-video sum of the score (or of the paired difference).
        counts: Per-video number of objects.
        seed: Seed of the random generator, so intervals are reproducible.
        resamples: Number of bootstrap resamples.
        level: Confidence level.

    Raises:
        ValueError: If there are no videos or no objects.
    """
    if len(sums) == 0 or counts.sum() <= 0:
        raise ValueError("bootstrap needs at least one video with objects")
    # Resample whole videos with replacement: objects of one video are not independent.
    # Each resample's mean is total score / total objects of the drawn videos.
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(sums), size=(resamples, len(sums)))
    stats = sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    return _percentile_interval(stats, float(sums.sum() / counts.sum()), level)


def _per_video(
    scores: Mapping[str, Sequence[float]],
) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Summarise scores per video.

    Args:
        scores: For each video, the scores of its objects.

    Returns:
        Sorted video names, the sum of scores per video and the object count per video, in the
        same order.
    """
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
