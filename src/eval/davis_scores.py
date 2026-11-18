"""Per-object J and F from the official DAVIS 2017 semi-supervised evaluator (pinned, unmodified).

The evaluator skips the first and last frame, reads <pred_root>/<video>/<frame>.png and
exits the process on bad input, so the completeness check runs first and any SystemExit is
turned into ScoringError. Its DAVIS class defaults to 480p and DAVISEvaluation does not pass a
resolution on, so only 480p is accepted.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from src.eval.completeness import check_completeness
from src.eval.evaluators import load_davis
from src.eval.rows import ScoreRow, ScoringError, sorted_rows

DAVIS_RESOLUTION = "480p"
DAVIS_TASK = "semi-supervised"


def split_key(key: str) -> tuple[str, int]:
    """Split an evaluator key such as "bear_2" into the video name and the object number.

    Args:
        key: "<video>_<object number>"; the split is at the last underscore.
    """
    video, _, number = key.rpartition("_")
    return video, int(number)


def rows_from_metrics(metrics: dict[str, Any]) -> list[ScoreRow]:
    """Turn the evaluator's metrics into score rows.

    Args:
        metrics: Result of the DAVIS evaluation; per-object J and F are read from
            metrics["J"]["M_per_object"] and metrics["F"]["M_per_object"].

    Returns:
        ScoreRow list sorted by video and object, J and F as the evaluator reports them.
    """
    j_scores = metrics["J"]["M_per_object"]
    f_scores = metrics["F"]["M_per_object"]
    rows = []
    for key, j_value in j_scores.items():
        video, number = split_key(key)
        rows.append(ScoreRow(video, number, float(j_value), float(f_scores[key])))
    return sorted_rows(rows)


def score_davis(
    davis_root: Path,
    pred_root: Path,
    videos: Sequence[str],
    resolution: str = DAVIS_RESOLUTION,
    gt_set: str = "val",
) -> list[ScoreRow]:
    """Score predicted label maps with the official DAVIS 2017 semi-supervised evaluator.

    Args:
        davis_root: DAVIS folder (with Annotations and ImageSets).
        pred_root: Folder with one subfolder of predicted label-map PNGs per video.
        videos: Video names to score.
        resolution: Image resolution folder; only "480p" is accepted.
        gt_set: DAVIS set name, for example "val".

    Returns:
        One ScoreRow per (video, object).

    Raises:
        ScoringError: If the resolution is not 480p or the evaluator stops with SystemExit.
        IncompleteError: If predictions are missing or malformed (checked before scoring).
    """
    if resolution != DAVIS_RESOLUTION:
        raise ScoringError(f"the DAVIS evaluator scores {DAVIS_RESOLUTION} only, got {resolution}")
    # Bad input makes the evaluator exit, so predictions are checked first.
    check_completeness(davis_root / "Annotations" / resolution, pred_root, videos, sav=False)
    module = load_davis()
    # The evaluator prints to stdout; capture it to report errors or keep output clean.
    sink = io.StringIO()
    try:
        with redirect_stdout(sink):
            evaluation = module.DAVISEvaluation(
                davis_root=str(davis_root), task=DAVIS_TASK, gt_set=gt_set, sequences=list(videos)
            )
            metrics = evaluation.evaluate(str(pred_root))
    except SystemExit as error:
        raise ScoringError(f"the DAVIS evaluator stopped: {sink.getvalue().strip()}") from error
    return rows_from_metrics(metrics)
