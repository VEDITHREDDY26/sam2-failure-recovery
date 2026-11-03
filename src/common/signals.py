"""Signal CSV: one row per (video, object_id, frame_index), fixed column order.

object_score is SAM 2's object-score logit (or its fallback; NaN only if unavailable).
area is the mask pixel count; centre_x / centre_y is the centroid in pixel coordinates
(NaN when the mask is empty); empty is written as 0/1 and is true iff area == 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

SIGNAL_COLUMNS: tuple[str, ...] = (
    "video",
    "object_id",
    "frame_index",
    "object_score",
    "area",
    "centre_x",
    "centre_y",
    "empty",
)
_KEY = ["video", "object_id", "frame_index"]
_INT_COLUMNS = ("object_id", "frame_index", "area")
_FLOAT_COLUMNS = ("object_score", "centre_x", "centre_y")


def _validate(df: pd.DataFrame, where: str) -> pd.DataFrame:
    if tuple(df.columns) != SIGNAL_COLUMNS:
        raise ValueError(f"{where}: columns {list(df.columns)} != {list(SIGNAL_COLUMNS)}")
    out = df.copy()
    try:
        out["video"] = out["video"].astype(str)
        for name in _INT_COLUMNS:
            if out[name].isna().any():
                raise ValueError(f"column {name} has missing values")
            out[name] = out[name].astype(np.int64)
        for name in _FLOAT_COLUMNS:
            out[name] = out[name].astype(np.float64)
        flags = out["empty"]
        if not flags.isin([0, 1, True, False]).all():
            raise ValueError("column empty must be 0/1")
        out["empty"] = flags.astype(bool)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: bad dtype ({exc})") from exc
    if out.duplicated(_KEY).any():
        raise ValueError(f"{where}: duplicate (video, object_id, frame_index) rows")
    if (out["empty"] != (out["area"] == 0)).any():
        raise ValueError(f"{where}: empty must be true exactly when area == 0")
    if (out["area"] < 0).any():
        raise ValueError(f"{where}: area must be non-negative")
    if (out["centre_x"].isna() != out["empty"]).any() or (
        out["centre_y"].isna() != out["empty"]
    ).any():
        raise ValueError(f"{where}: centre must be NaN exactly when the mask is empty")
    return out


def write_signals(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    clean = _validate(df, str(path))
    clean["empty"] = clean["empty"].astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(path, index=False, float_format=None, na_rep="nan")


def read_signals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, float_precision="round_trip", dtype={"video": str})
    return _validate(df, str(path))


def signals_from_masks(
    video: str,
    object_id: int,
    masks: Sequence[npt.NDArray[np.bool_]],
    object_scores: Sequence[float],
) -> pd.DataFrame:
    """Signal rows for one object over consecutive frames (frame_index = list position)."""
    if len(masks) != len(object_scores):
        raise ValueError(f"{len(masks)} masks but {len(object_scores)} object scores")
    rows = []
    for index, (mask, score) in enumerate(zip(masks, object_scores, strict=True)):
        ys, xs = np.nonzero(mask)
        area = int(ys.size)
        cx = float(xs.mean()) if area else float("nan")
        cy = float(ys.mean()) if area else float("nan")
        rows.append((video, object_id, index, float(score), area, cx, cy, area == 0))
    return _validate(pd.DataFrame(rows, columns=list(SIGNAL_COLUMNS)), "signals_from_masks")
