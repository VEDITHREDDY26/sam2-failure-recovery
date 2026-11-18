"""Per-frame observations of one object: the only input the rule detector sees.

Area and centre follow src.common.signals.signals_from_masks (pixel count, centroid in pixel
coordinates, NaN centre when empty). The box is the tight xyxy box of the mask (None when empty).
Observations are built online from a mask and SAM 2's object score, or offline from the signal
CSV joined with the per-frame boxes CSV; both give identical values for the same mask.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.common.boxes import Box, box_from_mask

BOX_COLS = ("box_x0", "box_y0", "box_x1", "box_y1")
KEY = ["video", "object_id", "frame_index"]
TABLE_COLUMNS = (
    "video",
    "object_id",
    "frame_index",
    "object_score",
    "area",
    "centre_x",
    "centre_y",
    "height",
    "width",
    *BOX_COLS,
)


@dataclasses.dataclass(frozen=True)
class FrameObs:
    """What the rule detector sees of one object on one frame.

    Attributes:
        object_score: SAM 2's object score for the frame (NaN if unavailable).
        area: Mask size in pixels (0 if the mask is empty).
        centre_x: Mask centroid column in pixels (NaN when the mask is empty).
        centre_y: Mask centroid row in pixels (NaN when the mask is empty).
        box: Tight box (x0, y0, x1, y1) in pixels around the mask; None when the mask is empty or
            when the box is unknown.
        height: Frame height in pixels.
        width: Frame width in pixels.
        box_unknown: True when the mask is non-empty but its box was not stored (the object was
            hidden under another label in the stored label map).
    """

    object_score: float
    area: int
    centre_x: float
    centre_y: float
    box: Box | None
    height: int
    width: int
    box_unknown: bool = False

    def __post_init__(self) -> None:
        """Check that the frame size is positive and that the box agrees with the mask area."""
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"bad frame shape {self.height}x{self.width}")
        if self.box_unknown:
            if self.area == 0 or self.box is not None:
                raise ValueError("box_unknown needs area > 0 and no box")
        elif (self.area == 0) != (self.box is None):
            raise ValueError("box must be None exactly when area == 0")

    @property
    def empty(self) -> bool:
        """True if the mask has no pixels."""
        return self.area == 0


def obs_from_mask(mask: npt.NDArray[np.bool_], object_score: float) -> FrameObs:
    """Observation for one frame from a bool HxW mask and SAM 2's object score."""
    # Centroid is the mean pixel position; an empty mask has no centre.
    ys, xs = np.nonzero(mask)
    area = int(ys.size)
    cx = float(xs.mean()) if area else float("nan")
    cy = float(ys.mean()) if area else float("nan")
    height, width = mask.shape
    return FrameObs(float(object_score), area, cx, cy, box_from_mask(mask), height, width)


def join_signals_boxes(
    signals: pd.DataFrame, boxes: pd.DataFrame, *, allow_overlapped: bool = False
) -> pd.DataFrame:
    """Per-frame table (TABLE_COLUMNS), sorted, one row per boxes row.

    Boxes cover each object from its prompt frame on; signal rows before that are dropped.
    Refuses duplicate keys, boxes rows without a signal row, a box that disagrees with the
    emptiness of the mask, and frames that are not consecutive within an object.
    With allow_overlapped, rows with area > 0 and all four box cells empty are accepted (the
    stored label map hid the object under another one); their box stays empty (see
    count_overlapped). Every other mismatch is still refused.
    """
    # Each (video, object, frame) key may appear only once in each table.
    for name, frame in (("signals", signals), ("boxes", boxes)):
        if frame.duplicated(KEY).any():
            raise ValueError(f"{name}: duplicate (video, object_id, frame_index) rows")
    sig = signals.copy()
    box = boxes.copy()
    sig["video"] = sig["video"].astype(str)
    box["video"] = box["video"].astype(str)
    box[list(BOX_COLS)] = box[list(BOX_COLS)].astype("float64")
    # Keep every boxes row and attach its signals; a missing signal row is an error below.
    merged = box.merge(
        sig[[*KEY, "object_score", "area", "centre_x", "centre_y"]],
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    if merged["area"].isna().any():
        raise ValueError("boxes rows without a matching signal row")
    # A box must be present exactly when the mask is non-empty.
    has_box = merged[list(BOX_COLS)].notna()
    if (has_box.any(axis=1) != has_box.all(axis=1)).any():
        raise ValueError("boxes: partly empty box columns")
    present = has_box["box_x0"].to_numpy(dtype=bool)
    nonzero = merged["area"].to_numpy(dtype=float) > 0
    mismatch = present != nonzero
    if allow_overlapped:
        mismatch &= ~(nonzero & ~present)
    if mismatch.any():
        raise ValueError("box present exactly when the signal area is non-zero")
    # Sort, then require consecutive frames per object so 'previous frame' is well defined.
    out = merged[list(TABLE_COLUMNS)].sort_values(KEY, kind="stable").reset_index(drop=True)
    out = out.astype({"height": "int64", "width": "int64"})
    gaps = out.groupby(["video", "object_id"])["frame_index"].diff().dropna()
    if (gaps != 1).any():
        raise ValueError("frames are not consecutive within an object")
    return out


def count_overlapped(table: pd.DataFrame) -> int:
    """Rows of a joined table with area > 0 but no box."""
    return int(((table["area"] > 0) & table["box_x0"].isna()).sum())


def obs_from_row(row: pd.Series) -> FrameObs:
    """Build an observation from one row of a joined per-frame table.

    Args:
        row: A row with the TABLE_COLUMNS fields; empty box cells are NaN.

    Returns:
        The observation; a non-empty mask with an empty box is marked as box_unknown.
    """
    box = None
    if not math.isnan(float(row["box_x0"])):
        box = Box(*(int(row[c]) for c in BOX_COLS))
    unknown = box is None and int(row["area"]) > 0
    return FrameObs(
        float(row["object_score"]),
        int(row["area"]),
        float(row["centre_x"]),
        float(row["centre_y"]),
        box,
        int(row["height"]),
        int(row["width"]),
        unknown,
    )
