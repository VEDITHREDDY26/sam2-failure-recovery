"""Axis-aligned boxes in xyxy pixel coordinates: left/top inclusive, right/bottom exclusive.

A box with x0=3, x1=5 covers pixel columns 3 and 4. This is the only module that converts
to and from the xywh layout that OpenCV trackers use.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True)
class Box:
    """Immutable axis-aligned box in pixel coordinates, xyxy with right/bottom exclusive.

    Attributes:
        x0: Left edge (first column covered), in pixels.
        y0: Top edge (first row covered), in pixels.
        x1: Right edge, exclusive, so the box width is x1 - x0.
        y1: Bottom edge, exclusive, so the box height is y1 - y0.
    """

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        """Reject boxes with no area.

        Raises:
            ValueError: If x1 <= x0 or y1 <= y0.
        """
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"invalid box {self}: need x1 > x0 and y1 > y0")

    @property
    def width(self) -> int:
        """Width in pixels (x1 - x0)."""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """Height in pixels (y1 - y0)."""
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        """Area in pixels (width times height)."""
        return self.width * self.height


def to_xywh(box: Box) -> tuple[int, int, int, int]:
    """Convert a box to the (x, y, width, height) layout that OpenCV trackers expect.

    Args:
        box: Box in xyxy pixel coordinates.

    Returns:
        Tuple (left, top, width, height) in pixels.
    """
    return box.x0, box.y0, box.width, box.height


def from_xywh(x: int, y: int, w: int, h: int) -> Box:
    """Build a box from the (x, y, width, height) layout that OpenCV trackers return.

    Args:
        x: Left edge in pixels.
        y: Top edge in pixels.
        w: Width in pixels.
        h: Height in pixels.

    Returns:
        The equivalent xyxy Box.

    Raises:
        ValueError: If w or h is not positive.
    """
    return Box(x, y, x + w, y + h)


def box_from_mask(mask: npt.NDArray[np.bool_] | npt.NDArray[np.uint8]) -> Box | None:
    """Tight box around the nonzero pixels of an HxW mask, or None if it is empty."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    # Same scan along the other axis; +1 makes the right/bottom edge exclusive.
    cols = np.flatnonzero(mask.any(axis=0))
    return Box(int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def clip(box: Box, height: int, width: int) -> Box | None:
    """Intersect a box with the image [0, width) x [0, height); None if nothing is left."""
    x0, y0 = max(box.x0, 0), max(box.y0, 0)
    x1, y1 = min(box.x1, width), min(box.y1, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x0, y0, x1, y1)
