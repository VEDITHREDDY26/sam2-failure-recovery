"""Verified recovery: accept a re-prompt only when SAM 2's own outputs at frame t confirm it.

The candidate is the first output at frame t of the restarted stream. Its features use only the
candidate, the flagged mask at t, the area of the last reliable mask and the box that was used
as the prompt; nothing after t and no ground truth. The rule is fixed: accept iff
cand_score >= s_min and area_ratio <= r_max and box_iou >= b_min. A non-finite threshold means
that check always passes.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.common.boxes import Box, box_from_mask
from src.common.config import ConfigError, load_config, register
from src.common.layout import run_dir, validate_name

MODELS = ("tiny", "large")
VERIFY_PATH = Path(__file__).resolve().parents[2] / "configs" / "verify" / "verify.yaml"
CANDIDATE_COLUMNS = (
    "video",
    "object_id",
    "frame_index",
    "cand_score",
    "orig_score",
    "area_ratio",
    "box_iou",
    "mask_iou_orig",
    "accepted",
    "reason",
)
ACCEPT_NOTHING = "accept_nothing"
Mask = npt.NDArray[np.bool_]


@dataclasses.dataclass(frozen=True)
class VerifyThresholds:
    """Frozen acceptance thresholds for verified recovery (one set per SAM 2 model).

    Attributes:
        s_min: Least SAM 2 object score of the candidate; non-finite = check always passes.
        r_max: Largest allowed area ratio (candidate vs last reliable mask); non-finite = off.
        b_min: Least IoU between the candidate's mask box and the prompt box; non-finite = off.
        accept_nothing: If True every candidate is rejected (the fallback when no setting helped).
    """

    s_min: float
    r_max: float
    b_min: float
    accept_nothing: bool

    def failed(self, cand_score: float, area_ratio: float, box_iou: float) -> list[str]:
        """Names of the failed checks (all of them when the setting is accept nothing)."""
        if self.accept_nothing:
            return [ACCEPT_NOTHING]
        # Each check applies only when its threshold is finite; NaN scores fail the score check.
        failed: list[str] = []
        if math.isfinite(self.s_min) and not cand_score >= self.s_min:
            failed.append("cand_score")
        if math.isfinite(self.r_max) and not area_ratio <= self.r_max:
            failed.append("area_ratio")
        if math.isfinite(self.b_min) and not box_iou >= self.b_min:
            failed.append("box_iou")
        return failed


@register("verify")
@dataclasses.dataclass(frozen=True)
class VerifyConfig:
    """Schema of the verify config file.

    Attributes:
        tiny: Thresholds for Hiera-Tiny.
        large: Thresholds for Hiera-Large.
    """

    tiny: VerifyThresholds
    large: VerifyThresholds

    def for_model(self, model: str) -> VerifyThresholds:
        """Pick the thresholds of one model.

        Args:
            model: "tiny" or "large".

        Returns:
            The thresholds of that model.

        Raises:
            ConfigError: If the model name is unknown.
        """
        if model not in MODELS:
            raise ConfigError(f"unknown model '{model}' (known: {list(MODELS)})")
        return self.tiny if model == "tiny" else self.large


def load_verify_thresholds(model: str, path: str | Path = VERIFY_PATH) -> VerifyThresholds:
    """Load the frozen verification thresholds of one model.

    Args:
        model: "tiny" or "large".
        path: Thresholds file; defaults to configs/verify/verify.yaml.

    Returns:
        The thresholds of that model.

    Raises:
        ConfigError: If the file does not exist yet, is invalid, or the model is unknown.
    """
    if not Path(path).is_file():
        raise ConfigError(f"verification not frozen yet: {path} does not exist")
    config = load_config(path)
    if not isinstance(config, VerifyConfig):
        raise ConfigError(f"{path}: not a verify config")
    return config.for_model(model)


@dataclasses.dataclass(frozen=True)
class CandidateRow:
    """One logged candidate of a re-prompt at frame t, with the features used to judge it.

    Attributes:
        video: Video name.
        object_id: Object id.
        frame_index: 0-based frame index of the re-prompt.
        cand_score: Object score of the candidate (first output of the restarted stream).
        orig_score: Object score of the flagged output before the re-prompt.
        area_ratio: Candidate area vs last reliable mask area (always >= 1, inf if either is 0).
        box_iou: IoU of the candidate's mask box with the prompt box.
        mask_iou_orig: Mask IoU between the candidate and the flagged mask.
        accepted: True if every enabled check passed.
        reason: Comma-separated names of the failed checks (empty if accepted).
    """

    video: str
    object_id: int
    frame_index: int
    cand_score: float
    orig_score: float
    area_ratio: float
    box_iou: float
    mask_iou_orig: float
    accepted: bool
    reason: str

    def as_row(self) -> tuple[object, ...]:
        """Tuple of the fields in CANDIDATE_COLUMNS order."""
        return dataclasses.astuple(self)


def area_ratio(candidate_area: int, reliable_area: int) -> float:
    """Symmetric size change between two masks.

    Args:
        candidate_area: Pixel count of the candidate mask.
        reliable_area: Pixel count of the last reliable mask.

    Returns:
        max(a/b, b/a), or infinity if either area is zero.
    """
    if candidate_area == 0 or reliable_area == 0:
        return math.inf
    return max(candidate_area / reliable_area, reliable_area / candidate_area)


def box_iou(a: Box | None, b: Box | None) -> float:
    """Intersection over union of two boxes.

    Args:
        a: First box in xyxy pixels, or None.
        b: Second box in xyxy pixels, or None.

    Returns:
        IoU in [0, 1]; 0.0 if either box is None or they do not overlap.
    """
    if a is None or b is None:
        return 0.0
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a.area + b.area - inter)


def mask_iou(a: Mask, b: Mask) -> float:
    """Intersection over union of two boolean masks.

    Args:
        a: First mask.
        b: Second mask, same shape.

    Returns:
        IoU in [0, 1]; 0.0 if both are empty.
    """
    union = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / union if union else 0.0


def candidate_features(
    candidate: Mask, flagged: Mask, prompt_box: Box, reliable_area: int
) -> tuple[float, float, float]:
    """(area_ratio, box_iou, mask_iou_orig) of a candidate at frame t."""
    return (
        area_ratio(int(np.count_nonzero(candidate)), reliable_area),
        box_iou(box_from_mask(candidate), prompt_box),
        mask_iou(candidate, flagged),
    )


def candidate_path(output_root: Path, method: str, split: str, video: str) -> Path:
    """Path of the candidate log CSV of one video.

    Args:
        output_root: Root folder for run outputs.
        method: Method name.
        split: Split name.
        video: Video name.

    Returns:
        The path candidates/<video>.csv under the run folder.
    """
    name = validate_name(video, "video")
    return run_dir(output_root, method, split) / "candidates" / f"{name}.csv"


def write_candidates(path: Path, rows: Sequence[CandidateRow]) -> None:
    """Write the candidate rows of one video as CSV, creating folders as needed.

    Args:
        path: Destination CSV path.
        rows: Candidates in the order they were judged (may be empty).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.as_row() for r in rows], columns=list(CANDIDATE_COLUMNS)).to_csv(
        path, index=False
    )
