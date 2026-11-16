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
    s_min: float
    r_max: float
    b_min: float
    accept_nothing: bool

    def failed(self, cand_score: float, area_ratio: float, box_iou: float) -> list[str]:
        """Names of the failed checks (all of them when the setting is accept nothing)."""
        if self.accept_nothing:
            return [ACCEPT_NOTHING]
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
    tiny: VerifyThresholds
    large: VerifyThresholds

    def for_model(self, model: str) -> VerifyThresholds:
        if model not in MODELS:
            raise ConfigError(f"unknown model '{model}' (known: {list(MODELS)})")
        return self.tiny if model == "tiny" else self.large


def load_verify_thresholds(model: str, path: str | Path = VERIFY_PATH) -> VerifyThresholds:
    if not Path(path).is_file():
        raise ConfigError(f"verification not frozen yet: {path} does not exist")
    config = load_config(path)
    if not isinstance(config, VerifyConfig):
        raise ConfigError(f"{path}: not a verify config")
    return config.for_model(model)


@dataclasses.dataclass(frozen=True)
class CandidateRow:
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
        return dataclasses.astuple(self)


def area_ratio(candidate_area: int, reliable_area: int) -> float:
    if candidate_area == 0 or reliable_area == 0:
        return math.inf
    return max(candidate_area / reliable_area, reliable_area / candidate_area)


def box_iou(a: Box | None, b: Box | None) -> float:
    if a is None or b is None:
        return 0.0
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a.area + b.area - inter)


def mask_iou(a: Mask, b: Mask) -> float:
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
    name = validate_name(video, "video")
    return run_dir(output_root, method, split) / "candidates" / f"{name}.csv"


def write_candidates(path: Path, rows: Sequence[CandidateRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.as_row() for r in rows], columns=list(CANDIDATE_COLUMNS)).to_csv(
        path, index=False
    )
