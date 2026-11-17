"""ViT tracker (OpenCV VitTrack) kept aligned to SAM 2's masks, same conventions as AlignedCSRT.

Boxes are xyxy outside and xywh only at the OpenCV call. The model file lives outside the repo
under the data root; the config pins its sha256 and a missing or different file is refused. The
tracker's own score threshold is set to the configured value (0 by default), so a box and a score
come back on every frame and the caller decides what to do with them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from src.common.boxes import Box, box_from_mask, clip, from_xywh, to_xywh
from src.common.config import DATA_ROOT_ENV, ConfigError, load_config, register
from src.recovery.tracker import ReinitStatus, TrackResult, _check_frame

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "recovery" / "vit.yaml"


class RawViT(Protocol):
    def init(self, image: Any, bounding_box: Any) -> Any: ...

    def update(self, image: Any) -> Any: ...

    def getTrackingScore(self) -> float: ...  # noqa: N802


ViTFactory = Callable[[], RawViT]


@register("vit_tracker")
@dataclasses.dataclass(frozen=True)
class ViTConfig:
    model: str  # path relative to the data root
    sha256: str
    anchor_iou: float
    score_threshold_internal: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.anchor_iou <= 1.0:
            raise ConfigError("anchor_iou must be in [0, 1]")
        if len(self.sha256) != 64:
            raise ConfigError("sha256 must be a 64-character hex digest")


def load_vit_config(path: Path = CONFIG_PATH) -> ViTConfig:
    cfg = load_config(path)
    if not isinstance(cfg, ViTConfig):
        raise ConfigError(f"{path}: not a vit_tracker config")
    return cfg


def vit_model_path(cfg: ViTConfig, data_root: Path | None = None) -> Path:
    """Resolve and verify the model file; raises ConfigError if it is missing or altered."""
    if data_root is None:
        value = os.environ.get(DATA_ROOT_ENV, "").strip()
        if not value:
            raise ConfigError(f"environment variable {DATA_ROOT_ENV} is not set")
        data_root = Path(value)
    path = data_root / cfg.model
    if not path.is_file():
        raise ConfigError(f"ViT model file not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != cfg.sha256:
        raise ConfigError(f"ViT model {path}: sha256 {digest} does not match the pinned value")
    return path


def make_vit_factory(cfg: ViTConfig, data_root: Path | None = None) -> ViTFactory:
    model = vit_model_path(cfg, data_root)

    def factory() -> RawViT:
        params = cv2.TrackerVit_Params()  # type: ignore[attr-defined]
        params.net = str(model)
        params.tracking_score_threshold = cfg.score_threshold_internal
        return cv2.TrackerVit_create(params)  # type: ignore[attr-defined, no-any-return]

    return factory


class ViTTracker:
    def __init__(self, factory: ViTFactory) -> None:
        self._factory = factory
        self._tracker: RawViT | None = None
        self._shape = (0, 0)
        self.last_reliable_frame: int | None = None
        self.last_reliable_box: Box | None = None

    @property
    def initialised(self) -> bool:
        return self._tracker is not None

    def reinit(
        self, frame_bgr: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_], frame_idx: int
    ) -> ReinitStatus:
        """Restart on the box of a reliable mask; an empty mask or a refused box keeps the state."""
        _check_frame(frame_bgr)
        box = box_from_mask(mask)
        if box is None:
            return "empty_mask"
        tracker = self._factory()
        try:
            tracker.init(frame_bgr, to_xywh(box))
        except cv2.error:
            return "init_failed"
        self._tracker = tracker
        self._shape = (frame_bgr.shape[0], frame_bgr.shape[1])
        self.last_reliable_frame = frame_idx
        self.last_reliable_box = box
        return "reinitialised"

    def update(self, frame_bgr: npt.NDArray[np.uint8]) -> TrackResult:
        _check_frame(frame_bgr)
        if self._tracker is None:
            return TrackResult("not_initialised", None, None)
        try:
            found, rect = self._tracker.update(frame_bgr)
        except cv2.error:
            return TrackResult("lost", None, None)
        if not found:
            return TrackResult("lost", None, None)
        x, y, w, h = (int(round(v)) for v in rect)
        if w <= 0 or h <= 0:
            return TrackResult("lost", None, None)
        score = float(self._tracker.getTrackingScore())
        raw = from_xywh(x, y, w, h)
        clipped = clip(raw, *self._shape)
        if clipped is None:
            return TrackResult("lost", None, raw, score)
        return TrackResult("ok", clipped, raw, score)
