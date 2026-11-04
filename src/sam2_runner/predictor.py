"""Seam between this project and SAM 2's video predictor.

The protocol mirrors the three SAM2VideoPredictor methods the project calls (pinned source:
sam2/sam2_video_predictor.py). torch and sam2 are imported only inside `build_predictor`,
so everything else runs on a CPU-only machine with a fake predictor.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from src.common.config import ConfigError, ModelsConfig, load_config

MODELS = ("tiny", "large")
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"


class VideoPredictor(Protocol):
    def init_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> Any: ...

    def add_new_points_or_box(
        self,
        inference_state: Any,
        frame_idx: int,
        obj_id: int,
        points: Any = None,
        labels: Any = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        box: Sequence[float] | None = None,
    ) -> tuple[int, list[int], Any]: ...

    def propagate_in_video(
        self,
        inference_state: Any,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], Any]]: ...


def to_numpy(value: Any) -> npt.NDArray[Any]:
    """Numpy array from a numpy array or a torch tensor (detach, move to CPU, convert)."""
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value.detach().cpu().numpy())


def load_models_config(path: Path = MODELS_CONFIG) -> ModelsConfig:
    cfg = load_config(path)
    if not isinstance(cfg, ModelsConfig):
        raise ConfigError(f"{path}: not a models config")
    return cfg


def checkpoint_dir(data_root: Path, cfg: ModelsConfig | None = None) -> Path:
    cfg = cfg or load_models_config()
    return data_root / cfg.checkpoint_dir


def build_predictor(
    model: str,
    data_root: Path,
    device: str = "cuda",
    cfg: ModelsConfig | None = None,
) -> VideoPredictor:
    """Build SAM2VideoPredictor for 'tiny' or 'large' (needs torch and sam2; Colab only)."""
    if model not in MODELS:
        raise ValueError(f"unknown model '{model}': expected one of {MODELS}")
    cfg = cfg or load_models_config()
    spec = cfg.tiny if model == "tiny" else cfg.large
    ckpt = cfg.checkpoint_path(data_root, spec)
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} not found; run scripts/get_checkpoints.sh")
    build = importlib.import_module("sam2.build_sam").build_sam2_video_predictor
    predictor: VideoPredictor = build(spec.model_cfg, str(ckpt), device=device)
    return predictor
