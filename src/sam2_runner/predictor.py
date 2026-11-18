"""Seam between this project and SAM 2's video predictor.

The protocol mirrors the four SAM2VideoPredictor methods the project calls (pinned source:
sam2/sam2_video_predictor.py; reset_state clears all objects and prompts but keeps the loaded
frames). torch and sam2 are imported only inside `build_predictor`, so everything else runs on
a CPU-only machine with a fake predictor.
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
    """Interface of the SAM 2 video predictor methods this project calls.

    A fake with the same four methods can stand in for it, so the runner is testable on a CPU.
    """

    def init_state(
        self,
        video_path: str,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> Any:
        """Load a video's JPEG frames and return SAM 2's inference state.

        Args:
            video_path: Folder of JPEG frames named by their numeric stem.
            offload_video_to_cpu: Keep the decoded frames in CPU memory instead of on the GPU.
            offload_state_to_cpu: Keep the inference state in CPU memory.
            async_loading_frames: Load frames in a background thread.

        Returns:
            SAM 2's opaque inference state for this video.
        """
        ...

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
    ) -> tuple[int, list[int], Any]:
        """Prompt one object on one frame and return its mask logits for that frame.

        Args:
            inference_state: State returned by init_state.
            frame_idx: 0-based index of the frame that gets the prompt.
            obj_id: Object id the prompt belongs to.
            points: Optional point prompts (this project prompts with boxes only).
            labels: Labels of the points.
            clear_old_points: Drop earlier prompts of this object on this frame.
            normalize_coords: Whether SAM 2 rescales the coordinates by the video size.
            box: Box corners [x0, y0, x1, y1] in pixels of the original frame.

        Returns:
            (frame index, object ids, mask logits) for the prompted frame.
        """
        ...

    def propagate_in_video(
        self,
        inference_state: Any,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, list[int], Any]]:
        """Track the prompted objects through the video, yielding one result per frame.

        Args:
            inference_state: State returned by init_state.
            start_frame_idx: 0-based frame to start from; None starts from the earliest frame
                that has a prompt.
            max_frame_num_to_track: How many frames to track after start_frame_idx (so this many
                plus one in total, counting the start frame); None means to the end of the video.
            reverse: Track backwards instead of forwards.

        Yields:
            (frame index, object ids, mask logits) for each tracked frame.
        """
        ...

    def reset_state(self, inference_state: Any) -> None:
        """Clear all objects and prompts but keep the loaded frames.

        Args:
            inference_state: State returned by init_state.
        """
        ...


def to_numpy(value: Any) -> npt.NDArray[Any]:
    """Numpy array from a numpy array or a torch tensor (detach, move to CPU, convert)."""
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value.detach().cpu().numpy())


def load_models_config(path: Path = MODELS_CONFIG) -> ModelsConfig:
    """Load the model settings (checkpoint folder and the tiny / large entries).

    Args:
        path: YAML file to read; defaults to configs/models.yaml.

    Returns:
        The parsed models config.

    Raises:
        ConfigError: If the file is not a models config.
    """
    cfg = load_config(path)
    if not isinstance(cfg, ModelsConfig):
        raise ConfigError(f"{path}: not a models config")
    return cfg


def checkpoint_dir(data_root: Path, cfg: ModelsConfig | None = None) -> Path:
    """Return the folder under the data root that holds the checkpoints.

    Args:
        data_root: Root folder of the data (Drive folder on Colab).
        cfg: Models config; loaded from configs/models.yaml when None.

    Returns:
        data_root joined with the checkpoint folder name from the config.
    """
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
    # Fail early with a clear message if the checkpoint has not been downloaded.
    ckpt = cfg.checkpoint_path(data_root, spec)
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} not found; run scripts/get_checkpoints.sh")
    # Import sam2 here so the rest of the project works without torch or sam2.
    build = importlib.import_module("sam2.build_sam").build_sam2_video_predictor
    predictor: VideoPredictor = build(spec.model_cfg, str(ckpt), device=device)
    return predictor
