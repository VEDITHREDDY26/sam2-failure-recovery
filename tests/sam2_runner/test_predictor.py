from pathlib import Path

import numpy as np
import pytest

from src.sam2_runner.predictor import (
    build_predictor,
    checkpoint_dir,
    load_models_config,
    to_numpy,
)


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.steps: list[str] = []

    def detach(self) -> "_FakeTensor":
        self.steps.append("detach")
        return self

    def cpu(self) -> "_FakeTensor":
        self.steps.append("cpu")
        return self

    def numpy(self) -> np.ndarray:
        self.steps.append("numpy")
        return self.array


def test_to_numpy_passes_arrays_through() -> None:
    array = np.arange(3)
    assert to_numpy(array) is array


def test_to_numpy_duck_types_tensors() -> None:
    tensor = _FakeTensor(np.ones((2, 2)))
    out = to_numpy(tensor)
    assert tensor.steps == ["detach", "cpu", "numpy"]
    assert out.shape == (2, 2)


def test_models_config_paths(tmp_path: Path) -> None:
    cfg = load_models_config()
    assert checkpoint_dir(tmp_path, cfg) == tmp_path / "checkpoints"
    assert cfg.checkpoint_path(tmp_path, cfg.tiny) == tmp_path / "checkpoints/sam2.1_hiera_tiny.pt"
    assert cfg.large.model_cfg == "configs/sam2.1/sam2.1_hiera_l.yaml"


def test_build_predictor_rejects_bad_model_and_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_predictor("small", tmp_path)
    with pytest.raises(FileNotFoundError, match="get_checkpoints"):
        build_predictor("tiny", tmp_path)
