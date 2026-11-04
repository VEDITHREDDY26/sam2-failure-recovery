import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.sam2_runner.predictor import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "get_checkpoints.sh"


def _run(env_root: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "SAM2FR_DATA_ROOT"}
    env["PYTHON"] = sys.executable
    if env_root is not None:
        env["SAM2FR_DATA_ROOT"] = str(env_root)
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, check=False
    )


def test_fails_clearly_without_data_root() -> None:
    result = _run(None, "--print-dir")
    assert result.returncode != 0
    assert "SAM2FR_DATA_ROOT" in result.stderr


def test_print_dir_uses_config(tmp_path: Path) -> None:
    result = _run(tmp_path, "--print-dir")
    assert result.returncode == 0
    assert result.stdout.strip() == str(tmp_path / "checkpoints")


@pytest.mark.parametrize("name", ["sam2.1_hiera_tiny.pt", "sam2.1_hiera_large.pt"])
def test_existing_files_are_skipped_without_network(tmp_path: Path, name: str) -> None:
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    for n in ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_large.pt"):
        (ckpt / n).write_bytes(b"x")
    result = _run(tmp_path)
    assert result.returncode == 0
    assert f"skip {name}" in result.stdout
    assert "downloading" not in result.stdout
