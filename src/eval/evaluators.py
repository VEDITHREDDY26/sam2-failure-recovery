"""Pinned, unmodified checkouts of the official DAVIS 2017 and SA-V evaluators.

They are fetched into <repo>/third_party/ (git-ignored) by `python -m src.eval.get_evaluators`
and imported from there; they are never pip-installed and never edited.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party"
FIX_COMMAND = "python -m src.eval.get_evaluators"
SAV_MODULE_NAME = "sav_benchmark_pinned"


class EvaluatorError(RuntimeError):
    """The pinned evaluator checkout is missing, at another commit, or modified."""


@dataclass(frozen=True)
class Pin:
    name: str
    url: str
    commit: str
    sparse_paths: tuple[str, ...] = ()

    def path(self, root: Path = THIRD_PARTY) -> Path:
        return root / self.name


DAVIS_PIN = Pin(
    name="davis2017-evaluation",
    url="https://github.com/davisvideochallenge/davis2017-evaluation.git",
    commit="ac7c43fca936f9722837b7fbd337d284ba37004b",
)
SAV_PIN = Pin(
    name="sam2",
    url="https://github.com/facebookresearch/sam2.git",
    commit="2b90b9f5ceec907a1c18123530e92e794ad901a4",
    sparse_paths=("sav_dataset",),
)
PINS = (DAVIS_PIN, SAV_PIN)
SAV_BENCHMARK_FILE = Path("sav_dataset") / "utils" / "sav_benchmark.py"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise EvaluatorError(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
    return result.stdout.strip()


def check_checkout(pin: Pin, root: Path = THIRD_PARTY) -> Path:
    """Return the checkout path if it is at the pinned commit with a clean tree."""
    repo = pin.path(root)
    if not (repo / ".git").exists():
        raise EvaluatorError(f"{pin.name} checkout is missing at {repo}; run: {FIX_COMMAND}")
    head = git(repo, "rev-parse", "HEAD")
    if head != pin.commit:
        raise EvaluatorError(f"{pin.name} is at {head}, expected {pin.commit}; run: {FIX_COMMAND}")
    if git(repo, "status", "--porcelain"):
        raise EvaluatorError(f"{pin.name} has local modifications in {repo}; run: {FIX_COMMAND}")
    return repo


def load_davis(root: Path = THIRD_PARTY) -> ModuleType:
    """The official `davis2017.evaluation` module from the pinned checkout."""
    repo = check_checkout(DAVIS_PIN, root)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    spec = importlib.util.find_spec("davis2017.evaluation")
    if spec is None or spec.origin is None or not Path(spec.origin).is_relative_to(repo):
        raise EvaluatorError(f"davis2017 resolves outside {repo}; remove other installs of it")
    return importlib.import_module("davis2017.evaluation")


def load_sav_benchmark(root: Path = THIRD_PARTY) -> ModuleType:
    """The official `sav_benchmark` module, loaded by file path under a unique name."""
    repo = check_checkout(SAV_PIN, root)
    existing = sys.modules.get(SAV_MODULE_NAME)
    if existing is not None and Path(existing.__file__ or "").is_relative_to(repo):
        return existing
    spec = importlib.util.spec_from_file_location(SAV_MODULE_NAME, repo / SAV_BENCHMARK_FILE)
    if spec is None or spec.loader is None:
        raise EvaluatorError(f"cannot load {repo / SAV_BENCHMARK_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[SAV_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
