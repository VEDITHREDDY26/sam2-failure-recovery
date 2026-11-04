import shutil
from pathlib import Path

import pytest

from src.eval.evaluators import (
    DAVIS_PIN,
    FIX_COMMAND,
    PINS,
    SAV_BENCHMARK_FILE,
    SAV_PIN,
    THIRD_PARTY,
    EvaluatorError,
    check_checkout,
    git,
    load_davis,
    load_sav_benchmark,
)


def _copy(tmp_path: Path, name: str) -> Path:
    root = tmp_path / "third_party"
    root.mkdir(exist_ok=True)
    shutil.copytree(THIRD_PARTY / name, root / name)
    return root


def test_real_checkouts_are_pinned_and_clean() -> None:
    for pin in PINS:
        assert check_checkout(pin) == THIRD_PARTY / pin.name


def test_loaders_return_official_classes() -> None:
    assert load_davis().DAVISEvaluation.__name__ == "DAVISEvaluation"
    assert load_sav_benchmark().VideoEvaluator.__name__ == "VideoEvaluator"


def test_missing_checkout_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(EvaluatorError, match="missing") as info:
        check_checkout(DAVIS_PIN, tmp_path)
    assert FIX_COMMAND in str(info.value)


def test_modified_davis_checkout_is_refused(tmp_path: Path) -> None:
    root = _copy(tmp_path, DAVIS_PIN.name)
    target = root / DAVIS_PIN.name / "davis2017" / "metrics.py"
    target.write_text(target.read_text() + "\n# edited\n")
    with pytest.raises(EvaluatorError, match="modifications"):
        load_davis(root)


def test_modified_sav_checkout_is_refused(tmp_path: Path) -> None:
    root = _copy(tmp_path, SAV_PIN.name)
    target = root / SAV_PIN.name / SAV_BENCHMARK_FILE
    target.write_text(target.read_text() + "\n# edited\n")
    with pytest.raises(EvaluatorError, match="modifications"):
        load_sav_benchmark(root)


def test_wrong_commit_is_refused(tmp_path: Path) -> None:
    root = _copy(tmp_path, DAVIS_PIN.name)
    repo = root / DAVIS_PIN.name
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--allow-empty", "-qm", "x")
    with pytest.raises(EvaluatorError, match="expected"):
        check_checkout(DAVIS_PIN, root)
