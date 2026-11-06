import ast
from pathlib import Path

import pytest

from src.sam2_runner.predictor import REPO_ROOT

LABELS = "src.detection.labels"
CHECKED = ("src/detection", "src/recovery", "src/sam2_runner", "src/baseline", "src/oracle")


def _modules(tree: ast.AST, package: str) -> set[str]:
    """Every dotted module name the file imports, including names imported from a package."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(parts + ([node.module] if node.module else []))
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _package_of(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(rel.parts[:-1])


def _imports(path: Path) -> set[str]:
    return _modules(ast.parse(path.read_text()), _package_of(path))


def _files() -> list[Path]:
    files = []
    for folder in CHECKED:
        for path in sorted((REPO_ROOT / folder).rglob("*.py")):
            if path != REPO_ROOT / "src" / "detection" / "labels.py":
                files.append(path)
    return files


def test_checked_folders_contain_python_files() -> None:
    assert len(_files()) > 10


def test_nothing_else_imports_the_labels() -> None:
    offenders = [str(p.relative_to(REPO_ROOT)) for p in _files() if LABELS in _imports(p)]
    assert offenders == []


def test_labels_do_not_import_oracle_code() -> None:
    labels = REPO_ROOT / "src" / "detection" / "labels.py"
    assert not [m for m in _imports(labels) if m == "src.oracle" or m.startswith("src.oracle.")]


@pytest.mark.parametrize(
    "source",
    [
        "import src.detection.labels",
        "import src.detection.labels as lab",
        "from src.detection.labels import is_failure",
        "from src.detection import labels",
        "from . import labels",
        "from .labels import FAILURE_IOU",
    ],
)
def test_the_scan_catches_every_import_form(source: str) -> None:
    assert LABELS in _modules(ast.parse(source), "src.detection")
