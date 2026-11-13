import ast
from pathlib import Path

import pytest

from src.sam2_runner.predictor import REPO_ROOT

ORACLE = "src.oracle"
CHECKED = ("src/recovery", "src/detection")


def _modules(tree: ast.AST, package: str) -> set[str]:
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


def _is_oracle(module: str) -> bool:
    return module == ORACLE or module.startswith(ORACLE + ".")


def _files() -> list[Path]:
    return [p for folder in CHECKED for p in sorted((REPO_ROOT / folder).rglob("*.py"))]


def _package_of(path: Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts[:-1])


def test_checked_folders_contain_python_files() -> None:
    names = {p.parent.name for p in _files()}
    assert {"recovery", "detection"} <= names
    assert len(_files()) >= 4


def test_recovery_and_detection_never_import_the_oracle() -> None:
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in _files()
        if any(_is_oracle(m) for m in _modules(ast.parse(p.read_text()), _package_of(p)))
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("source", "package"),
    [
        ("import src.oracle", "src.recovery"),
        ("import src.oracle.flags as flags", "src.recovery"),
        ("from src.oracle import oracle_flag", "src.recovery"),
        ("from src.oracle.flags import oracle_box", "src.detection"),
        ("from src import oracle", "src.recovery"),
        ("from .. import oracle", "src.recovery"),
        ("from ..oracle import flags", "src.recovery"),
        ("from ...oracle.flags import oracle_flag", "src.detection.sub"),
    ],
)
def test_the_scan_catches_every_import_form(source: str, package: str) -> None:
    assert any(_is_oracle(m) for m in _modules(ast.parse(source), package))


def test_the_scan_ignores_unrelated_imports() -> None:
    tree = ast.parse("from src.common import boxes\nimport src.oraclex")
    assert not any(_is_oracle(m) for m in _modules(tree, "src.recovery"))
