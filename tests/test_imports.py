import importlib

import pytest

PACKAGES = [
    "src",
    "src.common",
    "src.data",
    "src.baseline",
    "src.oracle",
    "src.recovery",
    "src.sam2_runner",
    "src.detection",
    "src.eval",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    assert importlib.import_module(name) is not None
