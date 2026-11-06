import ast
import json
import re

import pytest

from src.sam2_runner.predictor import REPO_ROOT

NOTEBOOK = REPO_ROOT / "notebooks" / "run_sam2.ipynb"


@pytest.fixture(scope="module")
def code() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_all_six_run_commands_are_present(code: str) -> None:
    for model in ("tiny", "large"):
        for split in ("davis_val", "sav_val", "sav_test"):
            pattern = (
                rf"-m src\.sam2_runner\.run --model {model} "
                rf"--prompts \{{PROMPTS_DIR\}}/{split}\.json"
            )
            assert len(re.findall(pattern, code)) == 1, (model, split)


def test_install_builds_sam2_against_colab_torch(code: str) -> None:
    assert "SAM2_BUILD_ALLOW_ERRORS=0 pip install --no-build-isolation" in code
    assert 'l.startswith("SAM-2")' in code and "requirements-colab.txt" in code
    assert "pip uninstall -y SAM-2" in code
    requirement = next(
        line
        for line in (REPO_ROOT / "requirements-colab.txt").read_text().splitlines()
        if line.startswith("SAM-2")
    )
    assert "@2b90b9f5ceec907a1c18123530e92e794ad901a4" in requirement


def test_hole_filling_check_stops_the_notebook_on_failure(code: str) -> None:
    assert "from sam2 import _C" in code
    assert "get_connected_components" in code and "fill_holes_in_mask_scores" in code
    assert 'simplefilter("error")' in code
    assert "assert " in code and "hole filling" in code.lower()
    assert "raise RuntimeError" in code and "Send the full output" in code
    assert code.index("from sam2 import _C") < code.index("src.sam2_runner.run --model")


def test_prompts_check_and_score_cells_follow_the_runs(code: str) -> None:
    for split in ("davis_val", "sav_val", "sav_test"):
        assert f"experiments.sam2_runs.make_prompts --split {split}" in code
    last_run = code.rindex("src.sam2_runner.run --model")
    assert (
        last_run
        < code.index("src.eval.check_complete")
        < code.index("experiments.sam2_runs.score_all")
    )
    assert "get_checkpoints.sh" in code and "src.eval.get_evaluators" in code
    assert 'BRANCH = "phase-2b-sam2-runs"' in code
    assert "sam2_*.csv" in code


def test_code_cells_parse_apart_from_shell_magics() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        plain = "\n".join(
            (
                line[: len(line) - len(line.lstrip())] + "pass"
                if line.lstrip().startswith(("!", "%"))
                else line
            )
            for line in lines
        )
        ast.parse(plain)
