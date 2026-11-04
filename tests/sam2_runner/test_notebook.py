import json

from src.sam2_runner.predictor import REPO_ROOT


def test_notebook_is_valid_and_runs_smoke_for_both_models() -> None:
    nb = json.loads((REPO_ROOT / "notebooks" / "sam2_smoke.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "-m src.sam2_runner.smoke --model tiny" in code
    assert "-m src.sam2_runner.smoke --model large" in code
    assert "get_checkpoints.sh" in code
    assert "requirements-colab.txt" in code
    assert 'BRANCH = "phase-0b-sam2"' in code
    assert "inspect.getsourcefile" in code
    assert '"object_score_logits" in line' in code
