import json

from src.sam2_runner.predictor import REPO_ROOT


def test_notebook_is_valid_and_runs_spike_for_both_models() -> None:
    nb = json.loads((REPO_ROOT / "notebooks" / "reprompt_spike.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "-m experiments.spikes.reprompt --model tiny" in code
    assert "-m experiments.spikes.reprompt --model large" in code
    assert "get_checkpoints.sh" in code
    assert "requirements-colab.txt" in code
    assert 'BRANCH = "phase-1a-data"' in code
    assert "SAM2FR_DATA_ROOT" in code
    assert "output_path" in code
