from pathlib import Path

import numpy as np
import pytest

from src.common.layout import signals_path
from src.common.signals import read_signals
from src.sam2_runner.scores import fallback_score, object_score, read_stored_score
from src.sam2_runner.smoke import SPLIT, Prompts, run_smoke
from tests.fixtures.synthetic_video import OBJECT2_BOX, SyntheticVideo
from tests.sam2_runner.fakes import NEGATIVE, POSITIVE, FakePredictor

METHOD = "sam2_tiny_smoke"


class _Tensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array


def _state(frame_out: object, key: str = "non_cond_frame_outputs") -> dict:
    per_obj: dict[str, dict] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    per_obj[key][3] = frame_out
    return {"obj_id_to_idx": {7: 1}, "output_dict_per_obj": {1: per_obj}}


def test_read_stored_score_from_tensor_like_and_cond_frames() -> None:
    state = _state({"object_score_logits": _Tensor(np.array([[2.5]], dtype=np.float32))})
    assert read_stored_score(state, 7, 3) == 2.5
    cond = _state({"object_score_logits": np.array([[-1.5]])}, "cond_frame_outputs")
    assert read_stored_score(cond, 7, 3) == -1.5


def test_read_stored_score_missing_cases_and_current_frame_only() -> None:
    state = _state({"object_score_logits": np.array([[2.5]])})
    assert read_stored_score(state, 7, 4) is None
    assert read_stored_score(state, 8, 3) is None
    assert read_stored_score(_state({"object_score_logits": None}), 7, 3) is None
    assert read_stored_score(_state({"pred_masks": 1}), 7, 3) is None
    assert read_stored_score({}, 7, 3) is None
    with pytest.raises(ValueError):
        read_stored_score(_state({"object_score_logits": np.zeros(2)}), 7, 3)


def test_fallback_is_mean_foreground_logit_or_max_when_empty() -> None:
    logits = np.array([[4.0, 2.0, -3.0], [-1.0, 0.0, 6.0]])
    assert fallback_score(logits) == pytest.approx(4.0)
    assert fallback_score(np.array([[-5.0, -2.0], [-9.0, 0.0]])) == 0.0
    assert fallback_score(np.full((2, 2), -1024.0)) == -1024.0


def test_object_score_prefers_stored_value() -> None:
    state = _state({"object_score_logits": np.array([[9.0]])})
    assert object_score(state, 7, 3, np.full((2, 2), 1.0)) == (9.0, True)
    assert object_score({}, 7, 3, np.full((2, 2), 1.0)) == (1.0, False)


def _prompts(video: SyntheticVideo) -> Prompts:
    box1 = video.facts[0].object1_box
    assert box1 is not None
    return Prompts(video.name, 0, {2: OBJECT2_BOX, 1: box1})


def test_stored_scores_reach_the_csv_including_frame_zero(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    n = synthetic_video.num_frames
    stored = {(o, f): 100.0 * o + f + 0.25 for o in (1, 2) for f in range(n)}
    stored[(1, 5)] = _Tensor(np.array([[-7.5]], dtype=np.float32))
    fake = FakePredictor(synthetic_video, stored_scores=stored)
    result = run_smoke(
        fake, synthetic_video.frames_dir, _prompts(synthetic_video), tmp_path, METHOD
    )
    assert fake.state["obj_id_to_idx"] == {2: 0, 1: 1}
    df = read_signals(signals_path(tmp_path, METHOD, SPLIT, synthetic_video.name))
    for o in (1, 2):
        for f in range(n):
            got = df[(df.object_id == o) & (df.frame_index == f)].iloc[0].object_score
            want = -7.5 if (o, f) == (1, 5) else 100.0 * o + f + 0.25
            assert got == pytest.approx(want)
    assert result.stored_rows == 2 * n
    assert (
        result.source_summary
        == f"object_score source: object_score_logits for {2 * n}/{2 * n} rows"
    )


def test_fallback_values_in_csv_when_scores_absent(
    synthetic_video: SyntheticVideo, tmp_path: Path
) -> None:
    fake = FakePredictor(synthetic_video)
    result = run_smoke(
        fake, synthetic_video.frames_dir, _prompts(synthetic_video), tmp_path, METHOD
    )
    df = read_signals(signals_path(tmp_path, METHOD, SPLIT, synthetic_video.name))
    n = synthetic_video.num_frames
    assert result.stored_rows == 0
    assert "fallback for 40" in result.source_summary
    for f in range(n):
        row1 = df[(df.object_id == 1) & (df.frame_index == f)].iloc[0]
        want = NEGATIVE if synthetic_video.facts[f].object1_area == 0 else POSITIVE
        assert row1.object_score == want
        assert df[(df.object_id == 2) & (df.frame_index == f)].iloc[0].object_score == POSITIVE
    assert not df["object_score"].isna().any()
