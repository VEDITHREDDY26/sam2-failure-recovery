import numpy as np
import numpy.typing as npt
import pytest

from src.common.boxes import Box
from src.recovery.tracker import AlignedCSRT

H, W, SIZE = 96, 128, 24


def frame_with_square(x0: int, y0: int = 30) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(0)
    frame = rng.integers(200, 230, size=(H, W, 3), dtype=np.uint8)
    frame[y0 : y0 + SIZE, x0 : x0 + SIZE] = (30, 60, 200)
    frame[y0 + 6 : y0 + 12, x0 + 4 : x0 + 20] = (240, 240, 20)
    return frame


def mask_for(x0: int, y0: int = 30) -> npt.NDArray[np.bool_]:
    mask = np.zeros((H, W), dtype=np.bool_)
    mask[y0 : y0 + SIZE, x0 : x0 + SIZE] = True
    return mask


def test_update_before_init_is_not_initialised() -> None:
    result = AlignedCSRT().update(frame_with_square(10))
    assert (result.status, result.box) == ("not_initialised", None)


def test_reinit_then_update_follows_the_square() -> None:
    tracker = AlignedCSRT()
    assert tracker.reinit(frame_with_square(10), mask_for(10), 0) == "reinitialised"
    result = tracker.update(frame_with_square(14))
    assert result.status == "ok" and result.box is not None
    assert abs(result.box.x0 - 14) <= 4 and abs(result.box.y0 - 30) <= 4


def test_reinit_moves_the_tracker_to_the_new_mask() -> None:
    tracker = AlignedCSRT()
    tracker.reinit(frame_with_square(10), mask_for(10), 0)
    tracker.reinit(frame_with_square(80), mask_for(80), 7)
    assert tracker.last_reliable_frame == 7
    assert tracker.last_reliable_box == Box(80, 30, 80 + SIZE, 30 + SIZE)
    result = tracker.update(frame_with_square(82))
    assert result.box is not None and result.box.x0 > 60


def test_empty_mask_keeps_previous_state() -> None:
    tracker = AlignedCSRT()
    tracker.reinit(frame_with_square(10), mask_for(10), 0)
    empty = np.zeros((H, W), dtype=np.bool_)
    assert tracker.reinit(frame_with_square(20), empty, 5) == "empty_mask"
    assert tracker.last_reliable_frame == 0
    assert tracker.last_reliable_box == Box(10, 30, 10 + SIZE, 30 + SIZE)
    assert tracker.update(frame_with_square(12)).status == "ok"


def test_empty_mask_before_any_init_stays_not_initialised() -> None:
    tracker = AlignedCSRT()
    assert (
        tracker.reinit(frame_with_square(10), np.zeros((H, W), dtype=np.bool_), 0) == "empty_mask"
    )
    assert not tracker.initialised
    assert tracker.update(frame_with_square(10)).status == "not_initialised"


class _StubTracker:
    def __init__(self, found: bool, rect: tuple[float, float, float, float]) -> None:
        self.found, self.rect = found, rect

    def update(self, frame: object) -> tuple[bool, tuple[float, float, float, float]]:
        return self.found, self.rect


def with_stub(found: bool, rect: tuple[float, float, float, float]) -> AlignedCSRT:
    tracker = AlignedCSRT()
    tracker.reinit(frame_with_square(10), mask_for(10), 0)
    tracker._tracker = _StubTracker(found, rect)  # type: ignore[assignment]
    return tracker


def test_tracker_failure_reports_lost() -> None:
    result = with_stub(False, (0, 0, 0, 0)).update(frame_with_square(10))
    assert (result.status, result.box) == ("lost", None)


def test_box_entirely_outside_the_frame_is_lost() -> None:
    result = with_stub(True, (W + 5, 10, 20, 20)).update(frame_with_square(10))
    assert result.status == "lost" and result.box is None


def test_partly_outside_box_is_clipped_and_raw_box_kept() -> None:
    result = with_stub(True, (W - 10, 10, 30, 20)).update(frame_with_square(10))
    assert result.status == "ok"
    assert result.box == Box(W - 10, 10, W, 30)
    assert result.raw_box == Box(W - 10, 10, W + 20, 30)


def test_non_bgr_frame_is_rejected() -> None:
    with pytest.raises(ValueError):
        AlignedCSRT().update(np.zeros((H, W), dtype=np.uint8))
    with pytest.raises(ValueError):
        AlignedCSRT().update(np.zeros((H, W, 3), dtype=np.float32))


def thin_mask(x0: int, y0: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.bool_)
    mask[y0 : y0 + h, x0 : x0 + w] = True
    return mask


@pytest.mark.parametrize(
    "x0, y0, w, h",
    [(30, 30, 1, 1), (30, 10, 1, 20), (10, 30, 20, 1), (0, 0, 1, 1), (W - 2, H - 2, 2, 2)],
)
def test_a_box_csrt_cannot_start_on_keeps_previous_state(x0: int, y0: int, w: int, h: int) -> None:
    tracker = AlignedCSRT()
    tracker.reinit(frame_with_square(10), mask_for(10), 0)
    before = (tracker.last_reliable_frame, tracker.last_reliable_box)
    assert tracker.reinit(frame_with_square(20), thin_mask(x0, y0, w, h), 5) == "init_failed"
    assert (tracker.last_reliable_frame, tracker.last_reliable_box) == before
    assert tracker.update(frame_with_square(10)).status == "ok"


def test_a_box_csrt_cannot_start_on_before_any_init_stays_not_initialised() -> None:
    tracker = AlignedCSRT()
    assert tracker.reinit(frame_with_square(10), thin_mask(30, 10, 1, 20), 0) == "init_failed"
    assert not tracker.initialised
    assert tracker.update(frame_with_square(10)).status == "not_initialised"
