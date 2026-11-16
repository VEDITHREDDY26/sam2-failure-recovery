from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.common.boxes import Box
from src.common.config import DataConfig
from src.common.layout import events_path, mask_path, masks_dir, signals_path
from src.common.masks import read_label_map
from src.common.signals import SIGNAL_COLUMNS, read_signals
from src.recovery.box_check import BoxCheck
from src.recovery.events import EVENT_COLUMNS
from src.recovery.online_runner import (
    ONLINE_TIMING_COLUMNS,
    check_online_method,
    is_online_complete,
    run_batch,
    run_video,
)
from src.recovery.settings import RecoverySettings
from src.sam2_runner.prompts import BatchPrompts
from src.sam2_runner.timing import timing_path
from tests.recovery.fakes import (
    NUM_FRAMES,
    ScriptedBoxes,
    ScriptedDetector,
    StatefulFake,
    expected_mask,
    ok,
)
from tests.recovery.online_helpers import METHOD, SPLIT, START_BOX, VIDEO, make_tree, tree_hash

NEW_BOX = Box(30, 20, 42, 32)
SETTINGS = RecoverySettings(12, "reinit")


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DataConfig, BatchPrompts, Path]:
    cfg, prompts = make_tree(tmp_path, monkeypatch)
    return cfg, prompts, cfg.data_root / cfg.output_root


def batch(
    cfg: DataConfig,
    prompts: BatchPrompts,
    predictor: StatefulFake | None = None,
    flags: dict[tuple[int, int], tuple[str, ...]] | None = None,
    method: str = METHOD,
) -> tuple[int, int]:
    pred = predictor or StatefulFake()
    return run_batch(
        prompts,
        cfg,
        method,
        lambda: pred,
        lambda video: ScriptedDetector(flags),
        lambda video: ScriptedBoxes({(1, 5): ok(NEW_BOX)}),
        BoxCheck(),
        SETTINGS,
    )


def test_outputs_written_in_the_documented_layout(tree: Any) -> None:
    cfg, prompts, out = tree
    assert batch(cfg, prompts, flags={(1, 5): ("low_score",)}) == (1, 0)
    signals = read_signals(signals_path(out, METHOD, SPLIT, VIDEO))
    assert list(signals.columns) == list(SIGNAL_COLUMNS)
    assert len(signals) == NUM_FRAMES
    assert signals["object_score"].notna().all()
    events = pd.read_csv(events_path(out, METHOD, SPLIT, VIDEO))
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert events["action"].tolist() == ["reprompted"]
    timing = pd.read_csv(timing_path(out, METHOD, SPLIT, VIDEO))
    assert list(timing.columns) == list(ONLINE_TIMING_COLUMNS)
    assert timing["frames"].iloc[0] == NUM_FRAMES
    assert timing["restarts"].iloc[0] == 1
    assert timing["restart_s"].iloc[0] >= 0 and timing["compute_s"].iloc[0] > 0
    for index in range(NUM_FRAMES):
        label = read_label_map(mask_path(out, METHOD, SPLIT, VIDEO, f"{index:05d}"))
        box, shot = (START_BOX, 0) if index < 5 else (NEW_BOX, 5)
        assert np.array_equal(label == 1, expected_mask(box, shot, index))


def test_rerun_skips_complete_videos(tree: Any) -> None:
    cfg, prompts, out = tree
    batch(cfg, prompts)
    before = tree_hash(out)
    assert batch(cfg, prompts) == (0, 1)
    assert tree_hash(out) == before


class Killed(RuntimeError):
    pass


class DyingFake(StatefulFake):
    def propagate_in_video(self, *args: Any, **kwargs: Any) -> Any:
        for item in super().propagate_in_video(*args, **kwargs):
            if item[0] == 12:
                raise Killed("job killed")
            yield item


def test_killed_run_leaves_video_incomplete_and_rerun_redoes_it(tree: Any) -> None:
    cfg, prompts, out = tree
    with pytest.raises(Killed):
        batch(cfg, prompts, predictor=DyingFake())
    stems = [f"{i:05d}" for i in range(NUM_FRAMES)]
    assert mask_path(out, METHOD, SPLIT, VIDEO, stems[0]).is_file()
    assert not signals_path(out, METHOD, SPLIT, VIDEO).exists()
    assert not is_online_complete(out, METHOD, SPLIT, VIDEO, stems, [1])
    assert batch(cfg, prompts) == (1, 0)
    assert is_online_complete(out, METHOD, SPLIT, VIDEO, stems, [1])
    assert len(list(masks_dir(out, METHOD, SPLIT, VIDEO).glob("*.png"))) == NUM_FRAMES


def test_truncated_signal_file_counts_as_incomplete(tree: Any) -> None:
    cfg, prompts, out = tree
    batch(cfg, prompts)
    path = signals_path(out, METHOD, SPLIT, VIDEO)
    path.write_text(path.read_text().splitlines()[0] + "\n")
    assert batch(cfg, prompts) == (1, 0)


def test_cached_sam2_tree_is_byte_identical_after_a_run(tree: Any) -> None:
    cfg, prompts, out = tree
    cached = out / "sam2_tiny" / SPLIT / "masks" / VIDEO
    cached.mkdir(parents=True)
    for index in range(3):
        (cached / f"{index:05d}.png").write_bytes(bytes([index]) * 40)
    (out / "sam2_tiny" / SPLIT / "signals").mkdir()
    (out / "sam2_tiny" / SPLIT / "signals" / f"{VIDEO}.csv").write_text("x\n")
    before = tree_hash(out / "sam2_tiny")
    batch(cfg, prompts, flags={(1, 5): ("low_score",)})
    assert tree_hash(out / "sam2_tiny") == before


@pytest.mark.parametrize("method", ["sam2_tiny", "sam2_large_x", "baseline_csrt_grabcut", "sam2"])
def test_sam2_alone_and_baseline_methods_are_refused(tree: Any, method: str) -> None:
    cfg, prompts, out = tree
    with pytest.raises(ValueError, match="never writes"):
        check_online_method(method)
    with pytest.raises(ValueError, match="never writes"):
        batch(cfg, prompts, method=method)
    assert not (out / method).exists()


def test_fix_and_oracle_method_names_are_accepted() -> None:
    for name in ("fix_rules_csrt_tiny", "oracle_flags_csrt_tiny"):
        assert check_online_method(name) == name


def test_frames_are_read_as_bgr_arrays(tree: Any) -> None:
    cfg, prompts, out = tree
    seen: list[Any] = []

    class Spy(ScriptedBoxes):
        def start(self, obj_id: int, frame_index: int, frame: Any, mask: Any) -> None:
            seen.append(frame)
            super().start(obj_id, frame_index, frame, mask)

    from src.recovery.online_runner import frame_paths

    run_video(
        StatefulFake(),
        ScriptedDetector(),
        Spy(),
        BoxCheck(),
        SETTINGS,
        frame_paths(cfg.data_root / "DAVIS" / "JPEGImages" / "480p" / VIDEO)[0].parent,
        prompts.videos[VIDEO],
        out,
        METHOD,
        SPLIT,
        VIDEO,
    )
    assert seen[0].shape == (64, 64, 3) and seen[0].dtype == np.uint8


def test_two_prompt_frames_are_refused(tree: Any) -> None:
    from src.sam2_runner.prompts import Prompts

    cfg, _, out = tree
    prompts = Prompts(VIDEO, 0, {1: START_BOX, 2: NEW_BOX}, {2: 3})
    with pytest.raises(ValueError, match="one prompt frame"):
        run_video(
            StatefulFake(),
            ScriptedDetector(),
            ScriptedBoxes(),
            BoxCheck(),
            SETTINGS,
            cfg.data_root / "DAVIS" / "JPEGImages" / "480p" / VIDEO,
            prompts,
            out,
            METHOD,
            SPLIT,
            VIDEO,
        )
