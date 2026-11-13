from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from experiments.e2.table import UNCHANGED_TOL
from experiments.e10_overlays import overlays, select
from scripts import make_demo
from src.common.layout import mask_path
from src.common.masks import write_label_map
from src.data.video import VideoSample
from src.eval.rows import ScoreRow, write_scores_csv
from src.recovery.events import EVENT_COLUMNS

H, W, N = 48, 64, 12
SPLIT = "davis_val"


class FakeGt:
    def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
        return tuple(range(0, N, 2))

    def read_mask(self, obj_id: int, frame_index: int) -> np.ndarray:
        mask = np.zeros((H, W), dtype=bool)
        mask[10:30, 10:30] = True
        return mask


def make_video(root: Path, name: str) -> VideoSample:
    frames = root / "frames" / name
    frames.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(N):
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(
            frames / f"{i:05d}.jpg"
        )
    return VideoSample(
        "x", SPLIT, name, frames, tuple(f"{i:05d}" for i in range(N)), (1,), {1: "o"}, FakeGt()
    )


def write_masks(out: Path, method: str, name: str) -> None:
    for i in range(N):
        label = np.zeros((H, W), dtype=np.uint8)
        label[10 + i // 2 : 30, 12:34] = 1
        write_label_map(mask_path(out, method, SPLIT, name, f"{i:05d}"), label)


def test_rule_picks_improved_worse_and_median_with_ties_by_name() -> None:
    deltas = {
        "a": 0.30,
        "b": 0.30,
        "c": 0.20,
        "d": 0.01,
        "e": 0.0,
        "f": -0.05,
        "g": -0.05,
        "h": -0.2,
        "i": 0.0005,
    }
    picks = {(p.role, p.rank): p.video for p in select.select_videos(deltas)}
    assert picks[("improved", 1)] == "a" and picks[("improved", 2)] == "b"
    assert picks[("worse", 1)] == "h" and picks[("worse", 2)] == "f"
    assert picks[("median", 1)] in {"d", "e", "i"}
    assert len(picks) == 5


def test_rule_never_pads_and_skips_noise_level_changes() -> None:
    picks = select.select_videos({"a": 0.5, "b": 0.0004, "c": -0.0004})
    assert [(p.role, p.video) for p in picks] == [("improved", "a"), ("median", "b")]
    assert UNCHANGED_TOL == 0.001


def write_results(
    results: Path, scores: dict[str, tuple[float, float]], events: list[tuple]
) -> None:
    for method, idx in (("sam2_tiny", 0), ("fix_rules_csrt_tiny", 1)):
        rows = [ScoreRow(v, 1, s[idx], s[idx]) for v, s in scores.items()]
        write_scores_csv(results / f"{method}_{SPLIT}.csv", rows)
    pd.DataFrame(events, columns=list(EVENT_COLUMNS)).to_csv(
        results / f"events_fix_rules_csrt_tiny_{SPLIT}.csv", index=False
    )
    pd.DataFrame({"video": list(scores), "frames": [N] * len(scores)}).to_csv(
        results / f"fix_rules_csrt_tiny_timing_{SPLIT}.csv", index=False
    )


def event(video: str, frame: int, action: str) -> tuple:
    return (video, 1, frame, "low_score", action, "", "", "", "", "", 0)


def test_selection_csv_traces_each_clip_and_picks_the_frame(tmp_path: Path) -> None:
    scores = {"v1": (0.5, 0.9), "v2": (0.5, 0.2), "v3": (0.6, 0.6), "v4": (0.7, 0.75)}
    write_results(
        tmp_path,
        scores,
        [event("v1", 7, "rejected"), event("v1", 6, "reprompted"), event("v2", 9, "reprompted")],
    )
    table = select.build_selection(tmp_path, [SPLIT], "sam2_tiny", "fix_rules_csrt_tiny")
    assert list(table.columns) == list(select.SELECTION_COLUMNS)
    by = table.set_index("video")
    assert by.loc["v1", "role"] == "improved" and by.loc["v1", "frame_index"] == 6
    assert by.loc["v2", "role"] == "worse" and by.loc["v2", "frame_source"] == "first_reprompt"
    assert (by["frame_source"] == "middle").any() and by.loc["v3", "frame_index"] == N // 2
    assert (by["fix_file"] == f"fix_rules_csrt_tiny_{SPLIT}.csv").all()
    assert not by["oracle"].any()


def test_selection_refuses_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing input file"):
        select.build_selection(tmp_path, [SPLIT], "sam2_tiny", "fix_rules_csrt_tiny")


def selection_row(fix: str = "fix_rules_csrt_tiny") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": SPLIT,
                "role": "worse",
                "rank": 1,
                "video": "vid",
                "alone_method": "sam2_tiny",
                "fix_method": fix,
                "oracle": "oracle" in fix,
                "jf_alone": 0.8,
                "jf_fix": 0.4,
                "delta_jf": -0.4,
                "frame_index": 4,
                "frame_source": "first_reprompt",
                "alone_file": f"sam2_tiny_{SPLIT}.csv",
                "fix_file": f"{fix}_{SPLIT}.csv",
                "events_file": "e.csv",
            }
        ]
    )


def setup_clip(tmp_path: Path, fix: str) -> tuple[Path, overlays.LoadVideo]:
    out = tmp_path / "out"
    video = make_video(tmp_path, "vid")
    write_masks(out, "sam2_tiny", "vid")
    write_masks(out, fix, "vid")
    return out, lambda split, name: video


def test_overlay_figure_is_written_with_scores_in_the_title(tmp_path: Path) -> None:
    out, load = setup_clip(tmp_path, "fix_rules_csrt_tiny")
    paths = overlays.build_overlays(selection_row(), out, load, tmp_path / "figures")
    assert [p.name for p in paths] == [f"e10_{SPLIT}_vid.png"]
    assert paths[0].stat().st_size > 1000
    title = overlays.figure_title(selection_row().iloc[0])
    assert "0.800" in title and "0.400" in title and "ORACLE" not in title


def test_oracle_methods_are_labelled_oracle(tmp_path: Path) -> None:
    fix = "fix_oracle_flags_csrt_tiny"
    out, load = setup_clip(tmp_path, fix)
    row = selection_row(fix)
    paths = overlays.build_overlays(row, out, load, tmp_path / "figures")
    assert paths[0].name == f"e10_oracle_{SPLIT}_vid.png"
    assert overlays.figure_title(row.iloc[0]).startswith("ORACLE")
    assert "ORACLE" in " ".join(make_demo.clip_label(row.iloc[0]))


def test_demo_refuses_a_path_inside_the_repo(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="inside the repository"):
        make_demo.check_output_path(make_demo.REPO_ROOT / "figures" / "demo.mp4")
    with pytest.raises(SystemExit, match="mp4"):
        make_demo.check_output_path(tmp_path / "demo.avi")
    with pytest.raises(SystemExit, match="inside the repository"):
        make_demo.main(["--out", str(make_demo.REPO_ROOT / "demo.mp4")])


def test_demo_writes_a_readable_mp4(tmp_path: Path) -> None:
    out, load = setup_clip(tmp_path, "fix_rules_csrt_tiny")
    target = tmp_path / "drive" / "demo.mp4"
    count = make_demo.make_demo(selection_row(), out, load, target, before=2, after=3, fps=5)
    assert count == make_demo.CARD_SECONDS * 5 + 6
    cap = cv2.VideoCapture(str(target))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == count
    cap.release()
