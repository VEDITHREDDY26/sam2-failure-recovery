from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from experiments.e10_types import count, sample
from src.common.layout import mask_path
from src.common.masks import write_label_map
from src.data.video import VideoSample
from src.recovery.events import EVENT_COLUMNS

H, W, N = 48, 64, 40


class FakeGt:
    def __init__(self, frames: tuple[int, ...]) -> None:
        self.frames = frames

    def annotated_frames(self, obj_id: int) -> tuple[int, ...]:
        return self.frames

    def read_mask(self, obj_id: int, frame_index: int) -> np.ndarray:
        mask = np.zeros((H, W), dtype=bool)
        mask[10:30, 10:30] = True
        return mask


def make_video(root: Path, split: str, name: str, annotated: tuple[int, ...]) -> VideoSample:
    frames = root / "frames" / split / name
    frames.mkdir(parents=True, exist_ok=True)
    for i in range(N):
        Image.fromarray(np.full((H, W, 3), i * 5, dtype=np.uint8)).save(frames / f"{i:05d}.jpg")
    stems = tuple(f"{i:05d}" for i in range(N))
    return VideoSample("x", split, name, frames, stems, (1,), {1: "o"}, FakeGt(annotated))


def write_alone(out: Path, split: str, name: str, bad_from: int) -> None:
    for i in range(N):
        label = np.zeros((H, W), dtype=np.uint8)
        if i < bad_from:
            label[10:30, 10:30] = 1
        else:
            label[40:46, 50:60] = 1
        write_label_map(mask_path(out, "sam2_tiny", split, name, f"{i:05d}"), label)


def event(video: str, frame: int, action: str) -> tuple[object, ...]:
    return (video, 1, frame, "score", action, "", 5, 5, 20, 20, 0)


def write_events(results: Path, split: str, rows: list[tuple[object, ...]]) -> None:
    results.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    table.to_csv(results / f"events_fix_rules_csrt_tiny_{split}.csv", index=False)


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, Path, sample.LoadVideo]:
    out, results = tmp_path / "out", tmp_path / "results"
    videos: dict[tuple[str, str], VideoSample] = {}
    specs = {
        ("davis_val", "a"): tuple(range(N)),
        ("sav_test", "b"): (0, 10, 20, 30),
    }
    for (split, name), annotated in specs.items():
        videos[(split, name)] = make_video(tmp_path, split, name, annotated)
        write_alone(out, split, name, bad_from=15)
    write_events(
        results,
        "davis_val",
        [
            event("a", 5, "reprompted"),  # alone still right -> not a failure
            event("a", 16, "reprompted"),
            event("a", 16, "rejected"),  # same frame: one candidate
            event("a", 20, "cooldown"),  # excluded
            event("a", 25, "tracker_lost"),
        ],
    )
    write_events(
        results,
        "sav_test",
        [event("b", 12, "reprompted"), event("b", 35, "rejected")],  # 12 -> 20; 35 -> none
    )
    return out, results, lambda split, name: videos[(split, name)]


def test_candidates_rules(world: tuple[Path, Path, sample.LoadVideo]) -> None:
    out, results, load = world
    cand = sample.build_candidates(results, out, load)
    keys = list(
        zip(cand["split"], cand["video"], cand["frame_index"], cand["gt_frame_index"], strict=True)
    )
    assert keys == [
        ("davis_val", "a", 16, 16),
        ("davis_val", "a", 25, 25),
        ("sav_test", "b", 12, 20),
    ]
    assert (cand["iou"] < 0.5).all()


def test_sample_seeded_and_small_case(world: tuple[Path, Path, sample.LoadVideo]) -> None:
    out, results, load = world
    cand = sample.build_candidates(results, out, load)
    assert len(sample.draw_sample(cand)) == 3
    big = pd.concat([cand.assign(object_id=i) for i in range(1, 30)], ignore_index=True)
    first, second = sample.draw_sample(big), sample.draw_sample(big.sample(frac=1, random_state=1))
    assert len(first) == 40
    assert first.equals(second)
    assert first["sample_id"].tolist()[:2] == ["s01", "s02"]


def test_build_sample_writes_files_and_refuses_repo(
    world: tuple[Path, Path, sample.LoadVideo], tmp_path: Path
) -> None:
    out, results, load = world
    clips, csv_out = tmp_path / "clips", tmp_path / "sample.csv"
    made = sample.build_sample(results, out, load, clips, csv_out)
    assert len(made) == 3
    assert (clips / "s01.png").is_file() and (clips / "labels_template.csv").is_file()
    assert list(pd.read_csv(csv_out).columns) == list(sample.SAMPLE_COLUMNS)
    template = pd.read_csv(clips / "labels_template.csv")
    assert list(template.columns) == ["sample_id", "type", "notes"]
    with pytest.raises(SystemExit):
        sample.build_sample(results, out, load, sample.REPO_ROOT / "figures" / "x", csv_out)


def samples() -> pd.DataFrame:
    return pd.DataFrame(
        {"sample_id": ["s01", "s02", "s03", "s04"], "split": ["davis_val"] * 2 + ["sav_test"] * 2}
    )


def labels(types: list[str], ids: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"sample_id": ids or ["s01", "s02", "s03", "s04"], "type": types})


def test_count_validation_failures() -> None:
    ok = ["occlusion", "other", "other", "distractor"]
    count.validate(labels(ok), samples())
    for bad in (
        labels(["occlusion", "", "other", "other"]),
        labels(["occlusion", "blur", "other", "other"]),
        labels(ok[:3], ["s01", "s02", "s03"]),
        labels(ok, ["s01", "s01", "s03", "s04"]),
        labels(ok, ["s01", "s02", "s03", "s99"]),
    ):
        with pytest.raises(count.LabelError):
            count.validate(bad, samples())


def test_counts_and_shares() -> None:
    rows = count.count_types(labels(["occlusion", "other", "other", "distractor"]), samples())
    by_type = {r[0]: r[1:] for r in rows}
    assert by_type["other"] == [2, 0.5, 1, 1]
    assert by_type["occlusion"] == [1, 0.25, 1, 0]
    assert by_type["leaves_frame"] == [0, 0.0, 0, 0]
    assert sum(r[1] for r in rows) == 4


def test_count_main_writes_csv(tmp_path: Path) -> None:
    samples().to_csv(tmp_path / "s.csv", index=False)
    labels(["occlusion"] * 4).to_csv(tmp_path / "l.csv", index=False)
    count.main(
        [
            "--labels",
            str(tmp_path / "l.csv"),
            "--sample",
            str(tmp_path / "s.csv"),
            "--out",
            str(tmp_path / "o.csv"),
        ]
    )
    table = pd.read_csv(tmp_path / "o.csv")
    assert list(table.columns) == list(count.OUT_COLUMNS)
    assert table.loc[table["type"] == "occlusion", "count"].item() == 4


def count_colour(image: np.ndarray, colour: tuple[int, int, int]) -> int:
    return int((image == np.array(colour, dtype=np.uint8)).all(axis=-1).sum())


def strip_row(frame: int, with_box: bool = True) -> pd.Series:
    values: dict[str, object] = {
        "sample_id": "s01",
        "split": "sav_test",
        "video": "b",
        "object_id": 1,
        "frame_index": frame,
    }
    if with_box:
        values.update(box_x0=5, box_y0=5, box_x1=20, box_y1=20)
    return pd.Series(values)


def test_thumb_gt_outline_only_on_annotated_frame(
    world: tuple[Path, Path, sample.LoadVideo],
) -> None:
    out, _, load = world
    video = load("sav_test", "b")
    on = sample._thumb(video, out, strip_row(20), 20)
    off = sample._thumb(video, out, strip_row(20), 21)
    assert on.shape[1] == sample.THUMB_WIDTH
    assert count_colour(on, sample.YELLOW) > 0
    assert count_colour(off, sample.YELLOW) == 0
    assert count_colour(on, sample.GREEN) > 0


def test_thumb_empty_alone_and_gt_not_visible(tmp_path: Path) -> None:
    video = make_video(tmp_path, "sav_test", "b", (0, 10, 20, 30))
    out = tmp_path / "out"
    for i in range(N):
        write_label_map(
            mask_path(out, "sam2_tiny", "sav_test", "b", f"{i:05d}"), np.zeros((H, W), np.uint8)
        )
    thumb = sample._thumb(video, out, strip_row(20), 20)
    assert count_colour(thumb, sample.GREEN) == 0
    assert thumb.shape[0] > 0

    class EmptyGt(FakeGt):
        def read_mask(self, obj_id: int, frame_index: int) -> np.ndarray:
            return np.zeros((H, W), dtype=bool)

    video = replace(video, reader=EmptyGt((20,)))
    assert count_colour(sample._thumb(video, out, strip_row(20), 20), sample.YELLOW) == 0


def test_box_orange_on_flagged_frame_only(world: tuple[Path, Path, sample.LoadVideo]) -> None:
    out, _, load = world
    video = load("sav_test", "b")
    row = strip_row(20)
    assert count_colour(sample._thumb(video, out, row, 20), sample.ORANGE) > 0
    assert count_colour(sample._thumb(video, out, row, 18), sample.ORANGE) == 0
    assert count_colour(sample._thumb(video, out, strip_row(20, False), 20), sample.ORANGE) == 0


def test_strip_width(world: tuple[Path, Path, sample.LoadVideo], tmp_path: Path) -> None:
    out, _, load = world
    path = sample.render_strip(strip_row(20), load("sav_test", "b"), out, tmp_path / "s.png")
    assert np.asarray(Image.open(path)).shape[1] == 1600


def test_csv_outputs_unchanged(world: tuple[Path, Path, sample.LoadVideo], tmp_path: Path) -> None:
    out, results, load = world
    clips, csv_out = tmp_path / "clips", tmp_path / "sample.csv"
    made = sample.build_sample(results, out, load, clips, csv_out)
    old = tmp_path / "old.csv"
    drawn = sample.draw_sample(sample.build_candidates(results, out, load))
    sample.write_csv(old, sample.SAMPLE_COLUMNS, drawn.itertuples(index=False))  # type: ignore[arg-type]
    assert csv_out.read_bytes() == old.read_bytes()
    expected = "sample_id,type,notes\n" + "".join(f"{s},,\n" for s in made["sample_id"])
    assert (clips / "labels_template.csv").read_bytes() == expected.encode()
