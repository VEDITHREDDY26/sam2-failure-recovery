"""E2 table: SAM 2 alone vs SAM 2 + rules + recovery, Tiny / Large x DAVIS val / SA-V test.

    python -m experiments.e2.table [--results-dir results] [--out results/e2.csv]
        [--events-out results/events_e2.csv] [--models tiny large] [--splits davis_val sav_test]

Reads results/sam2_<model>_<split>.csv and results/fix_rules_csrt_<model>_<split>.csv (with the
_frames and events_ companions written by experiments.e2.score). J, F and J&F are means over
objects (as in E1). Per video J&F is the mean over the video's objects; a video is improved or
worse when its J&F changed by at least UNCHANGED_TOL, unchanged otherwise. Videos where the fix
made things worse are counted, never dropped. The two runs must score the same videos and objects.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.eval.rows import ScoreRow, read_scores_csv
from src.recovery.events import EVENT_COLUMNS

MODELS = ("tiny", "large")
TEST_SETS = ("davis_val", "sav_test")
UNCHANGED_TOL = 0.001
TABLE_COLUMNS = (
    "model",
    "test_set",
    "method",
    "videos",
    "objects",
    "J",
    "F",
    "J&F",
    "delta_JF",
    "videos_improved",
    "videos_worse",
    "videos_unchanged",
    "reprompts",
    "rejected",
    "tracker_lost",
    "frames_recovered",
    "frames_broken",
)
FRAME_COLUMNS = (
    "video",
    "object_id",
    "labelled_frames",
    "sam2_failures",
    "fix_failures",
    "frames_recovered",
    "frames_broken",
)
DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUT = "results/e2.csv"
DEFAULT_EVENTS_OUT = "results/events_e2.csv"


class E2Error(ValueError):
    """An E2 input is missing or inconsistent."""


def alone_method(model: str) -> str:
    """Method name of the SAM 2 alone run for a model size.

    Args:
        model: "tiny" or "large".

    Returns:
        The name sam2_<model>.
    """
    return f"sam2_{model}"


def fix_method(model: str) -> str:
    """Method name of the E2 run (rules + CSRT recovery) for a model size.

    Args:
        model: "tiny" or "large".

    Returns:
        The name fix_rules_csrt_<model>.
    """
    return f"fix_rules_csrt_{model}"


def _require(path: Path) -> Path:
    """Return the path if the file exists.

    Raises:
        E2Error: If the file is missing.
    """
    if not path.is_file():
        raise E2Error(f"missing input file: {path}")
    return path


def load_scores(path: Path) -> list[ScoreRow]:
    """Read the per-object scores of a run.

    Args:
        path: Scores CSV with one row per (video, object).

    Returns:
        The score rows.

    Raises:
        E2Error: If the file is missing, empty or has duplicate (video, object) rows.
    """
    rows = read_scores_csv(_require(path))
    if not rows:
        raise E2Error(f"{path}: no rows")
    keys = [(r.video, r.object) for r in rows]
    if len(set(keys)) != len(keys):
        raise E2Error(f"{path}: duplicate (video, object) rows")
    return rows


def per_video_jf(rows: Sequence[ScoreRow]) -> dict[str, float]:
    """Average J&F over the objects of each video.

    Args:
        rows: Per-object score rows.

    Returns:
        Mapping from video name to mean J&F.
    """
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.video, []).append(row.jf)
    return {video: sum(v) / len(v) for video, v in grouped.items()}


def count_changes(alone: dict[str, float], fix: dict[str, float]) -> tuple[int, int, int]:
    """(improved, worse, unchanged) over the videos both runs scored."""
    improved = worse = unchanged = 0
    for video, base in alone.items():
        delta = fix[video] - base
        if delta >= UNCHANGED_TOL:
            improved += 1
        elif delta <= -UNCHANGED_TOL:
            worse += 1
        else:
            unchanged += 1
    return improved, worse, unchanged


def means(rows: Sequence[ScoreRow]) -> dict[str, float]:
    """Average J, F and J&F over all objects (the same convention as E1).

    Args:
        rows: Per-object score rows; must not be empty.

    Returns:
        Mapping with the keys "J", "F" and "J&F".
    """
    n = len(rows)
    return {
        "J": sum(r.j for r in rows) / n,
        "F": sum(r.f for r in rows) / n,
        "J&F": sum(r.jf for r in rows) / n,
    }


def load_events(path: Path) -> pd.DataFrame:
    """Read the events file of a run and check its columns.

    Args:
        path: Events CSV.

    Returns:
        The events table.

    Raises:
        E2Error: If the file is missing or its columns are not EVENT_COLUMNS.
    """
    table = pd.read_csv(_require(path), dtype={"video": str})
    if list(table.columns) != list(EVENT_COLUMNS):
        raise E2Error(f"{path}: columns are {list(table.columns)}")
    return table


def load_frames(path: Path) -> pd.DataFrame:
    """Read the frames file of a run and check its columns.

    Args:
        path: Frames CSV written by the scoring step.

    Returns:
        The per-object frame counts table.

    Raises:
        E2Error: If the file is missing or its columns are not FRAME_COLUMNS.
    """
    table = pd.read_csv(_require(path), dtype={"video": str})
    if list(table.columns) != list(FRAME_COLUMNS):
        raise E2Error(f"{path}: columns are {list(table.columns)}")
    return table


def _same_objects(alone: Sequence[ScoreRow], fix: Sequence[ScoreRow], label: str) -> None:
    """Check that SAM 2 alone and the fix scored the same (video, object) pairs.

    Args:
        alone: Score rows of SAM 2 alone.
        fix: Score rows of the fixed run.
        label: Text naming the model and split, used in the error message.

    Raises:
        E2Error: If the two runs cover different videos or objects.
    """
    if {(r.video, r.object) for r in alone} != {(r.video, r.object) for r in fix}:
        raise E2Error(f"{label}: SAM 2 alone and the fix score different videos or objects")


def build_tables(
    results_dir: Path, models: Sequence[str] = MODELS, splits: Sequence[str] = TEST_SETS
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Build the E2 table rows and the merged events of all models and test sets.

    Args:
        results_dir: Folder with the result CSVs.
        models: Model sizes to include.
        splits: Test sets to include.

    Returns:
        The table rows (two per model and test set: SAM 2 alone, then the fix) and the events of
        all runs with "model" and "split" columns in front.

    Raises:
        E2Error: If an input is missing or inconsistent.
    """
    table: list[dict[str, object]] = []
    all_events: list[pd.DataFrame] = []
    # One pair of rows (SAM 2 alone, then the fix) per model and test set.
    for model in models:
        for split in splits:
            label = f"{model}/{split}"
            alone = load_scores(results_dir / f"{alone_method(model)}_{split}.csv")
            fix_name = f"{fix_method(model)}_{split}"
            fix = load_scores(results_dir / f"{fix_name}.csv")
            # Both runs must score the same objects and the frames file must cover them.
            _same_objects(alone, fix, label)
            events = load_events(results_dir / f"events_{fix_name}.csv")
            frames = load_frames(results_dir / f"{fix_name}_frames.csv")
            if {(v, o) for v, o in zip(frames["video"], frames["object_id"], strict=True)} != {
                (r.video, r.object) for r in fix
            }:
                raise E2Error(f"{label}: the frames file covers different objects than the scores")
            # Per-video J&F decides improved / worse / unchanged; worse videos are kept.
            alone_jf, fix_jf = per_video_jf(alone), per_video_jf(fix)
            improved, worse, unchanged = count_changes(alone_jf, fix_jf)
            base, new = means(alone), means(fix)
            common = {"model": model, "test_set": split, "videos": len(alone_jf)}
            common["objects"] = len(alone)
            # SAM 2 alone row: no change, and no recovery columns.
            table.append(
                {**common, "method": alone_method(model), **base}
                | {
                    "delta_JF": 0.0,
                    "videos_improved": "",
                    "videos_worse": "",
                    "videos_unchanged": "",
                }
                | {k: "" for k in TABLE_COLUMNS[12:]}
            )
            # Fix row: change in J&F plus event and frame counts.
            actions = events["action"]
            table.append(
                {**common, "method": fix_method(model), **new}
                | {
                    "delta_JF": new["J&F"] - base["J&F"],
                    "videos_improved": improved,
                    "videos_worse": worse,
                    "videos_unchanged": unchanged,
                    "reprompts": int((actions == "reprompted").sum()),
                    "rejected": int((actions == "rejected").sum()),
                    "tracker_lost": int((actions == "tracker_lost").sum()),
                    "frames_recovered": int(frames["frames_recovered"].sum()),
                    "frames_broken": int(frames["frames_broken"].sum()),
                }
            )
            all_events.append(events.assign(model=model, split=split))
    # Tag each run's events with model and split before merging.
    merged = pd.concat(all_events, ignore_index=True)
    return table, merged[["model", "split", *EVENT_COLUMNS]]


def write_table(path: Path, table: Sequence[dict[str, object]]) -> None:
    """Write the table as CSV through a temporary file.

    Args:
        path: Output file; its folder is created if needed.
        table: Rows to write; floats are written with repr so no precision is lost.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TABLE_COLUMNS)
        for row in table:
            writer.writerow(
                [repr(v) if isinstance(v, float) else v for v in (row[c] for c in TABLE_COLUMNS)]
            )
    os.replace(temp, path)


def main(argv: Sequence[str] | None = None) -> None:
    """Build the E2 table and the merged events file (command line).

    Reads the score, events and frames CSVs in --results-dir. Writes --out and --events-out.

    Args:
        argv: Command-line arguments; None means sys.argv.

    Raises:
        E2Error: If an input is missing or inconsistent.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--events-out", default=DEFAULT_EVENTS_OUT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--splits", nargs="+", choices=TEST_SETS, default=list(TEST_SETS))
    args = parser.parse_args(argv)
    table, events = build_tables(Path(args.results_dir), args.models, args.splits)
    write_table(Path(args.out), table)
    Path(args.events_out).parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.events_out, index=False)


if __name__ == "__main__":
    main()
