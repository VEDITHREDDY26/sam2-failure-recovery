"""E11 table: SAM 2 alone vs the E2 method (accept every re-prompt) vs verified recovery.

    python -m experiments.verify.table [--results-dir results] [--out results/e11.csv]
        [--events-out results/events_e11.csv] [--models tiny large] [--splits davis_val sav_test]

Reads the scored files of sam2_<model>, fix_rules_csrt_<model> and fix_verified_rules_csrt_<model>
(with the _frames and events_ companions written by experiments.e2.score). Same columns and
counting rules as the E2 table, plus verify_rejected (candidates the rule rejected) and
delta_JF_vs_e2 (verified minus E2 method J&F). reprompts counts the re-prompts that were kept.
Videos where a method made things worse are counted, never dropped.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.e2.table import (
    TEST_SETS,
    E2Error,
    count_changes,
    load_events,
    load_frames,
    load_scores,
    means,
    per_video_jf,
)
from experiments.e2.table import _same_objects as same_objects
from src.eval.rows import ScoreRow
from src.recovery.events import EVENT_COLUMNS

MODELS = ("tiny", "large")
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
    "delta_JF_vs_e2",
    "videos_improved",
    "videos_worse",
    "videos_unchanged",
    "reprompts",
    "verify_rejected",
    "rejected",
    "tracker_lost",
    "frames_recovered",
    "frames_broken",
)
DEFAULT_OUT = "results/e11.csv"
DEFAULT_EVENTS_OUT = "results/events_e11.csv"


def alone_method(model: str) -> str:
    return f"sam2_{model}"


def e2_method(model: str) -> str:
    return f"fix_rules_csrt_{model}"


def verified_method(model: str) -> str:
    return f"fix_verified_rules_csrt_{model}"


def _blank() -> dict[str, object]:
    return {c: "" for c in TABLE_COLUMNS}


def _fix_row(
    common: dict[str, object],
    method: str,
    rows: Sequence[ScoreRow],
    alone: Sequence[ScoreRow],
    events: pd.DataFrame,
    frames: pd.DataFrame,
) -> dict[str, object]:
    base, new = means(alone), means(rows)
    improved, worse, unchanged = count_changes(per_video_jf(alone), per_video_jf(rows))
    actions = events["action"]
    return (
        _blank()
        | common
        | {"method": method, **new}
        | {
            "delta_JF": new["J&F"] - base["J&F"],
            "videos_improved": improved,
            "videos_worse": worse,
            "videos_unchanged": unchanged,
            "reprompts": int((actions == "reprompted").sum()),
            "verify_rejected": int((actions == "verify_rejected").sum()),
            "rejected": int((actions == "rejected").sum()),
            "tracker_lost": int((actions == "tracker_lost").sum()),
            "frames_recovered": int(frames["frames_recovered"].sum()),
            "frames_broken": int(frames["frames_broken"].sum()),
        }
    )


def build_tables(
    results_dir: Path, models: Sequence[str] = MODELS, splits: Sequence[str] = TEST_SETS
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    table: list[dict[str, object]] = []
    all_events: list[pd.DataFrame] = []
    for model in models:
        for split in splits:
            alone = load_scores(results_dir / f"{alone_method(model)}_{split}.csv")
            common: dict[str, object] = {
                "model": model,
                "test_set": split,
                "videos": len(per_video_jf(alone)),
                "objects": len(alone),
            }
            table.append(_blank() | common | {"method": alone_method(model), **means(alone)})
            table[-1]["delta_JF"] = 0.0
            scored: dict[str, float] = {}
            for method in (e2_method(model), verified_method(model)):
                name = f"{method}_{split}"
                rows = load_scores(results_dir / f"{name}.csv")
                same_objects(alone, rows, f"{model}/{split}/{method}")
                events = load_events(results_dir / f"events_{name}.csv")
                frames = load_frames(results_dir / f"{name}_frames.csv")
                keys = set(zip(frames["video"], frames["object_id"], strict=True))
                if keys != {(r.video, r.object) for r in rows}:
                    raise E2Error(f"{model}/{split}/{method}: frames file covers other objects")
                table.append(_fix_row(common, method, rows, alone, events, frames))
                scored[method] = means(rows)["J&F"]
                all_events.append(events.assign(model=model, split=split, method=method))
            table[-1]["delta_JF_vs_e2"] = scored[verified_method(model)] - scored[e2_method(model)]
    merged = pd.concat(all_events, ignore_index=True)
    return table, merged[["model", "split", "method", *EVENT_COLUMNS]]


def write_table(path: Path, table: Sequence[dict[str, object]]) -> None:
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", default="results")
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
