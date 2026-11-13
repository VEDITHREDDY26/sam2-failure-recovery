"""E10 clip selection: a written rule, applied to the E2 per-video results (no choice by eye).

    python -m experiments.e10_overlays.select [--results-dir results] [--splits davis_val sav_test]
        [--alone sam2_tiny] [--fix fix_rules_csrt_tiny] [--out results/e10_selection.csv]

Rule (fixed before any result existed): per test set, from per-video J&F deltas (fix - SAM 2 alone): the two
most improved (delta >= 0.001), the two most worsened (delta <= -0.001; fewer if fewer exist) and
the video nearest the median delta among the others; ties by video name. The frame is the first
"reprompted" event of the fix run, the middle frame of the video if there is none. Every selected
row names the results files and the video it comes from.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.e2.table import UNCHANGED_TOL, load_scores, per_video_jf
from src.common.layout import is_oracle_method, validate_method

SPLITS = ("davis_val", "sav_test")
PER_ROLE = 2
DEFAULT_OUT = "results/e10_selection.csv"
SELECTION_COLUMNS = (
    "split",
    "role",
    "rank",
    "video",
    "alone_method",
    "fix_method",
    "oracle",
    "jf_alone",
    "jf_fix",
    "delta_jf",
    "frame_index",
    "frame_source",
    "alone_file",
    "fix_file",
    "events_file",
)


class SelectionError(ValueError):
    """An input of the selection is missing or inconsistent."""


@dataclass(frozen=True)
class Pick:
    role: str
    rank: int
    video: str
    delta: float


def select_videos(deltas: dict[str, float], per_role: int = PER_ROLE) -> list[Pick]:
    """The written rule on a video -> delta table."""
    if not deltas:
        raise SelectionError("no videos to select from")
    by_gain = sorted(deltas, key=lambda v: (-deltas[v], v))
    improved = [v for v in by_gain if deltas[v] >= UNCHANGED_TOL][:per_role]
    worse = sorted((v for v in deltas if deltas[v] <= -UNCHANGED_TOL), key=lambda v: (deltas[v], v))
    worse = worse[:per_role]
    picks = [Pick("improved", i + 1, v, deltas[v]) for i, v in enumerate(improved)]
    picks += [Pick("worse", i + 1, v, deltas[v]) for i, v in enumerate(worse)]
    taken = {p.video for p in picks}
    rest = {v: d for v, d in deltas.items() if v not in taken}
    if rest:
        median = statistics.median(deltas.values())
        nearest = min(rest, key=lambda v: (abs(rest[v] - median), v))
        picks.append(Pick("median", 1, nearest, deltas[nearest]))
    return picks


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SelectionError(f"missing input file: {path}")
    return pd.read_csv(path, dtype={"video": str})


def clip_frame(video: str, events: pd.DataFrame, timing: pd.DataFrame) -> tuple[int, str]:
    """First re-prompt frame of the video, else its middle frame."""
    own = events[(events["video"] == video) & (events["action"] == "reprompted")]
    if len(own):
        return int(own["frame_index"].min()), "first_reprompt"
    frames = timing.loc[timing["video"] == video, "frames"]
    if len(frames) != 1:
        raise SelectionError(f"no frame count for '{video}' in the timing file")
    return int(frames.iloc[0]) // 2, "middle"


def selection_for_split(
    results_dir: Path, split: str, alone: str, fix: str
) -> list[dict[str, object]]:
    alone_file, fix_file = f"{alone}_{split}.csv", f"{fix}_{split}.csv"
    events_file, timing_file = f"events_{fix}_{split}.csv", f"{fix}_timing_{split}.csv"
    jf_alone = per_video_jf(load_scores(results_dir / alone_file))
    jf_fix = per_video_jf(load_scores(results_dir / fix_file))
    if set(jf_alone) != set(jf_fix):
        raise SelectionError(f"{split}: SAM 2 alone and the fix scored different videos")
    events, timing = _read(results_dir / events_file), _read(results_dir / timing_file)
    deltas = {v: jf_fix[v] - jf_alone[v] for v in jf_alone}
    rows: list[dict[str, object]] = []
    for pick in select_videos(deltas):
        frame, source = clip_frame(pick.video, events, timing)
        rows.append(
            {
                "split": split,
                "role": pick.role,
                "rank": pick.rank,
                "video": pick.video,
                "alone_method": alone,
                "fix_method": fix,
                "oracle": is_oracle_method(fix),
                "jf_alone": jf_alone[pick.video],
                "jf_fix": jf_fix[pick.video],
                "delta_jf": pick.delta,
                "frame_index": frame,
                "frame_source": source,
                "alone_file": alone_file,
                "fix_file": fix_file,
                "events_file": events_file,
            }
        )
    return rows


def build_selection(results_dir: Path, splits: Sequence[str], alone: str, fix: str) -> pd.DataFrame:
    validate_method(alone)
    validate_method(fix)
    rows = [r for s in splits for r in selection_for_split(results_dir, s, alone, fix)]
    return pd.DataFrame(rows, columns=list(SELECTION_COLUMNS))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--alone", default="sam2_tiny")
    parser.add_argument("--fix", default="fix_rules_csrt_tiny")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    args = parser.parse_args(argv)
    table = build_selection(args.results_dir, args.splits, args.alone, args.fix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"{len(table)} clips -> {args.out}")


if __name__ == "__main__":
    main()
