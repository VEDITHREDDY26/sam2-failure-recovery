"""E12 part 5: failure type against re-prompt outcome (analysis, needs the filled type labels).

    python -m experiments.e12.types_outcome --labels <filled labels CSV> \
        [--results-dir results] [--out results/e12_types_outcome.csv]

Joins the sampled failure candidates (results/e10_types_sample.csv) and their labelled types with
the re-prompts of the rules + CSRT fix (Tiny) on split, video, object_id and frame_index. Per type:
sampled candidates, how many were re-prompted, and how many of those were wrong afterwards
(IoU after < 0.5). Missing or invalid labels are refused.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.e10_types.count import validate
from experiments.e10_types.sample import SPLITS, TYPES, write_csv
from experiments.e12.candidates import CandidateError

KEY = ["split", "video", "object_id", "frame_index"]
WRONG_BELOW = 0.5
OUT_COLUMNS = ("type", "sampled", "reprompted", "wrong")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUT = Path("results/e12_types_outcome.csv")


def read_reprompts(results_dir: Path) -> pd.DataFrame:
    parts = []
    for split in SPLITS:
        path = results_dir / f"fix_rules_csrt_tiny_{split}_reprompts.csv"
        if not path.is_file():
            raise CandidateError(f"missing input file: {path}")
        parts.append(pd.read_csv(path, dtype={"video": str}).assign(split=split))
    return pd.concat(parts, ignore_index=True)[[*KEY, "iou_after"]]


def join_outcomes(
    sample: pd.DataFrame, labels: pd.DataFrame, reprompts: pd.DataFrame
) -> pd.DataFrame:
    labelled = sample.merge(labels, on="sample_id", validate="one_to_one")
    return labelled.merge(reprompts, on=KEY, how="left", validate="many_to_one")


def summarise(joined: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    for kind in TYPES:
        of_kind = joined[joined["type"] == kind]
        reprompted = of_kind["iou_after"].notna()
        wrong = of_kind["iou_after"] < WRONG_BELOW
        rows.append([kind, len(of_kind), int(reprompted.sum()), int(wrong.sum())])
    return rows


def build(labels_path: Path, results_dir: Path) -> list[list[object]]:
    sample_path = results_dir / "e10_types_sample.csv"
    for path in (labels_path, sample_path):
        if not path.is_file():
            raise CandidateError(f"missing input file: {path}")
    sample = pd.read_csv(sample_path, dtype={"video": str})
    labels = validate(pd.read_csv(labels_path, dtype=str, keep_default_na=False), sample)
    return summarise(join_outcomes(sample, labels, read_reprompts(results_dir)))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    rows = build(args.labels, args.results_dir)
    write_csv(args.out, OUT_COLUMNS, rows)
    print(f"{len(rows)} rows -> {args.out} (analysis)")


if __name__ == "__main__":
    main()
