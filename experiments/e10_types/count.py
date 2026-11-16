"""E10 failure types, step 2: validate the filled labelling CSV and count the types.

    python -m experiments.e10_types.count --labels <filled labels CSV> \
        [--sample results/e10_types_sample.csv] [--out results/e10_types.csv]

The labels file has sample_id, type, notes. Every sample_id of the sample must appear exactly once
with an allowed type; blanks, unknown ids and unknown types are refused. Writes one row per type
(all allowed types, zero counts included) with count, share and per-test-set counts.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.e10_types.sample import SPLITS, TYPES, write_csv

OUT_COLUMNS = ("type", "count", "share", *(f"count_{s}" for s in SPLITS))


class LabelError(ValueError):
    """The labelling file is incomplete or invalid."""


def validate(labels: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    if not {"sample_id", "type"} <= set(labels.columns):
        raise LabelError(f"labels need columns sample_id and type, got {list(labels.columns)}")
    ids = labels["sample_id"].astype(str).str.strip()
    types = labels["type"].fillna("").astype(str).str.strip()
    if (types == "").any():
        raise LabelError(f"blank type for {sorted(ids[types == ''])}")
    if ids.duplicated().any():
        raise LabelError(f"duplicate sample_id {sorted(ids[ids.duplicated()])}")
    expected = set(sample["sample_id"])
    if set(ids) - expected:
        raise LabelError(f"unknown sample_id {sorted(set(ids) - expected)}")
    if expected - set(ids):
        raise LabelError(f"missing sample_id {sorted(expected - set(ids))}")
    bad = sorted(set(types) - set(TYPES))
    if bad:
        raise LabelError(f"type not allowed {bad}; allowed {list(TYPES)}")
    return pd.DataFrame({"sample_id": ids, "type": types})


def count_types(labels: pd.DataFrame, sample: pd.DataFrame) -> list[list[object]]:
    joined = labels.merge(sample[["sample_id", "split"]], on="sample_id")
    total = len(joined)
    rows: list[list[object]] = []
    for kind in TYPES:
        of_kind = joined[joined["type"] == kind]
        per_split = [int((of_kind["split"] == s).sum()) for s in SPLITS]
        rows.append([kind, len(of_kind), len(of_kind) / total if total else 0.0, *per_split])
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=Path("results/e10_types_sample.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/e10_types.csv"))
    args = parser.parse_args(argv)
    sample = pd.read_csv(args.sample, dtype={"video": str})
    labels = validate(pd.read_csv(args.labels, dtype=str, keep_default_na=False), sample)
    rows = count_types(labels, sample)
    write_csv(args.out, OUT_COLUMNS, rows)
    print(f"{len(labels)} labels -> {args.out}")


if __name__ == "__main__":
    main()
