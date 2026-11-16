"""Check that all four E7 arms started from the same prompts and scored the same videos.

    python -m experiments.e7.check_arms [--results-dir results] [--prompts-dir DIR]
        [--out results/e7_arms_check.csv]

Per test set: every arm (and SAM 2 alone) is run from one prompts file, whose SHA-256 is recorded;
every arm's score file must cover exactly the same (video, object) pairs, and those must be the
objects the prompts file gives a box. Any mismatch raises, so the table is never built on arms
that are not comparable. Pass --arm-prompts ARM=PATH to point one arm at another file (used to
prove the check fails when prompts differ).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments.e7.arms import ALONE, ARMS, TEST_SETS
from src.common.config import DATA_ROOT_ENV
from src.eval.rows import read_scores_csv
from src.sam2_runner.prompts import load_batch_prompts
from src.sam2_runner.smoke import load_data_config

CHECK_COLUMNS = ("test_set", "arm", "prompts_sha256", "videos", "objects")


class ArmsMismatch(ValueError):
    """The arms did not start from the same prompts or cover the same videos and objects."""


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scored_objects(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        raise ArmsMismatch(f"missing score file: {path}")
    return {(r.video, r.object) for r in read_scores_csv(path)}


def prompted_objects(path: Path) -> set[tuple[str, int]]:
    batch = load_batch_prompts(path)
    return {(video, obj) for video, p in batch.videos.items() for obj in p.boxes}


def check_arms(
    results_dir: Path, prompts: Mapping[str, Path], test_sets: Sequence[str] = TEST_SETS
) -> list[tuple[str, str, str, int, int]]:
    """Rows (test_set, arm, hash, videos, objects); raises ArmsMismatch on any difference."""
    names = [ALONE, *(a.name for a in ARMS)]
    methods = {ALONE: ALONE, **{a.name: a.method for a in ARMS}}
    rows: list[tuple[str, str, str, int, int]] = []
    for split in test_sets:
        hashes = {n: file_hash(prompts[f"{n}:{split}"]) for n in names}
        if len(set(hashes.values())) != 1:
            odd = sorted(hashes)
            raise ArmsMismatch(f"{split}: arms used different prompts files: {odd} {hashes}")
        reference = prompted_objects(prompts[f"{ALONE}:{split}"])
        for name in names:
            found = scored_objects(results_dir / f"{methods[name]}_{split}.csv")
            if not found <= reference or {v for v, _ in found} != {v for v, _ in reference}:
                raise ArmsMismatch(f"{split}/{name}: videos or objects differ from the prompts")
            if found != scored_objects(results_dir / f"{ALONE}_{split}.csv"):
                raise ArmsMismatch(f"{split}/{name}: differs from the SAM 2 alone videos/objects")
            rows.append((split, name, hashes[name], len({v for v, _ in found}), len(found)))
    return rows


def write_check(path: Path, rows: Sequence[tuple[str, str, str, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CHECK_COLUMNS)
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--prompts-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/e7_arms_check.csv"))
    parser.add_argument("--splits", nargs="+", choices=TEST_SETS, default=list(TEST_SETS))
    parser.add_argument("--arm-prompts", nargs="*", default=[], metavar="ARM:SPLIT=PATH")
    args = parser.parse_args(argv)
    prompts_dir = args.prompts_dir
    if prompts_dir is None:
        if not os.environ.get(DATA_ROOT_ENV):
            parser.error(f"{DATA_ROOT_ENV} is not set (or pass --prompts-dir)")
        cfg = load_data_config()
        prompts_dir = cfg.data_root / cfg.output_root / "prompts"
    prompts = {
        f"{n}:{s}": prompts_dir / f"{s}.json"
        for n in [ALONE, *(a.name for a in ARMS)]
        for s in args.splits
    }
    for item in args.arm_prompts:
        key, _, path = item.partition("=")
        if key not in prompts:
            parser.error(f"unknown arm/split '{key}'")
        prompts[key] = Path(path)
    write_check(args.out, check_arms(args.results_dir, prompts, args.splits))


if __name__ == "__main__":
    main()
