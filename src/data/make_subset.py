"""Seeded 60-video subset of the SA-V test list.

    python -m src.data.make_subset --seed 5415 --size 60 [--list-file PATH] [--check]

Reads the 150 test video ids, takes `random.Random(seed).sample(sorted(ids), size)` and writes
the chosen ids sorted, one per line, to configs/splits/sav_test_subset60.txt. The file is made
once, before any SA-V test result exists, and is never overwritten with different ids.
`--check` re-derives the subset and exits 1 if the committed file differs.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from src.common.config import load_config
from src.data.sav import list_sav_videos

TEST_VIDEOS = 150  # dataset fact: SA-V test has 150 videos (SAM 2 README)
DEFAULT_SEED = 5415
DEFAULT_SIZE = 60
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "splits" / "sav_test_subset60.txt"
DEFAULT_RESULTS = REPO_ROOT / "results"
RESULTS_TOKEN = "sav_test"


class SubsetError(RuntimeError):
    """The subset cannot be made or does not match."""


def read_id_list(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"SA-V test list not found: {path}")
    return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())


def validate_ids(ids: Sequence[str]) -> None:
    if len(set(ids)) != len(ids):
        raise SubsetError("the test list has duplicate video ids")
    if len(ids) != TEST_VIDEOS:
        raise SubsetError(f"expected {TEST_VIDEOS} test videos, found {len(ids)}")


def pick_subset(ids: Sequence[str], seed: int, size: int) -> tuple[str, ...]:
    validate_ids(ids)
    if not 0 < size <= len(ids):
        raise SubsetError(f"size {size} outside 1..{len(ids)}")
    return tuple(sorted(random.Random(seed).sample(sorted(ids), size)))


def load_subset(path: Path) -> tuple[str, ...]:
    """Ids of a subset file, in file order; '#' lines and blank lines are skipped."""
    if not path.is_file():
        raise FileNotFoundError(f"subset file not found: {path}")
    lines = (line.strip() for line in path.read_text().splitlines())
    return tuple(line for line in lines if line and not line.startswith("#"))


def render(ids: Sequence[str], seed: int, size: int, source: str, command: str) -> str:
    header = [
        f"# SA-V test subset: seed {seed}, size {size}",
        f"# source list: {source}",
        f"# command: {command}",
    ]
    return "\n".join([*header, *ids]) + "\n"


def results_naming_sav_test(results_dir: Path) -> list[Path]:
    if not results_dir.is_dir():
        return []
    return sorted(
        p for p in results_dir.rglob("*") if RESULTS_TOKEN in str(p.relative_to(results_dir))
    )


def _command(args: argparse.Namespace) -> str:
    parts = ["python -m src.data.make_subset", f"--seed {args.seed}", f"--size {args.size}"]
    if args.list_file is not None:
        parts.append(f"--list-file {args.list_file.name}")
    return " ".join(parts)


def _source_ids(args: argparse.Namespace) -> tuple[tuple[str, ...], str]:
    if args.list_file is not None:
        return read_id_list(args.list_file), args.list_file.name
    cfg = load_config(args.config)
    return list_sav_videos(cfg, "test"), f"{cfg.sav.test_split}.txt"


def _check(args: argparse.Namespace, ids: Sequence[str]) -> int:
    expected = pick_subset(ids, args.seed, args.size)
    actual = load_subset(args.output)
    if set(actual) - set(ids):
        print(f"FAIL: ids not in the test list: {sorted(set(actual) - set(ids))}")
        return 1
    if actual != expected:
        print(f"FAIL: {args.output} differs from the subset derived with seed {args.seed}")
        return 1
    print(f"OK: {args.output} matches seed {args.seed}, size {args.size} ({len(actual)} videos)")
    return 0


def _write(args: argparse.Namespace, ids: Sequence[str], source: str) -> int:
    chosen = pick_subset(ids, args.seed, args.size)
    if args.output.exists():
        if load_subset(args.output) == chosen:
            print(f"unchanged: {args.output} already holds this subset")
            return 0
        print(f"FAIL: {args.output} exists with different ids; refusing to overwrite")
        return 1
    found = results_naming_sav_test(args.results_dir)
    if found:
        print(f"FAIL: results already exist for SA-V test ({found[0]}); the subset must come first")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(chosen, args.seed, args.size, source, _command(args)))
    print(f"wrote {len(chosen)} ids to {args.output}")
    return 0


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.data.make_subset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--list-file", type=Path, default=None, help="copy of sav_test.txt")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        ids, source = _source_ids(args)
        return _check(args, ids) if args.check else _write(args, ids, source)
    except (SubsetError, FileNotFoundError) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
