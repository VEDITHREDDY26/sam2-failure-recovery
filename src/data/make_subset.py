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
    """Read video ids from a plain text file, one per line, ignoring blank lines.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"SA-V test list not found: {path}")
    return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())


def validate_ids(ids: Sequence[str]) -> None:
    """Check that the SA-V test list has no duplicates and exactly the expected 150 ids.

    Raises:
        SubsetError: If the list has duplicates or the wrong length.
    """
    if len(set(ids)) != len(ids):
        raise SubsetError("the test list has duplicate video ids")
    if len(ids) != TEST_VIDEOS:
        raise SubsetError(f"expected {TEST_VIDEOS} test videos, found {len(ids)}")


def pick_subset(ids: Sequence[str], seed: int, size: int) -> tuple[str, ...]:
    """Draw the seeded random subset of video ids.

    The ids are sorted before sampling so the result depends only on the seed and the set of ids,
    not on the order of the source file.

    Args:
        ids: All test video ids.
        seed: Seed of the random generator.
        size: Number of videos to pick (1 to len(ids)).

    Returns:
        The chosen ids, sorted.

    Raises:
        SubsetError: If the list is invalid or size is out of range.
    """
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
    """Format the subset file: '#' header lines (seed, source, command), then one id per line."""
    header = [
        f"# SA-V test subset: seed {seed}, size {size}",
        f"# source list: {source}",
        f"# command: {command}",
    ]
    return "\n".join([*header, *ids]) + "\n"


def results_naming_sav_test(results_dir: Path) -> list[Path]:
    """List files under the results folder whose relative path mentions the SA-V test split."""
    if not results_dir.is_dir():
        return []
    return sorted(
        p for p in results_dir.rglob("*") if RESULTS_TOKEN in str(p.relative_to(results_dir))
    )


def _command(args: argparse.Namespace) -> str:
    """Rebuild the command line (without paths) that produced the subset, for the file header."""
    parts = ["python -m src.data.make_subset", f"--seed {args.seed}", f"--size {args.size}"]
    if args.list_file is not None:
        parts.append(f"--list-file {args.list_file.name}")
    return " ".join(parts)


def _source_ids(args: argparse.Namespace) -> tuple[tuple[str, ...], str]:
    """Return the test video ids and the name of the list they came from.

    Uses --list-file when given, otherwise the SA-V test list of the data configuration.
    """
    if args.list_file is not None:
        return read_id_list(args.list_file), args.list_file.name
    cfg = load_config(args.config)
    return list_sav_videos(cfg, "test"), f"{cfg.sav.test_split}.txt"


def _check(args: argparse.Namespace, ids: Sequence[str]) -> int:
    """Re-derive the subset and compare it with the stored file.

    Returns:
        0 if the file matches the subset for the seed and size, otherwise 1 after printing why.
    """
    # Check both membership in the test list and exact equality with the seeded draw.
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
    """Write the subset file unless that would be unsafe.

    An existing file with the same ids is left alone, one with different ids is never overwritten,
    and nothing is written once results for the SA-V test split exist, so the subset is fixed
    before any test result is seen.

    Returns:
        0 on success or no change, 1 on refusal.
    """
    chosen = pick_subset(ids, args.seed, args.size)
    # Never replace an existing subset with different ids.
    if args.output.exists():
        if load_subset(args.output) == chosen:
            print(f"unchanged: {args.output} already holds this subset")
            return 0
        print(f"FAIL: {args.output} exists with different ids; refusing to overwrite")
        return 1
    # The subset must be fixed before any SA-V test result exists.
    found = results_naming_sav_test(args.results_dir)
    if found:
        print(f"FAIL: results already exist for SA-V test ({found[0]}); the subset must come first")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(chosen, args.seed, args.size, source, _command(args)))
    print(f"wrote {len(chosen)} ids to {args.output}")
    return 0


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments of the subset tool."""
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
    """Command-line entry point: create or verify the fixed SA-V test subset.

    Reads the SA-V test list (from the data configuration or --list-file). Without --check it
    writes the subset file (default configs/splits/sav_test_subset60.txt); with --check it only
    compares.

    Args:
        argv: Command-line arguments; sys.argv is used when None.

    Returns:
        0 on success; 1 on mismatch, refusal, a missing file or an invalid list.
    """
    args = _parse(argv)
    try:
        ids, source = _source_ids(args)
        return _check(args, ids) if args.check else _write(args, ids, source)
    except (SubsetError, FileNotFoundError) as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
