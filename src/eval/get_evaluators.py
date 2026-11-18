"""Fetch the pinned evaluators into third_party/ (safe to run again).

python -m src.eval.get_evaluators
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.eval.evaluators import PINS, THIRD_PARTY, EvaluatorError, Pin, check_checkout, git


def fetch(pin: Pin, root: Path = THIRD_PARTY) -> Path:
    """Make sure a pinned checkout exists at its commit and return its folder.

    Does nothing if the checkout is already correct. A checkout with local changes is left alone
    and the error is re-raised.

    Args:
        pin: Pinned repository.
        root: Folder that holds the checkouts.

    Raises:
        EvaluatorError: If the checkout has local modifications or git fails.
    """
    repo = pin.path(root)
    # Already correct: nothing to do. Local modifications are never overwritten.
    try:
        return check_checkout(pin, root)
    except EvaluatorError:
        if (repo / ".git").exists() and git(repo, "status", "--porcelain"):
            raise
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        git(repo, "init", "--quiet")
        git(repo, "remote", "add", "origin", pin.url)
    # Fetch only the pinned commit (depth 1) and, for SA-V, only its sav_dataset folder.
    if pin.sparse_paths:
        git(repo, "sparse-checkout", "set", *pin.sparse_paths)
    git(repo, "fetch", "--quiet", "--depth", "1", "origin", pin.commit)
    git(repo, "checkout", "--quiet", pin.commit)
    return check_checkout(pin, root)


def main() -> None:
    """Command-line entry point: fetch every pinned evaluator and print where it is ready.

    Raises:
        SystemExit: With the error text (status 1) if a checkout cannot be made.
    """
    for pin in PINS:
        try:
            path = fetch(pin)
        except EvaluatorError as error:
            sys.exit(str(error))
        sys.stdout.write(f"{pin.name}: {pin.commit[:10]} ready at {path}\n")


if __name__ == "__main__":
    main()
