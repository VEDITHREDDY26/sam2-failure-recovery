"""Fetch the pinned evaluators into third_party/ (safe to run again).

python -m src.eval.get_evaluators
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.eval.evaluators import PINS, THIRD_PARTY, EvaluatorError, Pin, check_checkout, git


def fetch(pin: Pin, root: Path = THIRD_PARTY) -> Path:
    repo = pin.path(root)
    try:
        return check_checkout(pin, root)
    except EvaluatorError:
        if (repo / ".git").exists() and git(repo, "status", "--porcelain"):
            raise
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        git(repo, "init", "--quiet")
        git(repo, "remote", "add", "origin", pin.url)
    if pin.sparse_paths:
        git(repo, "sparse-checkout", "set", *pin.sparse_paths)
    git(repo, "fetch", "--quiet", "--depth", "1", "origin", pin.commit)
    git(repo, "checkout", "--quiet", pin.commit)
    return check_checkout(pin, root)


def main() -> None:
    for pin in PINS:
        try:
            path = fetch(pin)
        except EvaluatorError as error:
            sys.exit(str(error))
        sys.stdout.write(f"{pin.name}: {pin.commit[:10]} ready at {path}\n")


if __name__ == "__main__":
    main()
