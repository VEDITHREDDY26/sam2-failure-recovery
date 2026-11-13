"""E3 figure: J&F (left) and detector recall (right) with each rule removed, per test set.

    python -m experiments.e3.figure [--table results/e3.csv] [--figure-dir figures]

Built from results/e3.csv only.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURE_NAME = "e3_ablation.png"
FIGURE_SIZE = (11.0, 4.5)
DPI = 150
BAR_WIDTH = 0.35
Y_RANGE = (0.0, 1.0)
PANELS = (("J&F", "J&F (mean over objects)"), ("recall", "Detector recall (annotation windows)"))


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_e3(table_path: Path, figure_dir: Path) -> Path:
    rows = read_table(table_path)
    removed = list(dict.fromkeys(r["removed_rule"] for r in rows))
    test_sets = list(dict.fromkeys(r["test_set"] for r in rows))
    by_key = {(r["removed_rule"], r["test_set"]): r for r in rows}
    fig, axes = plt.subplots(1, len(PANELS), figsize=FIGURE_SIZE, dpi=DPI)
    offset = (len(test_sets) - 1) / 2
    for ax, (column, label) in zip(axes, PANELS, strict=True):
        for i, test_set in enumerate(test_sets):
            xs = [g + (i - offset) * BAR_WIDTH for g in range(len(removed))]
            heights = [float(by_key[(r, test_set)][column]) for r in removed]
            ax.bar(xs, heights, BAR_WIDTH, label=test_set)
        ax.set_xticks(range(len(removed)), [f"no {r}" if r != "none" else "full" for r in removed])
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.set_xlabel("Rule removed (Tiny)")
        ax.set_ylabel(label)
        ax.set_ylim(*Y_RANGE)
        ax.legend(fontsize=8)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / FIGURE_NAME
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--table", type=Path, default=Path("results/e3.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    plot_e3(args.table, args.figure_dir)


if __name__ == "__main__":
    main()
