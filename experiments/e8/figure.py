"""E8 figure: J&F of SAM 2 alone and the two tracker variants per test set, wrong re-prompts shown.

    python -m experiments.e8.figure [--table results/e8.csv] [--figure-dir figures]

Built from results/e8.csv only.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURE_NAME = "e8_jf.png"
FIGURE_SIZE = (7.0, 4.5)
DPI = 150
BAR_WIDTH = 0.25
Y_RANGE = (0.0, 1.0)
LABEL_OFFSET = 0.01


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_e8(table_path: Path, figure_dir: Path) -> Path:
    rows = read_table(table_path)
    test_sets = list(dict.fromkeys(r["test_set"] for r in rows))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    by_key = {(r["variant"], r["test_set"]): r for r in rows}
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    offset = (len(variants) - 1) / 2
    for i, variant in enumerate(variants):
        xs = [g + (i - offset) * BAR_WIDTH for g in range(len(test_sets))]
        picked = [by_key[(variant, t)] for t in test_sets]
        heights = [float(r["J&F"]) for r in picked]
        ax.bar(xs, heights, BAR_WIDTH, label=variant)
        for x, h, r in zip(xs, heights, picked, strict=True):
            if r["wrong_reprompts"]:
                text = f"{r['wrong_reprompts']}/{r['reprompts']} wrong"
                ax.text(x, h + LABEL_OFFSET, text, ha="center", fontsize=7)
    ax.set_xticks(range(len(test_sets)), test_sets)
    ax.set_xlabel("Test set (Tiny)")
    ax.set_ylabel("J&F (mean over objects)")
    ax.set_ylim(*Y_RANGE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / FIGURE_NAME
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--table", type=Path, default=Path("results/e8.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    plot_e8(args.table, args.figure_dir)


if __name__ == "__main__":
    main()
