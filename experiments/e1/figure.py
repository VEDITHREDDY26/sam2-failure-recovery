"""E1 figure: J&F per method, grouped by test set, with FPS printed on each bar."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURE_NAME = "e1_jf_fps.png"
FIGURE_SIZE = (8.0, 4.5)
DPI = 150
BAR_WIDTH = 0.25
Y_RANGE = (0.0, 1.0)
LABEL_OFFSET = 0.01


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_jf_fps(table_path: Path, figure_dir: Path) -> Path:
    rows = read_table(table_path)
    test_sets = list(dict.fromkeys(r["test_set"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    by_key = {(r["method"], r["test_set"]): r for r in rows}
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    offset = (len(methods) - 1) / 2
    for i, method in enumerate(methods):
        xs = [g + (i - offset) * BAR_WIDTH for g in range(len(test_sets))]
        picked = [by_key[(method, t)] for t in test_sets]
        heights = [float(r["J&F"]) for r in picked]
        ax.bar(xs, heights, BAR_WIDTH, label=method)
        for x, h, r in zip(xs, heights, picked, strict=True):
            ax.text(x, h + LABEL_OFFSET, f"{float(r['fps']):.2f} FPS", ha="center", fontsize=7)
    ax.set_xticks(range(len(test_sets)), test_sets)
    ax.set_xlabel("Test set")
    ax.set_ylabel("J&F (mean over objects)")
    ax.set_ylim(*Y_RANGE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / FIGURE_NAME
    fig.savefig(path)
    plt.close(fig)
    return path
