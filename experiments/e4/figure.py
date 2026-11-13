"""E4 figure: precision-recall curves of the threshold sweeps, one panel per model.

    python -m experiments.e4.figure [--table results/e4.csv] [--figure-dir figures]

Built from results/e4.csv only. Lines = score and area sweeps, dots = the 2-D grid, star = the
frozen operating point.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURE_NAME = "e4_pr_curve.png"
FIGURE_SIZE = (10.0, 4.5)
DPI = 150
LABELS = {"score_threshold": "score threshold swept", "area_ratio": "area ratio swept"}


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_e4(table_path: Path, figure_dir: Path) -> Path:
    rows = read_table(table_path)
    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, axes = plt.subplots(1, len(models), figsize=FIGURE_SIZE, dpi=DPI, squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        mine = [r for r in rows if r["model"] == model]
        grid = [r for r in mine if r["swept"] == "both"]
        ax.scatter(
            [float(r["recall"]) for r in grid],
            [float(r["precision"]) for r in grid],
            s=6,
            color="lightgray",
            label="2-D grid",
        )
        for swept, label in LABELS.items():
            line = sorted(
                (r for r in mine if r["swept"] == swept), key=lambda r: float(r["recall"])
            )
            ax.plot(
                [float(r["recall"]) for r in line],
                [float(r["precision"]) for r in line],
                marker="o",
                markersize=3,
                label=label,
            )
        for r in mine:
            if r["frozen"] == "True" and r["swept"] == "both":
                ax.plot(float(r["recall"]), float(r["precision"]), "k*", markersize=13,
                        label="frozen point")  # fmt: skip
        ax.set_title(f"{model} (SA-V val windows)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / FIGURE_NAME
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--table", type=Path, default=Path("results/e4.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    plot_e4(args.table, args.figure_dir)


if __name__ == "__main__":
    main()
