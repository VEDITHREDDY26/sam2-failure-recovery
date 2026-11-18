"""E13 figure: change in J&F against SAM 2 alone for each rung of the ladder, per test set.

    python -m experiments.e13.figure [--table results/e13.csv] [--figure-dir figures]

One panel per test set. Tiny: rules + CSRT (E2), rules + the chosen box, rules + the chosen box +
gate, and the oracle-box upper reference (hatched, labelled oracle). Large: gated next to E2 when
present. Bars are the change in J&F with the 95% bootstrap interval. Reads results/e13.csv only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.e13.table import SET_TITLES, TABLE_COLUMNS  # noqa: E402

FIGURE_NAME = "e13_ladder.png"
FIGURE_SIZE = (11.0, 4.8)
DPI = 150
ROLE_ORDER = ("e2", "variant_a", "variant_b", "oracle")
COLOURS = {"e2": "#8c8c8c", "variant_a": "#6baed6", "variant_b": "#2171b5", "oracle": "#ffffff"}
GAP = 0.8


def read_table(path: Path) -> list[dict[str, str]]:
    """Read the E13 table CSV, checking that its columns are exactly TABLE_COLUMNS.

    Args:
        path: Path to results/e13.csv.

    Returns:
        The rows as dictionaries of strings keyed by column name.

    Raises:
        ValueError: If the header does not match TABLE_COLUMNS.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TABLE_COLUMNS:
            raise ValueError(f"{path}: columns are {reader.fieldnames}")
        return list(reader)


def panel_bars(rows: list[dict[str, str]], split: str) -> list[tuple[float, dict[str, str]]]:
    """(x position, row) for the ladder rows of one split, Tiny first and Large after a gap."""
    bars: list[tuple[float, dict[str, str]]] = []
    x = 0.0
    # Models appear in file order; within a model the bars follow ROLE_ORDER.
    for model in dict.fromkeys(r["model"] for r in rows):
        picked = [
            r
            for role in ROLE_ORDER
            for r in rows
            if r["model"] == model and r["split"] == split and r["role"] == role
        ]
        for row in picked:
            bars.append((x, row))
            x += 1.0
        # Leave a gap between the Tiny and Large groups.
        x += GAP
    return bars


def tick_label(row: dict[str, str]) -> str:
    """Build the multi-line x-axis label of one bar from its table row.

    Args:
        row: One row of the E13 table (needs the "model" and "label" fields).

    Returns:
        The model name followed by the row label broken over several lines.
    """
    text = row["label"].removeprefix("rules + ").replace(" + ", "\n+ ").replace("_", "\n")
    text = text.replace(" (", "\n(")
    return f"{row['model']}\n{text}"


def plot_ladder(table_path: Path, figure_dir: Path) -> Path:
    """Draw the ladder figure (change in J&F against SAM 2 alone) and save it as a PNG.

    Args:
        table_path: Path to results/e13.csv.
        figure_dir: Folder to save the figure in; it is created if needed.

    Returns:
        The path of the saved figure.

    Raises:
        ValueError: If the table has the wrong columns.
    """
    # One panel per test set that has rows in the table.
    rows = read_table(table_path)
    splits = [s for s in SET_TITLES if any(r["split"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(splits), figsize=FIGURE_SIZE, dpi=DPI, squeeze=False)
    for ax, split in zip(axes[0], splits, strict=True):
        bars = panel_bars(rows, split)
        # Bars are in percentage points: the table stores J&F on a 0 to 1 scale.
        for x, row in bars:
            delta = 100 * float(row["delta_JF"])
            low, high = 100 * float(row["delta_low"]), 100 * float(row["delta_high"])
            oracle = row["oracle"] == "True"
            ax.bar(
                x,
                delta,
                0.8,
                color=COLOURS[row["role"]],
                edgecolor="black",
                hatch="//" if oracle else None,
                yerr=[[delta - low], [high - delta]],
                capsize=3,
            )
        # Zero line: bars above it improve on SAM 2 alone, bars below it are worse.
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(
            [x for x, _ in bars],
            [tick_label(r) for _, r in bars],
            fontsize=6,
        )
        ax.set_title(SET_TITLES[split])
        ax.set_ylabel("change in J&F vs SAM 2 alone (points)")
    # Save the figure and release it so repeated calls do not pile up figures.
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / FIGURE_NAME
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Build the E13 ladder figure from the E13 table (command line).

    Reads --table (default results/e13.csv) and writes figures/e13_ladder.png in --figure-dir.

    Args:
        argv: Command-line arguments; None means sys.argv.

    Returns:
        0 on success, 1 if the table file is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--table", type=Path, default=Path("results/e13.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    if not args.table.is_file():
        sys.stdout.write(f"{args.table} is missing: run experiments.e13.table first\n")
        return 1
    plot_ladder(args.table, args.figure_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
