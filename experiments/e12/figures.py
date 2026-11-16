"""E12 figures (analysis), built only from the E12 tables and results/e7.csv.

    python -m experiments.e12.figures [--results-dir results] [--figure-dir figures]

Writes e12_by_quality.png, e7_decomposition.png and e12_false_alarms.png. Oracle arms and
ground-truth based quantities are labelled as such in the legend and the titles.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

TEST_SETS = ("davis_val", "sav_test")
TEST_SET_TITLES = {"davis_val": "DAVIS 2017 val", "sav_test": "SA-V test subset"}
BINS = ("lost", "partial", "good", "all")
POINTS = 100.0
DPI = 150
BY_QUALITY_SERIES = (
    ("tiny", "fix_rules_csrt_tiny", "E2 Tiny (rules + CSRT)", "#1f77b4"),
    ("large", "fix_rules_csrt_large", "E2 Large (rules + CSRT)", "#2ca02c"),
    ("tiny", "fix_rules_oracle_box_tiny", "E7 oracle: rules + oracle box (Tiny)", "#ff7f0e"),
    (
        "tiny",
        "fix_oracle_flags_oracle_box_tiny",
        "E7 oracle: oracle flags + oracle box (Tiny)",
        "#d62728",
    ),
)
E7_ARMS = (
    ("rules_csrt", "rules + CSRT (real)", False, "#1f77b4"),
    ("rules_oracle_box", "rules + oracle box (oracle)", True, "#ff7f0e"),
    ("oracle_flags_csrt", "oracle flags + CSRT (oracle)", True, "#9467bd"),
    ("oracle_flags_oracle_box", "oracle flags + oracle box (oracle)", True, "#d62728"),
)
E7_BASE_ARM = "sam2_tiny"
STRATEGY_LABELS = (
    ("accept_all", "accept all"),
    ("best_grid", "best grid"),
    ("perfect_verifier", "perfect verifier\n(oracle)"),
)


class FigureError(ValueError):
    """An input table is missing or lacks a needed row."""


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FigureError(f"missing input file: {path}")
    return pd.read_csv(path)


def plot_by_quality(table: pd.DataFrame, figure_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(TEST_SETS), figsize=(13.0, 5.2), dpi=DPI, sharey=True)
    width = 0.8 / len(BY_QUALITY_SERIES)
    for ax, test_set in zip(axes, TEST_SETS, strict=True):
        part = table[table["test_set"] == test_set]
        ticks = []
        for name in BINS:
            count = part[(part["bin"] == name) & (part["method"] == BY_QUALITY_SERIES[0][1])]
            ticks.append(f"{name}\nn={int(count['objects'].iloc[0])} (Tiny)")
        for s, (model, method, label, colour) in enumerate(BY_QUALITY_SERIES):
            rows = part[(part["model"] == model) & (part["method"] == method)]
            rows = rows.set_index("bin").reindex(list(BINS))
            delta = rows["delta_jf"].astype(float) * POINTS
            err = [
                (delta - rows["ci_low"].astype(float) * POINTS).fillna(0),
                (rows["ci_high"].astype(float) * POINTS - delta).fillna(0),
            ]
            x = [b + (s - (len(BY_QUALITY_SERIES) - 1) / 2) * width for b in range(len(BINS))]
            ax.bar(
                x,
                delta.fillna(0),
                width,
                yerr=err,
                capsize=2,
                color=colour,
                hatch="//" if "oracle" in label else None,
                label=label,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(BINS)), ticks)
        ax.set_title(TEST_SET_TITLES[test_set])
        ax.set_xlabel("SAM 2 alone J&F of the object (bins use ground truth)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("change in J&F vs SAM 2 alone (J&F points)\n95% CI over videos")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("E12 analysis: where recovery helps, by how well SAM 2 alone tracked the object")
    return save(fig, figure_dir / "e12_by_quality.png", bottom=0.14)


def plot_decomposition(e7: pd.DataFrame, figure_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(TEST_SETS), figsize=(13.0, 5.2), dpi=DPI)
    for ax, test_set in zip(axes, TEST_SETS, strict=True):
        part = e7[e7["test_set"] == test_set].set_index("arm")
        for arm in (*(a[0] for a in E7_ARMS), E7_BASE_ARM):
            if arm not in part.index:
                raise FigureError(f"results/e7.csv has no {arm} row for {test_set}")
        jf = part["J&F"].astype(float) * POINTS
        delta = part["delta_JF"].astype(float) * POINTS
        base = float(jf[E7_BASE_ARM])
        for i, (arm, label, oracle, colour) in enumerate(E7_ARMS):
            value = float(jf[arm])
            ax.bar(
                i,
                value,
                color=colour,
                hatch="//" if oracle else None,
                label=label,
            )
            ax.text(
                i,
                value + 0.3,
                f"{float(delta[arm]):+.1f}",
                ha="center",
                fontsize=9,
            )
        ax.axhline(base, color="black", linestyle="--", label=f"SAM 2 alone ({base:.1f})")
        ax.set_ylim(base - 12, base + 12)
        ax.set_xticks([])
        ax.set_ylabel("J&F (points)")
        ax.set_title(TEST_SET_TITLES[test_set])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(
        "E7 decomposition (Tiny), analysis: oracle arms hatched; number above a bar = change vs "
        "SAM 2 alone in J&F points (y axis truncated)"
    )
    return save(fig, figure_dir / "e7_decomposition.png", bottom=0.14)


def plot_false_alarms(split: pd.DataFrame, bound: pd.DataFrame, figure_dir: Path) -> Path:
    models = list(dict.fromkeys(split["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(11.0, 5.0), dpi=DPI)
    for ax, model in zip(axes, models, strict=True):
        classes = split[split["model"] == model].set_index("class")["gain_sum"].astype(float)
        strategies = bound[(bound["model"] == model) & (bound["kind"] == "strategy")]
        strategies = strategies.set_index("name")["gain_sum"].astype(float)
        labels = ["real failures\n(GT split)", "false alarms\n(GT split)"]
        values = [classes["real_failure"], classes["false_alarm"]]
        colours = ["#2ca02c", "#d62728"]
        for key, label in STRATEGY_LABELS:
            labels.append(label)
            values.append(strategies[key])
            colours.append("#7f7f7f" if key != "perfect_verifier" else "#9467bd")
        bars = ax.bar(range(len(values)), values, color=colours)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (4 if value >= 0 else -4),
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(values)), labels, fontsize=8)
        ax.set_ylabel("summed IoU gain over re-prompt candidates (SA-V val)")
        ax.set_title(f"SAM 2 {model.capitalize()}")
        ax.margins(y=0.15)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        fontsize=10,
        t="E12 analysis: gain from re-prompts on real failures vs false alarms (SA-V val, "
        "accept-all run); the frozen choice (accept nothing) is 0",
    )
    return save(fig, figure_dir / "e12_false_alarms.png")


def save(fig: Figure, path: Path, bottom: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def make_figures(results_dir: Path, figure_dir: Path) -> list[Path]:
    by_quality = read_table(results_dir / "e12_by_quality.csv")
    e7 = read_table(results_dir / "e7.csv")
    split = read_table(results_dir / "e12_false_alarms.csv")
    bound = read_table(results_dir / "e12_upper_bound.csv")
    return [
        plot_by_quality(by_quality, figure_dir),
        plot_decomposition(e7, figure_dir),
        plot_false_alarms(split, bound, figure_dir),
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    for path in make_figures(args.results_dir, args.figure_dir):
        print(path)


if __name__ == "__main__":
    main()
