"""E13 table: the chosen box source without and with the gate, against SAM 2 alone and E2.

    python -m experiments.e13.table [--results-dir results] [--choice results/e13_choice.csv]
        [--out results/e13.csv] [--tex-out results/e13.tex]

Per model with a go in results/e13_choice.csv and per test set: SAM 2 alone, E2 (rules + CSRT),
variant (a) (the chosen box source, no gate; Tiny only; when the chosen source is CSRT the existing
E2 / E8 run stands for it), variant (b) (with the frozen gate) and, for Tiny, the E7 rules + oracle
box run, marked oracle=True. Method names come from experiments.e13.make_configs. J&F is the mean
over objects with a 95% bootstrap interval over videos (seed 5415, 10,000 resamples); the changes
against SAM 2 alone and against E2 are paired on the same objects. Improved / worse / unchanged
videos use the main-table rule; worse videos are kept. Re-prompts and gated events come from the
events files, wrong re-prompts (IoU < 0.5 at the re-prompted frame) from the _reprompts files, FPS
from the timing files as in the main table. The exact columns (reprompts_exact, wrong_exact) count
only the re-prompts whose own frame is annotated and how many of those are wrong, so they do not
depend on the next annotated frame used by the wrong-re-prompt rule on SA-V.
Nothing is written unless every input exists.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.e2.table import alone_method, fix_method, load_events
from experiments.e8.table import load_reprompts
from experiments.e13 import make_configs
from src.eval.bootstrap import bootstrap_delta, bootstrap_mean
from src.eval.rows import ScoreRow
from src.eval.tables import count_changes, load_scores, load_timing, per_video_scores

TEST_SETS = ("davis_val", "sav_test")
ORACLE_METHOD = "fix_rules_oracle_box_tiny"
ORACLE_MODEL = "tiny"
DEFAULT_OUT = "results/e13.csv"
DEFAULT_TEX_OUT = "results/e13.tex"
TABLE_COLUMNS = (
    "model",
    "split",
    "method",
    "label",
    "role",
    "oracle",
    "videos",
    "objects",
    "JF",
    "JF_low",
    "JF_high",
    "delta_JF",
    "delta_low",
    "delta_high",
    "delta_vs_e2",
    "delta_vs_e2_low",
    "delta_vs_e2_high",
    "videos_improved",
    "videos_worse",
    "videos_unchanged",
    "reprompts",
    "wrong_reprompts",
    "reprompts_exact",
    "wrong_exact",
    "unscored_reprompts",
    "gated",
    "gated_low_score",
    "gated_agrees_with_sam2",
    "fps",
)
SET_TITLES = {"davis_val": "DAVIS 2017 val", "sav_test": "SA-V test subset"}


class E13TableError(ValueError):
    """An E13 table input is missing or inconsistent."""


@dataclass(frozen=True)
class Variant:
    method: str
    label: str
    role: str
    oracle: bool = False
    needs_reprompts: bool = True


def variants(model: str, arm: str) -> list[Variant]:
    """Rows after SAM 2 alone, in table order, with names from make_configs."""
    out = [Variant(fix_method(model), "rules + CSRT (E2)", "e2", needs_reprompts=model == "tiny")]
    own = make_configs.ungated_method(arm, model)
    if model in make_configs.UNGATED_MODELS and own != out[0].method:
        out.append(Variant(own, f"rules + {arm}", "variant_a"))
    out.append(Variant(make_configs.gated_method(arm, model), f"rules + {arm} + gate", "variant_b"))
    if model == ORACLE_MODEL:
        out.append(Variant(ORACLE_METHOD, "rules + oracle box (E7, oracle)", "oracle", True, False))
    return out


def _keys(rows: Sequence[ScoreRow]) -> set[tuple[str, int]]:
    return {(r.video, r.object) for r in rows}


def _empty_row(
    model: str, split: str, method: str, label: str, role: str, oracle: bool
) -> dict[str, object]:
    row: dict[str, object] = {c: "" for c in TABLE_COLUMNS}
    return row | {
        "model": model,
        "split": split,
        "method": method,
        "label": label,
        "role": role,
        "oracle": oracle,
    }


def _scores_and_fps(results: Path, method: str, split: str) -> tuple[list[ScoreRow], float]:
    rows = load_scores(results / f"{method}_{split}.csv")
    timing = load_timing(results / f"{method}_timing_{split}.csv")
    if {r.video for r in rows} != set(timing.frames):
        raise E13TableError(f"{method}/{split}: score videos differ from timing videos")
    return rows, sum(timing.frames.values()) / sum(timing.compute_s.values())


def _fill_scores(row: dict[str, object], rows: Sequence[ScoreRow], fps: float) -> None:
    per_video = per_video_scores(rows)
    interval = bootstrap_mean(per_video)
    row |= {
        "videos": len(per_video),
        "objects": len(rows),
        "JF": interval.estimate,
        "JF_low": interval.low,
        "JF_high": interval.high,
        "fps": fps,
    }


def _fill_changes(
    row: dict[str, object],
    rows: Sequence[ScoreRow],
    alone: Sequence[ScoreRow],
    e2: Sequence[ScoreRow] | None,
) -> None:
    new, base = per_video_scores(rows), per_video_scores(alone)
    delta = bootstrap_delta(new, base)
    improved, worse, unchanged = count_changes(base, new)
    row |= {
        "delta_JF": delta.estimate,
        "delta_low": delta.low,
        "delta_high": delta.high,
        "videos_improved": improved,
        "videos_worse": worse,
        "videos_unchanged": unchanged,
    }
    if e2 is not None:
        vs = bootstrap_delta(new, per_video_scores(e2))
        row |= {
            "delta_vs_e2": vs.estimate,
            "delta_vs_e2_low": vs.low,
            "delta_vs_e2_high": vs.high,
        }


def _fill_events(row: dict[str, object], events: pd.DataFrame) -> None:
    actions, reasons = events["action"], events["reason"]
    gated = actions == "gated"
    row |= {
        "reprompts": int((actions == "reprompted").sum()),
        "gated": int(gated.sum()),
        "gated_low_score": int((gated & (reasons == "low_score")).sum()),
        "gated_agrees_with_sam2": int((gated & (reasons == "agrees_with_sam2")).sum()),
    }


def _fill_wrong(row: dict[str, object], reprompts: pd.DataFrame, events: pd.DataFrame) -> None:
    if len(reprompts) != int((events["action"] == "reprompted").sum()):
        raise E13TableError(f"{row['method']}/{row['split']}: re-prompt file and events disagree")
    exact = reprompts["gt_frame_index"] == reprompts["frame_index"]
    row |= {
        "wrong_reprompts": int((reprompts["wrong"] == True).sum()),  # noqa: E712
        "reprompts_exact": int(exact.sum()),
        "wrong_exact": int((exact & (reprompts["wrong"] == True)).sum()),  # noqa: E712
        "unscored_reprompts": int(reprompts["wrong"].isna().sum()),
    }


def split_rows(results: Path, model: str, arm: str, split: str) -> list[dict[str, object]]:
    alone_rows, alone_fps = _scores_and_fps(results, alone_method(model), split)
    base = _empty_row(model, split, alone_method(model), "SAM 2 alone", "alone", False)
    _fill_scores(base, alone_rows, alone_fps)
    base["delta_JF"] = 0.0
    out = [base]
    e2_rows: list[ScoreRow] | None = None
    for v in variants(model, arm):
        label = f"{model}/{split}/{v.method}"
        rows, fps = _scores_and_fps(results, v.method, split)
        if _keys(rows) != _keys(alone_rows):
            raise E13TableError(f"{label}: scores a different set of objects than SAM 2 alone")
        events = load_events(results / f"events_{v.method}_{split}.csv")
        row = _empty_row(model, split, v.method, v.label, v.role, v.oracle)
        _fill_scores(row, rows, fps)
        _fill_changes(row, rows, alone_rows, None if v.role == "e2" else e2_rows)
        _fill_events(row, events)
        path = results / f"{v.method}_{split}_reprompts.csv"
        if v.needs_reprompts or path.is_file():
            _fill_wrong(row, load_reprompts(path), events)
        e2_rows = rows if v.role == "e2" else e2_rows
        out.append(row)
    return out


def build_rows(
    results: Path, choice: Path, splits: Sequence[str] = TEST_SETS
) -> list[dict[str, object]]:
    arms = dict(make_configs.go_choices(choice))
    table: list[dict[str, object]] = []
    for model, arm in arms.items():
        for split in splits:
            table += split_rows(results, model, arm, split)
    return table


def write_table(path: Path, table: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TABLE_COLUMNS)
        for row in table:
            writer.writerow(
                [repr(v) if isinstance(v, float) else v for v in (row[c] for c in TABLE_COLUMNS)]
            )
    os.replace(temp, path)


def _signed(value: str) -> str:
    return f"{100 * float(value):+.1f}"


def _interval(row: dict[str, str], key: str, low: str, high: str) -> str:
    if row[low] == "":
        return "--"
    return f"{_signed(row[key])} [{_signed(row[low])}, {_signed(row[high])}]"


def _escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def _cells(row: dict[str, str]) -> list[str]:
    counts = "--"
    if row["videos_improved"] != "":
        counts = f"{row['videos_improved']} / {row['videos_worse']} / {row['videos_unchanged']}"
    wrong = f"{row['reprompts']} ({row['wrong_reprompts'] or '--'})" if row["reprompts"] else "--"
    exact = f"{row['reprompts_exact']} ({row['wrong_exact']})" if row["reprompts_exact"] else "--"
    return [
        _interval(row, "delta_JF", "delta_low", "delta_high"),
        counts,
        wrong,
        exact,
        f"{float(row['fps']):.2f}",
    ]


def latex_table(rows: Sequence[dict[str, str]]) -> str:
    keys = list(dict.fromkeys((r["model"], r["method"]) for r in rows))
    by_key = {(r["model"], r["method"], r["split"]): r for r in rows}
    splits = [s for s in TEST_SETS if any(r["split"] == s for r in rows)]
    head = " & ".join(rf"\multicolumn{{5}}{{c}}{{{SET_TITLES[s]}}}" for s in splits)
    lines = [
        r"\begin{tabular}{ll" + "ccccr" * len(splits) + "}",
        r"\toprule",
        f"Model & Method & {head} " + r"\\",
        " & & "
        + " & ".join(
            [
                r"$\Delta$ J\&F vs alone",
                "better / worse / same",
                "re-prompts (wrong)",
                "at annotated (wrong)",
                "FPS",
            ]
            * len(splits)
        )
        + r" \\",
        r"\midrule",
    ]
    for model, method in keys:
        cells: list[str] = []
        for split in splits:
            row = by_key.get((model, method, split))
            if row is None:
                raise E13TableError(f"no row for {model}/{method}/{split}")
            cells += _cells(row)
        label = next(r["label"] for r in rows if (r["model"], r["method"]) == (model, method))
        lines.append(" & ".join([model, _escape(label), *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TABLE_COLUMNS:
            raise E13TableError(f"{path}: columns are {reader.fieldnames}")
        return list(reader)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--choice", type=Path, default=make_configs.CHOICE_PATH)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--tex-out", type=Path, default=Path(DEFAULT_TEX_OUT))
    args = parser.parse_args(argv)
    try:
        table = build_rows(args.results_dir, args.choice)
    except (ValueError, FileNotFoundError) as error:
        sys.stdout.write(f"refused, nothing written: {error}\n")
        return 1
    write_table(args.out, table)
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.write_text(latex_table(read_table(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
