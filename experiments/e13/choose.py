"""E13 stage 1 table, stage 2 gate grid and the choice of box source and gate per model.

    python -m experiments.e13.choose [--results-dir results] [--config-dir configs/recovery]

Reads results/e13_pilot_candidates_<model>_sav_val.csv and the matching timing file (written by
experiments.e13.pilot --merge) and writes results/e13_pilot.csv (per model, arm and view),
results/e13_gate_grid.csv (per model, arm, view, s_min, a_max) and results/e13_choice.csv. A gate
file configs/recovery/box_gate_<model>.yaml is written only when the choice is a go.

Views: `exact` keeps candidates whose own frame is annotated (scored_frame == frame_index); `all`
keeps every candidate (each scored at the next annotated frame); `first_of_object` keeps the first
candidate of each object, over all candidates. The choice and the go rule use `exact` only: a box
at t is scored against the annotation at t, because the next-annotated-frame rule counts correct
boxes as wrong when the object moves fast. The grid is also written for `all`, for information.

A "re-prompt" is a candidate whose box source returned a box (status ok), that passes the gate and
then the frozen box check. useful = re-prompt at a real failure with a box of IoU >= 0.5 with the
ground truth; harmful = re-prompt at a not-visible candidate or with a box of IoU < 0.5 with the
ground truth; every other scored re-prompt is neutral. Unscored candidates are left out and counted.
Choice: highest net, then fewer re-prompts, then the simpler arm (ARM_ORDER), then the less
restrictive gate. Go only if net > 0 and net > the net of csrt_reinit with no gate.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.e13.pilot import MODELS, RESULTS_DIR, SPLIT
from src.recovery.config import CONFIG_DIR

ARM_ORDER = (
    "motion",
    "csrt_reinit",
    "csrt_continuous",
    "vit_reinit",
    "vit_continuous",
    "vit_anchored",
)
S_MIN_GRID = (None, 0.2, 0.3, 0.4, 0.5, 0.6)
A_MAX_GRID = (None, 0.7, 0.5, 0.3)
GOOD_IOU = 0.5
VIEWS = ("exact", "all", "first_of_object")
GRID_VIEWS = ("exact", "all")
CHOICE_VIEW = "exact"
BASELINE_ARM = "csrt_reinit"
PILOT_COLUMNS = (
    "model",
    "arm",
    "view",
    "candidates",
    "real_failure",
    "false_alarm",
    "not_visible",
    "unscored",
    "status_ok",
    "status_lost",
    "status_not_initialised",
    "box_check_pass",
    "real_good_box_share",
    "real_copies_sam2_share",
    "false_alarm_copies_sam2_share",
    "ms_per_frame",
)
GRID_COLUMNS = (
    "model",
    "arm",
    "view",
    "s_min",
    "a_max",
    "reprompts",
    "useful",
    "harmful",
    "neutral",
    "net",
)
CHOICE_COLUMNS = (
    "model",
    "arm",
    "view",
    "s_min",
    "a_max",
    "reprompts",
    "net",
    "baseline_net",
    "go",
    "scored_candidates",
)


def is_vit(arm: str) -> bool:
    """Tell whether an arm uses the ViT tracker (and so has a score gate to tune).

    Args:
        arm: Box-source arm name.

    Returns:
        True for the arms whose name starts with "vit_".
    """
    return arm.startswith("vit_")


def share(numerator: int, denominator: int) -> float:
    """Divide two counts, returning NaN when the denominator is zero.

    Args:
        numerator: Count of cases that meet a condition.
        denominator: Count of all cases.

    Returns:
        The fraction, or NaN if there are no cases.
    """
    return numerator / denominator if denominator else math.nan


def exact_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """Candidates whose own frame is annotated (scored frame equals the candidate frame)."""
    scored = pd.to_numeric(candidates["scored_frame"], errors="coerce")
    frame = pd.to_numeric(candidates["frame_index"], errors="coerce")
    return candidates[scored.notna() & (scored == frame)]


def view_rows(candidates: pd.DataFrame, view: str) -> pd.DataFrame:
    """Select the candidates that belong to one view of the pilot table.

    Args:
        candidates: Candidate rows of one arm.
        view: One of VIEWS: "exact", "first_of_object" or "all".

    Returns:
        The rows of that view ("all" returns every row).
    """
    if view == "exact":
        return exact_rows(candidates)
    if view == "first_of_object":
        return candidates[candidates["first_of_object"].astype(str) == "True"]
    return candidates


def reprompt_mask(table: pd.DataFrame, s_min: float | None, a_max: float | None) -> pd.Series[bool]:
    """Candidates that would be re-prompted: a box, the gate passes, the box check passes."""
    ok = table["status"] == "ok"
    passes = ok & (table["box_check_ok"].astype(str) == "True")
    if s_min is not None:
        passes &= table["score"].notna() & (table["score"] >= s_min)
    if a_max is not None:
        passes &= ~(table["iou_sam2"].fillna(0.0) >= a_max)
    return passes


def outcomes(table: pd.DataFrame, s_min: float | None, a_max: float | None) -> tuple[int, int, int]:
    """(useful, harmful, neutral) over the scored candidates of one arm."""
    # Candidates without ground truth cannot be judged, so they are left out.
    scored = table[table["cls"] != "unscored"]
    # Re-prompts are the candidates the gate and box check let through.
    sent = scored[reprompt_mask(scored, s_min, a_max)]
    good = sent["iou_gt"].fillna(0.0) >= GOOD_IOU
    useful = int(((sent["cls"] == "real_failure") & good).sum())
    harmful = int(((sent["cls"] == "not_visible") | ~good).sum())
    return useful, harmful, len(sent) - useful - harmful


def pilot_rows(
    model: str, candidates: pd.DataFrame, timing: pd.DataFrame
) -> list[dict[str, object]]:
    """Build the stage 1 table rows (one per arm and view) for one model.

    Args:
        model: "tiny" or "large".
        candidates: Candidate rows of all arms from the pilot.
        timing: Per-video tracker call counts and seconds from the pilot.

    Returns:
        One dictionary per (arm, view) with the PILOT_COLUMNS: counts by class, box status counts,
        box-check passes, shares of good boxes and of boxes that copy SAM 2, and milliseconds per
        tracker call.
    """
    rows: list[dict[str, object]] = []
    for arm in ARM_ORDER:
        # Tracker time per arm comes from the timing table, summed over videos.
        seconds = float(timing.loc[timing["arm"] == arm, "seconds"].sum())
        calls = int(timing.loc[timing["arm"] == arm, "calls"].sum())
        # Each arm is summarised once per view of the candidates.
        for view in VIEWS:
            part = view_rows(candidates[candidates["arm"] == arm], view)
            # Split the candidates by ground-truth class to measure how good the boxes are.
            real = part[part["cls"] == "real_failure"]
            alarms = part[part["cls"] == "false_alarm"]
            check = part["box_check_ok"].astype(str) == "True"
            rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "view": view,
                    "candidates": len(part),
                    **{c: int((part["cls"] == c).sum()) for c in _CLASSES},
                    **{f"status_{s}": int((part["status"] == s).sum()) for s in _STATUSES},
                    "box_check_pass": int(check.sum()),
                    "real_good_box_share": share(
                        int((real["iou_gt"].fillna(0.0) >= GOOD_IOU).sum()), len(real)
                    ),
                    "real_copies_sam2_share": share(
                        int((real["iou_sam2"].fillna(0.0) >= GOOD_IOU).sum()), len(real)
                    ),
                    "false_alarm_copies_sam2_share": share(
                        int((alarms["iou_sam2"].fillna(0.0) >= GOOD_IOU).sum()), len(alarms)
                    ),
                    "ms_per_frame": 1000 * seconds / calls if calls else math.nan,
                }
            )
    return rows


_CLASSES = ("real_failure", "false_alarm", "not_visible", "unscored")
_STATUSES = ("ok", "lost", "not_initialised")


def grid_rows(model: str, candidates: pd.DataFrame) -> list[dict[str, object]]:
    """Build the stage 2 gate grid (net re-prompt outcomes per gate setting) for one model.

    Args:
        model: "tiny" or "large".
        candidates: Candidate rows of all arms from the pilot.

    Returns:
        One dictionary per arm, grid view and gate setting with the GRID_COLUMNS. The score gate
        (s_min) is only varied for ViT arms; "off" marks a disabled gate.
    """
    rows: list[dict[str, object]] = []
    # Every arm and view is scored over the whole gate grid.
    for arm, view in itertools.product(ARM_ORDER, GRID_VIEWS):
        table = view_rows(candidates[candidates["arm"] == arm], view)
        # Only ViT arms have a confidence score to gate on; other arms use no score gate.
        for s_min in S_MIN_GRID if is_vit(arm) else (None,):
            for a_max in A_MAX_GRID:
                useful, harmful, neutral = outcomes(table, s_min, a_max)
                rows.append(
                    {
                        "model": model,
                        "arm": arm,
                        "view": view,
                        "s_min": "off" if s_min is None else s_min,
                        "a_max": "off" if a_max is None else a_max,
                        "reprompts": useful + harmful + neutral,
                        "useful": useful,
                        "harmful": harmful,
                        "neutral": neutral,
                        "net": useful - harmful,
                    }
                )
    return rows


def _restriction(row: pd.Series[Any]) -> int:
    """Count how many gates a grid row switches on (0, 1 or 2).

    Args:
        row: A grid row with "s_min" and "a_max" fields, "off" meaning disabled.

    Returns:
        The number of gates that are not "off".
    """
    return int(row["s_min"] != "off") + int(row["a_max"] != "off")


def choose(model: str, grid: pd.DataFrame, view: str = CHOICE_VIEW) -> dict[str, object]:
    """The grid row with the highest net; ties by re-prompts, arm order, then fewer gates."""
    # Keep the grid rows of this model and view; the choice uses one view only.
    table = grid[(grid["model"] == model) & (grid["view"] == view)].copy()
    # Rank arms by simplicity (ARM_ORDER) and count gates, for tie-breaking.
    table["arm_rank"] = table["arm"].map({a: i for i, a in enumerate(ARM_ORDER)})
    table["restriction"] = table.apply(_restriction, axis=1)
    # Highest net first, then fewer re-prompts, simpler arm, fewer gates.
    best = table.sort_values(
        ["net", "reprompts", "arm_rank", "restriction"], ascending=[False, True, True, True]
    ).iloc[0]
    # Reference: the CSRT reinit arm with no gate, which a new box source must beat.
    base = table[
        (table["arm"] == BASELINE_ARM) & (table["s_min"] == "off") & (table["a_max"] == "off")
    ]["net"].iloc[0]
    net = int(best["net"])
    return {
        "model": model,
        "arm": best["arm"],
        "view": view,
        "s_min": best["s_min"],
        "a_max": best["a_max"],
        "reprompts": int(best["reprompts"]),
        "net": net,
        "baseline_net": int(base),
        "go": bool(net > 0 and net > int(base)),
    }


def scored_candidates(candidates: pd.DataFrame, arm: object) -> int:
    """Count the exact-view candidates of an arm that have a ground-truth class.

    Args:
        candidates: Candidate rows of all arms.
        arm: Arm name to count for.

    Returns:
        Number of candidates whose own frame is annotated and that are not "unscored".
    """
    part = exact_rows(candidates[candidates["arm"] == arm])
    return int((part["cls"] != "unscored").sum())


def gate_yaml(choice: dict[str, object]) -> str:
    """Render the gate config text for a choice.

    Args:
        choice: A choice row with the "s_min" and "a_max" settings ("off" means disabled).

    Returns:
        YAML text of kind box_gate where a disabled gate is written as .inf.
    """

    def value(v: object) -> str:
        """Format one gate setting: ".inf" when off, otherwise the number."""
        return ".inf" if v == "off" else repr(float(v))  # type: ignore[arg-type]

    return f"kind: box_gate\ns_min: {value(choice['s_min'])}\na_max: {value(choice['a_max'])}\n"


def write_gate_file(config_dir: Path, choice: dict[str, object]) -> Path:
    """Write the gate config file for one model's choice.

    Args:
        config_dir: Folder for the recovery configs.
        choice: The choice row; "model" names the file.

    Returns:
        The path of the file written.
    """
    path = config_dir / f"box_gate_{choice['model']}.yaml"
    path.write_text(gate_yaml(choice))
    return path


def inputs_for(results_dir: Path, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the merged pilot candidates and timing tables of one model.

    Args:
        results_dir: Folder that holds the merged pilot CSVs.
        model: "tiny" or "large".

    Returns:
        The candidates table and the timing table.
    """
    cand = results_dir / f"e13_pilot_candidates_{model}_{SPLIT}.csv"
    timing = results_dir / f"e13_pilot_timing_{model}_{SPLIT}.csv"
    return pd.read_csv(cand, dtype={"video": str}), pd.read_csv(timing, dtype={"video": str})


def main(argv: Sequence[str] | None = None) -> int:
    """Make the stage 1 table, the gate grid and the choice per model (command line).

    Reads the merged pilot candidate and timing CSVs in --results-dir for each model that has
    them. Writes e13_pilot.csv, e13_gate_grid.csv and e13_choice.csv there, writes a gate file
    into --config-dir only for a model whose choice is a go, and prints a verdict line per model.

    Args:
        argv: Command-line arguments; None means sys.argv.

    Returns:
        0 on success.

    Raises:
        SystemExit: If no merged pilot candidates are found.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    args = parser.parse_args(argv)
    # Process only the models whose merged pilot files exist.
    models = [
        m for m in MODELS if (args.results_dir / f"e13_pilot_candidates_{m}_{SPLIT}.csv").is_file()
    ]
    if not models:
        raise SystemExit(f"no merged pilot candidates in {args.results_dir}")
    pilot_all: list[dict[str, object]] = []
    grid_all: list[dict[str, object]] = []
    by_model: dict[str, pd.DataFrame] = {}
    for model in models:
        candidates, timing = inputs_for(args.results_dir, model)
        pilot_all += pilot_rows(model, candidates, timing)
        grid_all += grid_rows(model, candidates)
        by_model[model] = candidates
    # Pick one arm and gate per model from the exact view.
    grid = pd.DataFrame(grid_all, columns=list(GRID_COLUMNS))
    choices = [choose(m, grid) for m in models]
    for choice in choices:
        table = by_model[str(choice["model"])]
        choice["scored_candidates"] = scored_candidates(table, choice["arm"])
    # Write the three result tables.
    pd.DataFrame(pilot_all, columns=list(PILOT_COLUMNS)).to_csv(
        args.results_dir / "e13_pilot.csv", index=False
    )
    grid.to_csv(args.results_dir / "e13_gate_grid.csv", index=False)
    pd.DataFrame(choices, columns=list(CHOICE_COLUMNS)).to_csv(
        args.results_dir / "e13_choice.csv", index=False
    )
    for choice in choices:
        # Report each choice; the gate file is written only for a go.
        verdict = "GO" if choice["go"] else "NO-GO"
        sys.stdout.write(
            f"{choice['model']}: {verdict} arm={choice['arm']} s_min={choice['s_min']} "
            f"a_max={choice['a_max']} net={choice['net']} (csrt_reinit ungated "
            f"{choice['baseline_net']}) exact_scored_candidates={choice['scored_candidates']}\n"
        )
        # For information only: what the all view would have chosen. It does not change the choice.
        info = choose(str(choice["model"]), grid, "all")
        sys.stdout.write(
            f"  info only, view all would choose arm={info['arm']} s_min={info['s_min']} "
            f"a_max={info['a_max']} net={info['net']} go={info['go']}\n"
        )
        if choice["go"]:
            sys.stdout.write(f"  wrote {write_gate_file(args.config_dir, choice)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
