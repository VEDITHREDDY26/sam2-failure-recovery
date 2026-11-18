"""Choose the verification thresholds on SA-V val and write the frozen config.

    python -m experiments.verify.tune [--results-dir results] [--models tiny large]
        [--config-out configs/verify/verify.yaml] [--results-out results/verify_sav_val.csv]

Reads results/verify_candidates_<model>_sav_val.csv (the labelled candidates of the accept-all
run). Grid (fixed): s_min in {off, -4, -2, 0, 2, 4, 6, 8, 10, 12}, r_max in {off, 1.5, 2, 3, 4},
b_min in {off, 0.3, 0.5, 0.7}; 200 settings per model, rule: accept iff cand_score >= s_min and
area_ratio <= r_max and box_iou >= b_min. Objective: the sum of gain over accepted candidates
(unscored candidates add nothing). Ties: fewer accepted candidates, then the stricter setting
(larger total strictness rank, then later in grid order). If no setting has a sum above 0, the
frozen setting is "accept nothing". Writes all 200 rows per model with the chosen row marked
(for accept nothing an extra row named accept_nothing is marked instead) and refuses to
overwrite an existing config. Only SA-V val candidates are read; test splits are refused.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.verify.draw_subset import OUT as SUBSET_FILE
from experiments.verify.draw_subset import read_names
from experiments.verify.label_candidates import labels_name
from src.recovery.verify import MODELS, VERIFY_PATH, VerifyThresholds

OFF = "off"
S_GRID = (None, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
R_GRID = (None, 1.5, 2.0, 3.0, 4.0)
B_GRID = (None, 0.3, 0.5, 0.7)
SPLIT = "sav_val"
RESULT_COLUMNS = (
    "model",
    "setting",
    "s_min",
    "r_max",
    "b_min",
    "candidates",
    "scored",
    "accepted",
    "rejected",
    "accepted_scored",
    "gain_sum",
    "gain_positive",
    "gain_negative",
    "chosen",
)
ACCEPT_NOTHING_NAME = "accept_nothing"


@dataclass(frozen=True)
class Setting:
    """One setting of the verification rule.

    Attributes:
        index: Position in the grid; the order of the grid is the tie-break order.
        s_min: Minimum candidate score, or None for no check.
        r_max: Maximum area ratio, or None for no check.
        b_min: Minimum box IoU, or None for no check.
    """

    index: int
    s_min: float | None
    r_max: float | None
    b_min: float | None

    @property
    def strictness(self) -> int:
        """Rank of how strict the setting is; larger means stricter (used to break ties)."""
        s = 0 if self.s_min is None else S_GRID.index(self.s_min)
        r = 0 if self.r_max is None else len(R_GRID) - R_GRID.index(self.r_max)
        b = 0 if self.b_min is None else B_GRID.index(self.b_min)
        return s + r + b

    def thresholds(self) -> VerifyThresholds:
        """The setting as thresholds; an unused check becomes an infinite bound."""
        return VerifyThresholds(
            -math.inf if self.s_min is None else self.s_min,
            math.inf if self.r_max is None else self.r_max,
            -math.inf if self.b_min is None else self.b_min,
            False,
        )


def grid() -> list[Setting]:
    """All settings of the fixed grid, in the order of s_min, then r_max, then b_min."""
    combos = itertools.product(S_GRID, R_GRID, B_GRID)
    return [Setting(i, s, r, b) for i, (s, r, b) in enumerate(combos)]


def accept_mask(table: pd.DataFrame, rule: VerifyThresholds) -> np.ndarray:
    """Tell which candidates a rule accepts.

    Args:
        table: Candidates with cand_score, area_ratio and box_iou columns.
        rule: Thresholds; a non-finite threshold skips its check.

    Returns:
        Boolean array, True where the candidate passes every active check.
    """
    ok = np.ones(len(table), dtype=bool)
    if math.isfinite(rule.s_min):
        ok &= (table["cand_score"].to_numpy(float) >= rule.s_min).astype(bool)
    if math.isfinite(rule.r_max):
        ok &= (table["area_ratio"].to_numpy(float) <= rule.r_max).astype(bool)
    if math.isfinite(rule.b_min):
        ok &= (table["box_iou"].to_numpy(float) >= rule.b_min).astype(bool)
    return ok


def _cell(value: float | None) -> object:
    """Table cell of a threshold: "off" when the check is not used."""
    return OFF if value is None else value


def score_setting(table: pd.DataFrame, model: str, setting: Setting) -> dict[str, object]:
    """Score one setting on the labelled candidates.

    Args:
        table: Labelled candidates of one model.
        model: SAM 2 size.
        setting: Setting to apply.

    Returns:
        A row with the accepted and rejected counts and the summed gain of the accepted candidates
        that have a gain, with their positive and negative counts; chosen is False.
    """
    # Apply the rule, then add the gains of the accepted candidates that were scored.
    accepted = accept_mask(table, setting.thresholds())
    gain = pd.to_numeric(table["gain"], errors="coerce").to_numpy(float)
    kept = gain[accepted]
    kept = kept[~np.isnan(kept)]
    return {
        "model": model,
        "setting": setting.index,
        "s_min": _cell(setting.s_min),
        "r_max": _cell(setting.r_max),
        "b_min": _cell(setting.b_min),
        "candidates": len(table),
        "scored": int(np.count_nonzero(~np.isnan(gain))),
        "accepted": int(accepted.sum()),
        "rejected": int(len(table) - accepted.sum()),
        "accepted_scored": len(kept),
        "gain_sum": float(kept.sum()),
        "gain_positive": int((kept > 0).sum()),
        "gain_negative": int((kept < 0).sum()),
        "chosen": False,
    }


def choose(rows: list[dict[str, object]], settings: list[Setting]) -> int | None:
    """Index of the best setting, or None when nothing beats accepting nothing."""
    best = max(
        range(len(rows)),
        key=lambda i: (
            rows[i]["gain_sum"],
            -int(rows[i]["accepted"]),  # type: ignore[call-overload]
            settings[i].strictness,
            i,
        ),
    )
    return best if float(rows[best]["gain_sum"]) > 0 else None  # type: ignore[arg-type]


def tune_model(table: pd.DataFrame, model: str) -> tuple[pd.DataFrame, VerifyThresholds]:
    """Pick the setting of one model.

    Args:
        table: Labelled candidates of the model.
        model: SAM 2 size.

    Returns:
        The result rows (every setting, the chosen one marked) and the chosen thresholds. When no
        setting gains, an extra accept_nothing row is marked and the thresholds accept nothing.
    """
    # Score every setting of the grid.
    settings = grid()
    rows = [score_setting(table, model, s) for s in settings]
    best = choose(rows, settings)
    # No setting gains: add an accept_nothing row and freeze that choice.
    if best is None:
        rows.append(
            {
                "model": model,
                "setting": ACCEPT_NOTHING_NAME,
                "s_min": "",
                "r_max": "",
                "b_min": "",
                "candidates": len(table),
                "scored": rows[0]["scored"],
                "accepted": 0,
                "rejected": len(table),
                "accepted_scored": 0,
                "gain_sum": 0.0,
                "gain_positive": 0,
                "gain_negative": 0,
                "chosen": True,
            }
        )
        return pd.DataFrame(rows, columns=list(RESULT_COLUMNS)), VerifyThresholds(
            -math.inf, math.inf, -math.inf, True
        )
    # Mark the winner and return its thresholds.
    rows[best]["chosen"] = True
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS)), settings[best].thresholds()


def read_candidates(results_dir: Path, model: str, split: str) -> pd.DataFrame:
    """Read the labelled candidates of a model from SA-V val.

    Args:
        results_dir: Folder holding the labelled candidate files.
        model: SAM 2 size.
        split: Must be sav_val.

    Returns:
        The candidate table.

    Raises:
        SystemExit: If the split is not sav_val, the file is missing or columns are missing.
    """
    if split != SPLIT:
        raise SystemExit(f"split '{split}' refused: the rule is tuned on {SPLIT} only")
    path = results_dir / labels_name(model)
    if not path.is_file():
        raise SystemExit(f"missing labelled candidates: {path}")
    # Keep video as text so names are not turned into numbers.
    table = pd.read_csv(path, dtype={"video": str})
    needed = {"video", "cand_score", "area_ratio", "box_iou", "gain"}
    if not needed <= set(table.columns):
        raise SystemExit(f"{path}: columns {sorted(needed - set(table.columns))} missing")
    return table


def check_subset(table: pd.DataFrame, model: str, subset_file: Path) -> None:
    """Check that the Large candidates only come from the tuning subset.

    Args:
        table: Labelled candidates of the model; nothing is checked for Tiny.
        model: SAM 2 size.
        subset_file: File listing the allowed videos.

    Raises:
        SystemExit: If a Large candidate belongs to a video outside the subset.
    """
    if model != "large":
        return
    outside = sorted(set(table["video"]) - set(read_names(subset_file)))
    if outside:
        raise SystemExit(f"large candidates outside the tuning subset: {outside[:5]}")


def thresholds_yaml(chosen: dict[str, VerifyThresholds]) -> str:
    """Render the chosen thresholds of all models as the text of the frozen config.

    Args:
        chosen: Thresholds per model.

    Returns:
        YAML text with kind verify and one block per model.
    """
    body: dict[str, object] = {"kind": "verify"}
    for model in MODELS:
        t = chosen[model]
        body[model] = {
            "s_min": t.s_min,
            "r_max": t.r_max,
            "b_min": t.b_min,
            "accept_nothing": t.accept_nothing,
        }
    return yaml.safe_dump(body, sort_keys=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: tune the verification rule on SA-V val and freeze it.

    Reads the labelled candidates of both models in --results-dir. Writes the grid results to
    --results-out and the chosen thresholds to --config-out, and prints one line per model.

    Args:
        argv: Command line arguments; None means sys.argv.

    Returns:
        Exit code 0 on success.

    Raises:
        SystemExit: If the split is not sav_val, not both models are given, the config file
            already exists (the setting is frozen once) or an input is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--config-out", type=Path, default=VERIFY_PATH)
    parser.add_argument("--results-out", type=Path, default=Path("results/verify_sav_val.csv"))
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    args = parser.parse_args(argv)
    # Guard the one-time freeze: both models together, no overwrite.
    if sorted(args.models) != sorted(MODELS):
        raise SystemExit(f"both models are tuned together: {list(MODELS)}")
    if args.config_out.exists():
        raise SystemExit(f"{args.config_out} exists: the verification setting is frozen once")
    tables: list[pd.DataFrame] = []
    chosen: dict[str, VerifyThresholds] = {}
    # Tune each model on its labelled candidates.
    for model in MODELS:
        candidates = read_candidates(args.results_dir, model, args.split)
        check_subset(candidates, model, args.subset_file)
        table, chosen[model] = tune_model(candidates, model)
        tables.append(table)
        pick = table[table["chosen"]].iloc[0]
        sys.stdout.write(
            f"{model}: {len(candidates)} candidates, chosen setting {pick['setting']} "
            f"(s_min {pick['s_min']}, r_max {pick['r_max']}, b_min {pick['b_min']}), "
            f"accepted {pick['accepted']}, gain sum {pick['gain_sum']:.3f}\n"
        )
    # Write the full grid results, then the frozen config.
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(args.results_out, index=False)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(thresholds_yaml(chosen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
