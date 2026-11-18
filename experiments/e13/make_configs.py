"""Write the E13 test-run configs from the frozen choice of box source and gate.

    python -m experiments.e13.make_configs [--choice results/e13_choice.csv] [--list]

Reads results/e13_choice.csv (model, arm, s_min, a_max, net, go; written by experiments.e13.choose)
and, for every model with go=True, writes the gated run config (fix_gated_rules_<arm>_<model>) that
points at configs/recovery/box_gate_<model>.yaml, and for Tiny only the ungated config
(fix_rules_<arm>_tiny). For the CSRT arms the ungated run already exists (E2 reinit, E8 continuous)
and no config is written; the script says which one stands for it. Nothing is written when the file
is missing, no model is a go, a gate file is missing, or a config exists with different content.
--list writes nothing and prints one line per run, "<model> <a|b> <config path> <new|existing>",
in the order the runs are submitted ("existing" = already run as E2 / E8, not run again).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.recovery.config import CONFIG_DIR, ROOT

CHOICE_PATH = ROOT / "results" / "e13_choice.csv"
ARMS = ("motion", "csrt_reinit", "csrt_continuous", "vit_reinit", "vit_continuous", "vit_anchored")
EXISTING_UNGATED = {
    "csrt_reinit": "rules_csrt_{model}.yaml",
    "csrt_continuous": "rules_csrt_continuous_{model}.yaml",
}
EXISTING_METHODS = {
    "csrt_reinit": "fix_rules_csrt_{model}",
    "csrt_continuous": "fix_continuous_rules_csrt_{model}",
}
UNGATED_MODELS = ("tiny",)


def gate_file_name(model: str) -> str:
    """Name of the gate config file for one model size.

    Args:
        model: "tiny" or "large".

    Returns:
        File name such as box_gate_tiny.yaml.
    """
    return f"box_gate_{model}.yaml"


def gated_method(arm: str, model: str) -> str:
    """Method name of the gated run for an arm and model.

    Args:
        arm: Box-source arm name, one of ARMS.
        model: "tiny" or "large".

    Returns:
        The name used for the run's results, e.g. fix_gated_rules_<arm>_<model>.
    """
    return f"fix_gated_rules_{arm}_{model}"


def ungated_method(arm: str, model: str) -> str:
    """Method name of variant (a): the E2/E8 run for a CSRT arm, else the new ungated run."""
    template = EXISTING_METHODS.get(arm)
    return template.format(model=model) if template else f"fix_rules_{arm}_{model}"


def has_own_ungated_config(arm: str) -> bool:
    """Tell whether the ungated run of an arm needs a new config.

    Args:
        arm: Box-source arm name.

    Returns:
        False for the CSRT arms whose ungated run already exists (E2 and E8), True otherwise.
    """
    return arm not in EXISTING_UNGATED


def config_text(method: str, arm: str, gate_config: str | None) -> str:
    """Render the text of one run config file.

    Args:
        method: Method name written into the config.
        arm: Box-source arm name; "csrt_*" arms use the CSRT source, other arms name the source.
        gate_config: Repo-relative path of the gate file, or None for an ungated run.

    Returns:
        YAML text ending with a newline. Only the continuous CSRT arm sets a tracker mode.
    """
    # CSRT arms share one box source; the tracker mode is only set for the continuous arm.
    csrt = arm.startswith("csrt_")
    source = "csrt" if csrt else arm
    mode = arm.removeprefix("csrt_") if arm == "csrt_continuous" else ""
    kind = "recovery_run" if gate_config is None else "gated_recovery_run"
    # Config keys in file order; the gate key is added only for gated runs.
    lines = [
        f"kind: {kind}",
        f"method: {method}",
        "detector: rules",
        f"box_source: {source}",
        f'tracker_mode: "{mode}"',
        "disabled_rules: []",
    ]
    if gate_config is not None:
        lines.append(f"gate_config: {gate_config}")
    return "\n".join(lines) + "\n"


def planned_configs(model: str, arm: str) -> dict[str, str]:
    """File name -> text of the configs to write for one model's choice."""
    gate = f"configs/recovery/{gate_file_name(model)}"
    out = {
        f"rules_gated_{arm}_{model}.yaml": config_text(gated_method(arm, model), arm, gate),
    }
    if model in UNGATED_MODELS and has_own_ungated_config(arm):
        out[f"rules_{arm}_{model}.yaml"] = config_text(ungated_method(arm, model), arm, None)
    return out


def read_choice(path: Path) -> pd.DataFrame:
    """Load the choice table written by the choose step.

    Args:
        path: Path to results/e13_choice.csv.

    Returns:
        The table with the "go" column converted to booleans.

    Raises:
        SystemExit: If the file is missing or lacks one of the needed columns.
    """
    if not path.is_file():
        raise SystemExit(f"{path} is missing: run experiments.e13.choose first")
    table = pd.read_csv(path)
    needed = {"model", "arm", "s_min", "a_max", "net", "go"}
    if not needed <= set(table.columns):
        raise SystemExit(f"{path}: needs columns {sorted(needed)}")
    table["go"] = table["go"].astype(str) == "True"
    return table


def go_choices(path: Path) -> list[tuple[str, str]]:
    """(model, arm) for every model whose choice is a go."""
    table = read_choice(path)
    chosen = table[table["go"]]
    if chosen.empty:
        raise SystemExit(f"{path}: no model is a go; no test run is made, nothing written")
    unknown = sorted(set(chosen["arm"]) - set(ARMS))
    if unknown:
        raise SystemExit(f"{path}: unknown arm {unknown}")
    return [(str(r.model), str(r.arm)) for r in chosen.itertuples()]


def plan(choice: Path, config_dir: Path) -> dict[Path, str]:
    """Work out every config file to write for the models whose choice is a go.

    Args:
        choice: Path to the choice CSV.
        config_dir: Folder that holds the recovery configs and the gate files.

    Returns:
        Mapping from config path to the text to write.

    Raises:
        SystemExit: If no model is a go, an arm is unknown, or a gate file is missing.
    """
    files: dict[Path, str] = {}
    # Gate files must already exist (written by the choose step) before configs are planned.
    for model, arm in go_choices(choice):
        gate = config_dir / gate_file_name(model)
        if not gate.is_file():
            raise SystemExit(f"{gate} is missing: run experiments.e13.choose first")
        files.update({config_dir / n: t for n, t in planned_configs(model, arm).items()})
    return files


def run_list(choice: Path, config_dir: Path) -> list[tuple[str, str, Path, str]]:
    """(model, variant, config path, new | existing) for every test run, ungated first."""
    runs: list[tuple[str, str, Path, str]] = []
    for model, arm in go_choices(choice):
        if model in UNGATED_MODELS:
            if has_own_ungated_config(arm):
                runs.append((model, "a", config_dir / f"rules_{arm}_{model}.yaml", "new"))
            else:
                existing = config_dir / EXISTING_UNGATED[arm].format(model=model)
                runs.append((model, "a", existing, "existing"))
        runs.append((model, "b", config_dir / f"rules_gated_{arm}_{model}.yaml", "new"))
    return runs


def check_no_conflict(files: dict[Path, str]) -> None:
    """Refuse to overwrite an existing config whose content would change.

    Args:
        files: Mapping from config path to the text that would be written.

    Raises:
        SystemExit: If a file exists with different content.
    """
    for path, text in files.items():
        if path.is_file() and path.read_text() != text:
            raise SystemExit(f"refusing to overwrite {path}: it exists with different content")


def main(argv: Sequence[str] | None = None) -> int:
    """Write (or list) the E13 run configs for the chosen box source (command line).

    Reads the choice CSV and the gate files in --config-dir. Without --list it writes the new
    config files and prints one line per file written. With --list it writes nothing and prints
    one line per run.

    Args:
        argv: Command-line arguments; None means sys.argv.

    Returns:
        0 on success.

    Raises:
        SystemExit: If an input is missing, no model is a go, or a config would be overwritten
            with different content.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--choice", type=Path, default=CHOICE_PATH)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    files = plan(args.choice, args.config_dir)
    if args.list:
        for model, variant, path, state in run_list(args.choice, args.config_dir):
            sys.stdout.write(f"{model} {variant} {path} {state}\n")
        return 0
    # Write nothing unless every target is new or already identical.
    check_no_conflict(files)
    for path, text in files.items():
        path.write_text(text)
        sys.stdout.write(f"wrote {path}\n")
    # Tell the reader which existing run stands for variant (a) of a CSRT arm.
    for model, arm in go_choices(args.choice):
        if model in UNGATED_MODELS and not has_own_ungated_config(arm):
            existing = args.config_dir / EXISTING_UNGATED[arm].format(model=model)
            sys.stdout.write(f"{model}: variant (a) is the existing run {existing}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
