"""Main results table with the verified-recovery rows added when their result files exist.

    python -m experiments.verify.main_table [--results-dir results]
        [--csv-out results/main_table.csv] [--tex-out results/main_table.tex]

Builds the same table as src.eval.tables (same inputs, bootstrap and layout) and adds a row
"SAM 2 <model> + verified fix" after the "+ fix" row of each model whose
fix_verified_rules_csrt_<model> score and timing files exist for both test sets. With no verified
files the output is identical to src.eval.tables. A model with only some verified files is an
error, so a partly run table is never written.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from src.eval import tables
from src.eval.tables import MethodSpec, TableError

MODELS = ("tiny", "large")


def verified_name(model: str) -> str:
    """Method name of the verified run for a model."""
    return f"fix_verified_rules_csrt_{model}"


def _verified_files(results_dir: Path, model: str) -> list[Path]:
    """List the score and timing files of the verified run for every test set."""
    name = verified_name(model)
    return [
        results_dir / f"{name}{middle}_{split}.csv"
        for split in tables.TEST_SETS
        for middle in ("", "_timing")
    ]


def present_models(results_dir: Path) -> list[str]:
    """Find the models whose verified result files all exist.

    Args:
        results_dir: Folder holding the result files.

    Returns:
        Models with all files present.

    Raises:
        TableError: If a model has only some of its files.
    """
    found: list[str] = []
    # A model gets a verified row only if all its files exist; some missing is an error.
    for model in MODELS:
        files = _verified_files(results_dir, model)
        have = [f.is_file() for f in files]
        if any(have) and not all(have):
            missing = [f.name for f, ok in zip(files, have, strict=True) if not ok]
            raise TableError(f"verified {model} files incomplete, missing {missing}")
        if all(have):
            found.append(model)
    return found


def methods_with_verified(models: Sequence[str]) -> tuple[MethodSpec, ...]:
    """Extend the table method list with a verified row after each model's fix row.

    Args:
        models: Models that have verified results.

    Returns:
        The method specs, in table order.
    """
    out: list[MethodSpec] = []
    # Insert the verified row right after the row of the fix method of the same model.
    for spec in tables.METHODS:
        out.append(spec)
        model = spec.name.removeprefix("fix_rules_csrt_")
        if spec.name.startswith("fix_rules_csrt_") and model in models:
            out.append(
                MethodSpec(
                    verified_name(model),
                    f"SAM 2 {model.capitalize()} + verified fix",
                    verified_name(model),
                    f"sam2_{model}",
                )
            )
    return tuple(out)


@contextmanager
def _methods(specs: tuple[MethodSpec, ...]) -> Iterator[None]:
    """Temporarily replace the method list of the main table module, then restore it."""
    original = tables.METHODS
    tables.METHODS = specs
    try:
        yield
    finally:
        tables.METHODS = original


def build(results_dir: Path, csv_out: Path, tex_out: Path) -> list[str]:
    """Write the main table (CSV and LaTeX) with the verified rows included.

    Args:
        results_dir: Folder holding the result files.
        csv_out: Output CSV path.
        tex_out: Output LaTeX path.

    Returns:
        The models whose verified rows were added.

    Raises:
        TableError: If the verified files of a model are incomplete.
    """
    models = present_models(results_dir)
    # Build the main table with the longer method list, then write CSV and LaTeX.
    with _methods(methods_with_verified(models)):
        rows = tables.build_table(results_dir)
        tables.write_csv(csv_out, rows)
        tables.write_tex(tex_out, tables.read_table(csv_out))
    return models


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point: write the main table with verified rows.

    Reads the result files in --results-dir and writes --csv-out and --tex-out. Prints which
    models got a verified row. Incomplete verified files raise TableError (traceback, non-zero
    exit).

    Args:
        argv: Command line arguments; None means sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results-dir", default=tables.DEFAULT_RESULTS_DIR)
    parser.add_argument("--csv-out", default=tables.DEFAULT_CSV_OUT)
    parser.add_argument("--tex-out", default=tables.DEFAULT_TEX_OUT)
    args = parser.parse_args(argv)
    models = build(Path(args.results_dir), Path(args.csv_out), Path(args.tex_out))
    print(f"main table written; verified rows for: {models or 'none'}")


if __name__ == "__main__":
    main()
