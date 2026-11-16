from pathlib import Path

import pytest

from experiments.verify import main_table
from src.eval import tables
from src.eval.tables import TableError
from tests.eval.test_tables import TINY, make_results, write_scores, write_timing


def add_verified(root: Path, model: str = "tiny") -> None:
    values = {k: v - 0.05 for k, v in TINY.items()}
    for split in tables.TEST_SETS:
        write_scores(root / f"fix_verified_rules_csrt_{model}_{split}.csv", values)
        write_timing(
            root / f"fix_verified_rules_csrt_{model}_timing_{split}.csv", ("gpu_name", "H100")
        )


def run(root: Path, name: str) -> tuple[str, str]:
    csv_out, tex_out = root / f"{name}.csv", root / f"{name}.tex"
    main_table.main(
        ["--results-dir", str(root), "--csv-out", str(csv_out), "--tex-out", str(tex_out)]
    )
    return csv_out.read_text(), tex_out.read_text()


def test_without_verified_files_the_output_is_the_standard_table(tmp_path: Path) -> None:
    root = make_results(tmp_path)
    csv_text, tex_text = run(root, "extended")
    tables.main(
        [
            "--results-dir",
            str(root),
            "--csv-out",
            str(root / "plain.csv"),
            "--tex-out",
            str(root / "plain.tex"),
        ]
    )
    assert csv_text == (root / "plain.csv").read_text()
    assert tex_text == (root / "plain.tex").read_text()


def test_verified_rows_follow_the_fix_row_of_their_model(tmp_path: Path) -> None:
    root = make_results(tmp_path)
    add_verified(root)
    csv_text, tex_text = run(root, "extended")
    methods = [line.split(",")[0] for line in csv_text.splitlines()[1:]]
    assert methods[: len(methods) // 2] == [
        "baseline_csrt_grabcut",
        "sam2_tiny",
        "fix_rules_csrt_tiny",
        "fix_verified_rules_csrt_tiny",
        "sam2_large",
        "fix_rules_csrt_large",
    ]
    assert "SAM 2 Tiny + verified fix" in tex_text
    assert len(tables.METHODS) == 5


def test_partial_verified_files_are_refused(tmp_path: Path) -> None:
    root = make_results(tmp_path)
    add_verified(root)
    (root / "fix_verified_rules_csrt_tiny_timing_sav_test.csv").unlink()
    with pytest.raises(TableError, match="incomplete"):
        run(root, "extended")
    assert not (root / "extended.csv").exists()
