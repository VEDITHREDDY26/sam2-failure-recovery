from pathlib import Path

import pandas as pd
import pytest

from experiments.e8 import figure, table, wrong_reprompts
from src.eval.rows import ScoreRow, write_scores_csv
from src.recovery.events import EVENT_COLUMNS
from tests.eval.sam2_trees import build_world, run_all
from tests.experiments.e2_trees import blank_object, make_online_run, write_events

CONFIG = "configs/recovery/rules_csrt_tiny.yaml"
CONTINUOUS = "configs/recovery/rules_csrt_continuous_tiny.yaml"
METHOD = "fix_rules_csrt_tiny"
SPLIT = "davis_val"


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    run_all(root, subset, models=("tiny",), splits=("davis_val", "sav_test"))
    return root / "outputs", subset


def run_helper(subset: Path, results: Path, *extra: str, config: str = CONFIG) -> int:
    args = ["--config", config, "--subset-file", str(subset), "--results-dir", str(results)]
    return wrong_reprompts.main([*args, *extra])


def test_wrong_reprompt_means_iou_below_half_at_the_reprompted_frame(
    world: tuple[Path, Path], tmp_path: Path
) -> None:
    out, subset = world
    make_online_run(out)
    blank_object(out, METHOD, SPLIT, "zebra", "00003", 1)
    write_events(
        out,
        METHOD,
        SPLIT,
        "zebra",
        [(1, 3, "reprompted"), (2, 5, "reprompted"), (1, 9, "rejected")],
    )
    results = tmp_path / "results"
    assert run_helper(subset, results, "--splits", SPLIT) == 0
    got = pd.read_csv(results / f"{METHOD}_{SPLIT}_reprompts.csv")
    assert list(got.columns) == list(wrong_reprompts.REPROMPT_COLUMNS)
    assert got["frame_index"].tolist() == [3, 5]
    assert got["wrong"].tolist() == [True, False]
    assert got["iou_after"].iloc[0] < 0.5 <= got["iou_after"].iloc[1]


def test_sav_val_is_refused(world: tuple[Path, Path], tmp_path: Path) -> None:
    _, subset = world
    with pytest.raises(SystemExit, match="refused"):
        run_helper(subset, tmp_path / "results", "--splits", "sav_val")
    assert not (tmp_path / "results").exists()


def test_alone_method_is_refused(world: tuple[Path, Path], tmp_path: Path) -> None:
    _, subset = world
    config = tmp_path / "sam2_tiny.yaml"
    config.write_text("kind: recovery_run\nmethod: sam2_tiny\ndetector: rules\nbox_source: csrt\n")
    with pytest.raises(ValueError, match="SAM 2 alone"):
        run_helper(subset, tmp_path / "results", config=str(config))


def scores(values: dict[tuple[str, int], float]) -> list[ScoreRow]:
    return [ScoreRow(v, o, j, j) for (v, o), j in values.items()]


def write_inputs(results: Path, wrong: dict[str, list[object]]) -> None:
    base = {("a", 1): 0.5, ("b", 1): 0.7}
    write_scores_csv(results / f"sam2_tiny_{SPLIT}.csv", scores(base))
    for method, gain in (("fix_rules_csrt_tiny", 0.1), ("fix_continuous_rules_csrt_tiny", 0.2)):
        name = f"{method}_{SPLIT}"
        write_scores_csv(results / f"{name}.csv", scores({k: v + gain for k, v in base.items()}))
        actions = ["reprompted"] * len(wrong[method]) + ["rejected"]
        rows = [("a", 1, i, "score", a, "", "", "", "", "", 0) for i, a in enumerate(actions)]
        pd.DataFrame(rows, columns=list(EVENT_COLUMNS)).to_csv(
            results / f"events_{name}.csv", index=False
        )
        data = [("a", 1, i, i, 0.4, w) for i, w in enumerate(wrong[method])]
        pd.DataFrame(data, columns=list(wrong_reprompts.REPROMPT_COLUMNS)).to_csv(
            results / f"{name}_reprompts.csv", index=False
        )


WRONG = {"fix_rules_csrt_tiny": [True, False], "fix_continuous_rules_csrt_tiny": [True, "", False]}


def test_table_counts_wrong_and_unscored_reprompts_per_variant(tmp_path: Path) -> None:
    write_inputs(tmp_path, WRONG)
    out = tmp_path / "e8.csv"
    table.main(["--results-dir", str(tmp_path), "--out", str(out), "--splits", SPLIT])
    got = pd.read_csv(out).set_index("variant")
    assert list(got.index) == ["sam2_alone", "reinit", "continuous"]
    assert got.loc["reinit", "wrong_reprompts"] == 1 and got.loc["reinit", "reprompts"] == 2
    assert got.loc["continuous", "wrong_reprompts"] == 1
    assert got.loc["continuous", "unscored_reprompts"] == 1
    assert got.loc["continuous", "delta_JF"] == pytest.approx(0.2)
    assert got.loc["reinit", "rejected"] == 1


def test_table_refuses_a_missing_reprompt_file(tmp_path: Path) -> None:
    write_inputs(tmp_path, WRONG)
    (tmp_path / f"fix_rules_csrt_tiny_{SPLIT}_reprompts.csv").unlink()
    with pytest.raises(table.E8Error, match="missing"):
        table.build_rows(tmp_path, [SPLIT])


def test_figure_is_built_from_the_table_only(tmp_path: Path) -> None:
    write_inputs(tmp_path, WRONG)
    out = tmp_path / "e8.csv"
    table.main(["--results-dir", str(tmp_path), "--out", str(out), "--splits", SPLIT])
    figure.main(["--table", str(out), "--figure-dir", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / figure.FIGURE_NAME).stat().st_size > 0


def test_continuous_variant_scores_through_the_e2_scorer(
    world: tuple[Path, Path], tmp_path: Path
) -> None:
    from experiments.e2 import score

    out, subset = world
    method = "fix_continuous_rules_csrt_tiny"
    make_online_run(out, method)
    results = tmp_path / "results"
    args = ["--config", CONTINUOUS, "--subset-file", str(subset), "--results-dir", str(results)]
    assert score.main([*args, "--splits", SPLIT]) == 0
    assert (results / f"{method}_{SPLIT}.csv").is_file()
