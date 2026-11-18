from pathlib import Path

import pandas as pd
import pytest

from experiments.e13 import figure, table
from src.eval.bootstrap import bootstrap_delta
from src.eval.rows import ScoreRow, write_scores_csv
from src.recovery.events import EVENT_COLUMNS

SPLITS = ("davis_val", "sav_test")
VIDEOS = ("a", "b", "c", "d")
BASE = {"a": 0.5, "b": 0.7, "c": 0.6, "d": 0.9}
GAIN = {
    "alone": dict.fromkeys(VIDEOS, 0.0),
    "fix_rules_csrt": {"a": -0.1, "b": -0.2, "c": 0.0, "d": 0.0},
    "fix_rules_vit_anchored": {"a": 0.1, "b": -0.1, "c": 0.0, "d": 0.05},
    "fix_gated_rules_vit_anchored": {"a": 0.1, "b": 0.0, "c": 0.0, "d": 0.05},
    "fix_rules_oracle_box": {"a": 0.2, "b": 0.1, "c": 0.0, "d": 0.0},
}


def method_of(kind: str, model: str) -> str:
    return f"sam2_{model}" if kind == "alone" else f"{kind}_{model}"


def write_method(results: Path, kind: str, model: str, split: str, reprompt_file: bool) -> None:
    method = method_of(kind, model)
    name = f"{method}_{split}"
    rows = [ScoreRow(v, 1, BASE[v] + GAIN[kind][v], BASE[v] + GAIN[kind][v]) for v in VIDEOS]
    rows.append(ScoreRow("a", 2, 0.4 + GAIN[kind]["a"], 0.4 + GAIN[kind]["a"]))
    write_scores_csv(results / f"{name}.csv", rows)
    timing = pd.DataFrame(
        {"video": list(VIDEOS), "frames": 10, "compute_s": 2.0, "gpu_name": "gpu"}
    )
    timing.to_csv(results / f"{method}_timing_{split}.csv", index=False)
    if kind == "alone":
        return
    actions = ["reprompted", "reprompted", "gated", "gated"]
    reasons = ["", "", "low_score", "agrees_with_sam2"]
    events = [
        ("a", 1, i, "score", x, r, "", "", "", "", 0)
        for i, (x, r) in enumerate(zip(actions, reasons, strict=True))
    ]
    pd.DataFrame(events, columns=list(EVENT_COLUMNS)).to_csv(
        results / f"events_{name}.csv", index=False
    )
    if reprompt_file:
        data = [("a", 1, 0, 1, 0.4, True), ("a", 1, 1, 1, 0.9, False)]
        pd.DataFrame(
            data,
            columns=["video", "object_id", "frame_index", "gt_frame_index", "iou_after", "wrong"],
        ).to_csv(results / f"{name}_reprompts.csv", index=False)


def build(tmp_path: Path, skip: tuple[str, ...] = ()) -> tuple[Path, Path]:
    results = tmp_path / "results"
    results.mkdir()
    choice = tmp_path / "choice.csv"
    choice.write_text(
        "model,arm,s_min,a_max,reprompts,net,baseline_net,go\n"
        "tiny,vit_anchored,0.4,off,5,3,1,True\nlarge,vit_anchored,0.4,off,5,3,1,True\n"
    )
    for split in SPLITS:
        for kind in GAIN:
            for model in ("tiny", "large"):
                if model == "large" and kind in ("fix_rules_vit_anchored", "fix_rules_oracle_box"):
                    continue
                if method_of(kind, model) in skip:
                    continue
                write_method(
                    results,
                    kind,
                    model,
                    split,
                    reprompt_file=not (model == "large" and kind == "fix_rules_csrt"),
                )
    return results, choice


def run(results: Path, choice: Path, out: Path) -> int:
    return table.main(
        [
            "--results-dir",
            str(results),
            "--choice",
            str(choice),
            "--out",
            str(out),
            "--tex-out",
            str(out.with_suffix(".tex")),
        ]
    )


def test_table_rows_values_and_columns(tmp_path: Path) -> None:
    results, choice = build(tmp_path)
    out = tmp_path / "e13.csv"
    assert run(results, choice, out) == 0
    got = pd.read_csv(out)
    assert list(got.columns) == list(table.TABLE_COLUMNS)
    tiny = got[(got["model"] == "tiny") & (got["split"] == "davis_val")].set_index("role")
    assert list(tiny.index) == ["alone", "e2", "variant_a", "variant_b", "oracle"]
    assert tiny.loc["variant_b", "method"] == "fix_gated_rules_vit_anchored_tiny"
    assert bool(tiny.loc["oracle", "oracle"]) and "oracle" in tiny.loc["oracle", "method"]
    assert not tiny.loc[["alone", "e2", "variant_a", "variant_b"], "oracle"].any()
    assert tiny.loc["alone", "objects"] == 5 and tiny.loc["alone", "videos"] == 4
    assert tiny.loc["alone", "fps"] == pytest.approx(5.0)
    assert tiny.loc["variant_b", "delta_JF"] == pytest.approx((0.1 + 0.1 + 0.05) / 5 + 0.0)
    assert tiny.loc["variant_b", "videos_improved"] == 2
    assert tiny.loc["variant_b", "videos_worse"] == 0
    assert tiny.loc["e2", "videos_worse"] == 2 and tiny.loc["e2", "videos_unchanged"] == 2
    assert tiny.loc["variant_a", "videos_worse"] == 1
    assert tiny.loc["variant_b", "reprompts"] == 2 and tiny.loc["variant_b", "wrong_reprompts"] == 1
    assert tiny.loc["variant_b", "reprompts_exact"] == 1
    assert tiny.loc["variant_b", "wrong_exact"] == 0
    assert tiny.loc["variant_b", "gated"] == 2
    assert tiny.loc["variant_b", "gated_low_score"] == 1
    assert tiny.loc["variant_b", "gated_agrees_with_sam2"] == 1
    assert pd.isna(tiny.loc["e2", "delta_vs_e2"]) and pd.isna(tiny.loc["alone", "delta_low"])


def test_intervals_are_the_seeded_paired_bootstrap(tmp_path: Path) -> None:
    results, choice = build(tmp_path)
    out = tmp_path / "e13.csv"
    run(results, choice, out)
    first = out.read_text()
    run(results, choice, out)
    assert out.read_text() == first
    got = pd.read_csv(out)
    row = got[
        (got["method"] == "fix_gated_rules_vit_anchored_tiny") & (got["split"] == "davis_val")
    ].iloc[0]
    new = {v: [BASE[v] + GAIN["fix_gated_rules_vit_anchored"][v]] for v in VIDEOS}
    new["a"].append(0.4 + 0.1)
    e2 = {v: [BASE[v] + GAIN["fix_rules_csrt"][v]] for v in VIDEOS}
    e2["a"].append(0.4 - 0.1)
    expected = bootstrap_delta(new, e2)
    assert row["delta_vs_e2"] == pytest.approx(expected.estimate)
    assert row["delta_vs_e2_low"] == pytest.approx(expected.low)
    assert row["delta_vs_e2_high"] == pytest.approx(expected.high)
    assert row["delta_low"] < row["delta_JF"] < row["delta_high"]


def test_large_has_alone_e2_and_gated_only(tmp_path: Path) -> None:
    results, choice = build(tmp_path)
    out = tmp_path / "e13.csv"
    run(results, choice, out)
    large = pd.read_csv(out).query("model == 'large' and split == 'sav_test'")
    assert list(large["role"]) == ["alone", "e2", "variant_b"]
    e2 = large.set_index("role").loc["e2"]
    assert pd.isna(e2["wrong_reprompts"]) and pd.isna(e2["wrong_exact"])
    assert pd.isna(e2["reprompts_exact"]) and e2["reprompts"] == 2


def test_csrt_reinit_choice_reuses_e2_as_variant_a(tmp_path: Path) -> None:
    results, choice = build(tmp_path)
    choice.write_text(
        "model,arm,s_min,a_max,reprompts,net,baseline_net,go\ntiny,csrt_reinit,off,0.5,5,3,1,True\n"
    )
    for split in SPLITS:
        write_method(results, "fix_gated_rules_vit_anchored", "tiny", split, True)
        gated = results / f"fix_gated_rules_vit_anchored_tiny_{split}.csv"
        gated.rename(results / f"fix_gated_rules_csrt_reinit_tiny_{split}.csv")
        for name in ("events_{}_{}.csv", "{}_timing_{}.csv", "{}_{}_reprompts.csv"):
            src = results / name.format("fix_gated_rules_vit_anchored_tiny", split)
            src.rename(results / name.format("fix_gated_rules_csrt_reinit_tiny", split))
    out = tmp_path / "e13.csv"
    assert run(results, choice, out) == 0
    roles = list(pd.read_csv(out).query("split == 'davis_val'")["role"])
    assert roles == ["alone", "e2", "variant_b", "oracle"]


@pytest.mark.parametrize(
    "missing", ["sam2_tiny", "fix_gated_rules_vit_anchored_large", "fix_rules_oracle_box_tiny"]
)
def test_missing_input_means_no_table(
    tmp_path: Path, missing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    results, choice = build(tmp_path, skip=(missing,))
    out = tmp_path / "e13.csv"
    assert run(results, choice, out) == 1
    assert "nothing written" in capsys.readouterr().out
    assert not out.exists() and not out.with_suffix(".tex").exists()


def test_missing_reprompt_file_for_a_new_variant_is_refused(tmp_path: Path) -> None:
    results, choice = build(tmp_path)
    (results / "fix_gated_rules_vit_anchored_tiny_davis_val_reprompts.csv").unlink()
    assert run(results, choice, tmp_path / "e13.csv") == 1


def test_latex_twin_is_written_from_the_csv(tmp_path: Path) -> None:
    results, choice = build(tmp_path)
    out = tmp_path / "e13.csv"
    run(results, choice, out)
    tex = out.with_suffix(".tex").read_text()
    assert tex.count("\\\\") >= 8 and "oracle" in tex and "DAVIS 2017 val" in tex


def test_figure_is_built_from_the_table_only(tmp_path: Path) -> None:
    results, choice = build(tmp_path)
    out = tmp_path / "e13.csv"
    run(results, choice, out)
    assert figure.main(["--table", str(out), "--figure-dir", str(tmp_path / "fig")]) == 0
    assert (tmp_path / "fig" / figure.FIGURE_NAME).stat().st_size > 0


def test_figure_refuses_a_missing_table(tmp_path: Path) -> None:
    assert figure.main(["--table", str(tmp_path / "none.csv"), "--figure-dir", str(tmp_path)]) == 1
    assert not (tmp_path / figure.FIGURE_NAME).exists()
