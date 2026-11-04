from __future__ import annotations

from pathlib import Path

import pytest

from src.data.make_subset import load_subset, main, pick_subset

IDS = tuple(f"sav_{n:06d}" for n in range(1, 151))


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    list_file = tmp_path / "sav_test.txt"
    list_file.write_text("".join(f"{i}\n" for i in reversed(IDS)))
    return {
        "list": list_file,
        "out": tmp_path / "splits" / "subset.txt",
        "results": tmp_path / "results",
    }


def _run(paths: dict[str, Path], *extra: str) -> int:
    return main(
        [
            "--list-file", str(paths["list"]),
            "--output", str(paths["out"]),
            "--results-dir", str(paths["results"]),
            *extra,
        ]
    )  # fmt: skip


def test_deterministic_and_well_formed(paths: dict[str, Path]) -> None:
    assert _run(paths) == 0
    first = paths["out"].read_text()
    paths["out"].unlink()
    assert _run(paths, "--seed", "5415", "--size", "60") == 0
    assert paths["out"].read_text() == first
    ids = load_subset(paths["out"])
    assert len(ids) == 60 == len(set(ids))
    assert list(ids) == sorted(ids)
    assert set(ids) <= set(IDS)
    header = first.splitlines()[:3]
    assert header[0] == "# SA-V test subset: seed 5415, size 60"
    assert header[
        2
    ] == "# command: python -m src.data.make_subset --seed 5415 --size 60 --list-file " + str(
        paths["list"]
    )


def test_input_order_does_not_matter() -> None:
    assert pick_subset(IDS, 5415, 60) == pick_subset(tuple(reversed(IDS)), 5415, 60)


def test_other_seed_gives_other_subset() -> None:
    assert pick_subset(IDS, 1, 60) != pick_subset(IDS, 2, 60)


def test_wrong_list_size_rejected(
    paths: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    paths["list"].write_text("".join(f"{i}\n" for i in IDS[:100]))
    assert _run(paths) == 1
    assert "expected 150 test videos, found 100" in capsys.readouterr().out
    assert not paths["out"].exists()


def test_overwrite_refused_when_ids_differ(paths: dict[str, Path]) -> None:
    assert _run(paths) == 0
    before = paths["out"].read_text()
    assert _run(paths, "--seed", "1") == 1
    assert paths["out"].read_text() == before
    assert _run(paths) == 0


def test_refuses_when_sav_test_results_exist(paths: dict[str, Path]) -> None:
    (paths["results"] / "e2").mkdir(parents=True)
    (paths["results"] / "e2" / "sav_test_tiny.csv").write_text("video\n")
    assert _run(paths) == 1
    assert not paths["out"].exists()


def test_other_results_do_not_block(paths: dict[str, Path]) -> None:
    (paths["results"]).mkdir()
    (paths["results"] / "davis_val_tiny.csv").write_text("video\n")
    assert _run(paths) == 0


def test_check_pass_and_fail(paths: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(paths) == 0
    assert _run(paths, "--check") == 0
    assert _run(paths, "--check", "--seed", "9") == 1
    ids = list(load_subset(paths["out"]))
    paths["out"].write_text("\n".join(["# edited", *ids[:-1], "sav_999999"]) + "\n")
    assert _run(paths, "--check") == 1
    assert "not in the test list" in capsys.readouterr().out


def test_check_missing_file(paths: dict[str, Path]) -> None:
    assert _run(paths, "--check") == 1


def test_reader_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "s.txt"
    path.write_text("# header\na\n\n  b \n# x\nc\n")
    assert load_subset(path) == ("a", "b", "c")
