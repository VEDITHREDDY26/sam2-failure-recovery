import ast
import json
import re
from pathlib import Path

import pytest

from src.sam2_runner.predictor import REPO_ROOT

NOTEBOOK = REPO_ROOT / "notebooks" / "run_sam2.ipynb"


@pytest.fixture(scope="module")
def code() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_all_six_run_commands_are_present(code: str) -> None:
    for model in ("tiny", "large"):
        for split in ("davis_val", "sav_val", "sav_test"):
            pattern = (
                rf"-m src\.sam2_runner\.run --model {model} "
                rf"--prompts \{{PROMPTS_DIR\}}/{split}\.json"
            )
            assert len(re.findall(pattern, code)) == 1, (model, split)


def test_install_builds_sam2_against_colab_torch(code: str) -> None:
    assert "SAM2_BUILD_ALLOW_ERRORS=0 pip install --no-build-isolation" in code
    assert 'l.startswith("SAM-2")' in code and "requirements-colab.txt" in code
    assert "pip uninstall -y SAM-2" in code
    requirement = next(
        line
        for line in (REPO_ROOT / "requirements-colab.txt").read_text().splitlines()
        if line.startswith("SAM-2")
    )
    assert "@2b90b9f5ceec907a1c18123530e92e794ad901a4" in requirement


def test_hole_filling_check_stops_the_notebook_on_failure(code: str) -> None:
    assert "from sam2 import _C" in code
    assert "get_connected_components" in code and "fill_holes_in_mask_scores" in code
    assert 'simplefilter("error")' in code
    assert "assert " in code and "hole filling" in code.lower()
    assert "raise RuntimeError" in code and "Send the full output" in code
    assert code.index("from sam2 import _C") < code.index("src.sam2_runner.run --model")


def test_prompts_check_and_score_cells_follow_the_runs(code: str) -> None:
    for split in ("davis_val", "sav_val", "sav_test"):
        assert f"experiments.sam2_runs.make_prompts --split {split}" in code
    last_run = code.rindex("src.sam2_runner.run --model")
    assert (
        last_run
        < code.index("src.eval.check_complete")
        < code.index("experiments.sam2_runs.score_all")
    )
    assert "get_checkpoints.sh" in code and "src.eval.get_evaluators" in code
    assert 'BRANCH = "phase-2b-sam2-runs"' in code
    assert "sam2_*.csv" in code


def test_code_cells_parse_apart_from_shell_magics() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        plain = "\n".join(
            (
                line[: len(line) - len(line.lstrip())] + "pass"
                if line.lstrip().startswith(("!", "%"))
                else line
            )
            for line in lines
        )
        ast.parse(plain)


def test_pilot_and_failure_stats_cells_are_placed_around_the_runs(code: str) -> None:
    first_run = code.index("src.sam2_runner.run --model tiny --prompts")
    assert code.index("--method sam2_tiny_pilot") < first_run
    assert "--method sam2_tiny " not in code and "--method sam2_tiny\n" not in code
    assert code.index("sam2_tiny_pilot") < first_run
    score = code.index("experiments.sam2_runs.score_all")
    stats = code.index("experiments.sam2_runs.failure_stats")
    assert score < stats < code.index('glob("failure_stats.csv")')
    assert "results/failure_stats.csv" in code and "expected 13" in code


def _cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _plain(cell: str) -> str:
    return "\n".join(
        "pass" if line.lstrip().startswith(("!", "%")) else line for line in cell.splitlines()
    )


def test_pilot_cells_run_on_a_fake_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.sam2_runner.prompts import parse_batch_prompts
    from src.sam2_runner.run import run_batch
    from src.sam2_runner.smoke import load_data_config
    from tests.eval.sam2_trees import build_world, run_split
    from tests.fixtures.synthetic_video import make_synthetic_video
    from tests.sam2_runner.fakes import FakePredictor

    root = tmp_path / "drive"
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(root))
    subset = build_world(root)
    source = make_synthetic_video(root / "fake_source", "source")
    run_split(root, "tiny", "sav_val", subset, source)
    cfg = load_data_config()
    first = next(c for c in _cells() if "PILOT_VIDEO, PILOT_OBJ, PILOT_START" in c)
    second = next(c for c in _cells() if "late = signals[" in c)
    scope: dict[str, object] = {
        "PROMPTS_DIR": cfg.data_root / cfg.output_root / "prompts",
        "cfg": cfg,
    }
    exec(_plain(first), scope)
    assert (scope["PILOT_VIDEO"], scope["PILOT_OBJ"], scope["PILOT_START"]) == ("vid_a", 3, 8)
    pilot_file = scope["pilot_file"]
    assert isinstance(pilot_file, Path)
    prompts = parse_batch_prompts(json.loads(pilot_file.read_text()))
    assert list(prompts.videos) == ["vid_a"]
    run_batch(prompts, cfg, "sam2_tiny_pilot", lambda: FakePredictor(source))
    exec(second, scope)
    signals = scope["signals"]
    before = scope["before"]
    assert len(before) == 8 and before["empty"].all() and before.object_score.isna().all()  # type: ignore[index,attr-defined]
    assert scope["PILOT_START"] == 8 and len(signals) > 0  # type: ignore[arg-type]


def _assigned_names(source: str) -> set[str]:
    """Names bound by assignment, import or for-loop in one cell (shell lines become pass)."""
    lines = [
        (
            line[: len(line) - len(line.lstrip())] + "pass"
            if line.lstrip().startswith(("!", "%"))
            else line
        )
        for line in source.splitlines()
    ]
    names: set[str] = set()
    for node in ast.walk(ast.parse("\n".join(lines))):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return names


def test_shell_lines_use_only_names_defined_in_earlier_cells() -> None:
    """A {NAME} in a ! line must be bound in that cell or an earlier one (in notebook order)."""
    nb = json.loads(NOTEBOOK.read_text())
    defined: set[str] = set()
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        defined |= _assigned_names(source)
        for line in source.splitlines():
            if line.lstrip().startswith("!"):
                for name in re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)", line):
                    assert name in defined, f"cell {index}: {{{name}}} used before it is defined"


def test_local_disk_cell_unpacks_sav_tars_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Drive holds DAVIS unpacked and SA-V only as tars; the cell builds the local root."""
    import shutil
    import subprocess

    from src.data.davis import list_davis_videos
    from src.data.sav import list_sav_videos
    from src.sam2_runner.smoke import load_data_config
    from tests.eval.sam2_trees import DAVIS, SAV_TEST, SAV_VAL, build_world

    world = tmp_path / "world"
    build_world(world)
    drive = tmp_path / "drive"
    shutil.copytree(world / "DAVIS", drive / "DAVIS")
    extra = drive / "DAVIS" / "JPEGImages" / "480p" / "train_only"  # not in val.txt: not copied
    extra.mkdir()
    (extra / "00000.jpg").write_bytes(b"x")
    (drive / "sav_tars").mkdir()
    for split in ("sav_val", "sav_test"):
        tar = drive / "sav_tars" / f"{split}.tar"
        subprocess.run(["tar", "-cf", str(tar), "-C", str(world / "SA-V"), split], check=True)
    local = tmp_path / "local"
    monkeypatch.setenv("SAM2FR_LOCAL_ROOT", str(local))
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(drive))  # the cell overwrites it; restored after
    cell = next(c for c in _cells() if "def local_sav_split" in c)

    scope: dict[str, object] = {"DRIVE_DATA_ROOT": str(drive)}
    exec(cell, scope)
    assert scope["PROMPTS_DIR"] == local / "outputs" / "prompts"
    assert (local / "outputs").resolve() == (drive / "outputs").resolve()
    assert (local / "checkpoints").resolve() == (drive / "checkpoints").resolve()
    cfg = load_data_config()
    assert cfg.data_root == local
    assert sorted(list_davis_videos(cfg)) == sorted(DAVIS)
    assert not (local / "DAVIS" / "JPEGImages" / "480p" / "train_only").exists()
    out = capsys.readouterr().out
    assert f"DAVIS {len(DAVIS)}/{len(DAVIS)}" in out and "unpacking" in out
    assert list_sav_videos(cfg, "val") == SAV_VAL
    assert list_sav_videos(cfg, "test") == SAV_TEST

    exec(cell, {"DRIVE_DATA_ROOT": str(drive)})  # second run: nothing unpacked again
    out = capsys.readouterr().out
    assert "unpacking" not in out and out.count("already on the local disk") == 2

    (local / "SA-V" / "sav_val" / ".unpacked").unlink()  # as if the session died mid-unpack
    (local / "SA-V" / "sav_val" / "sav_val.txt").unlink()
    exec(cell, {"DRIVE_DATA_ROOT": str(drive)})
    assert "sav_val: unpacking" in capsys.readouterr().out
    assert list_sav_videos(load_data_config(), "val") == SAV_VAL


def test_local_disk_cell_names_the_missing_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.data.helpers import build_davis_root

    drive = tmp_path / "drive"
    build_davis_root(drive)
    monkeypatch.setenv("SAM2FR_LOCAL_ROOT", str(tmp_path / "local"))
    monkeypatch.setenv("SAM2FR_DATA_ROOT", str(drive))
    cell = next(c for c in _cells() if "def local_sav_split" in c)
    with pytest.raises(FileNotFoundError, match="sav_tars/sav_val.tar"):
        exec(cell, {"DRIVE_DATA_ROOT": str(drive)})


def test_data_check_cell_follows_the_local_disk_cell() -> None:
    cells = _cells()
    local = next(i for i, c in enumerate(cells) if "def local_sav_split" in c)
    check = next(i for i, c in enumerate(cells) if "src.data.check --dataset sav --split test" in c)
    prompts = next(i for i, c in enumerate(cells) if "make_prompts --split davis_val" in c)
    assert local < check < prompts
