"""Per-video timing of the SAM 2 runs, and the collector that gathers it into one repo CSV.

    python -m src.sam2_runner.timing --model tiny --split davis_val \
        --videos-file videos.txt --out results/sam2_tiny_timing_davis_val.csv

Per video: <output_root>/<method>/<split>/timing/<video>.csv with the same core columns as the
baseline timing (video, frames, objects, compute_s, wall_s, fps) plus device fields.
compute_s covers prompting, SAM 2 propagation and the conversion of its outputs to numpy; wall_s
is the whole call (frame loading, PNG writing and signals included). The collector fails when a
listed video has no timing.
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import DATA_ROOT_ENV
from src.common.layout import run_dir, validate_method, validate_name, validate_split

CORE_COLUMNS = ("video", "frames", "objects", "compute_s", "wall_s", "fps")
DEVICE_COLUMNS = ("gpu_name", "torch_version", "cuda_version", "offload_video")
TIMING_COLUMNS = CORE_COLUMNS + DEVICE_COLUMNS


def timing_path(output_root: Path, method: str, split: str, video: str) -> Path:
    """Return the path of one video's timing CSV.

    Args:
        output_root: Root folder of run outputs.
        method: Method name.
        split: Split name.
        video: Video name (validated so it is safe in a path).

    Returns:
        <output_root>/<method>/<split>/timing/<video>.csv
    """
    return run_dir(output_root, method, split) / "timing" / f"{validate_name(video, 'video')}.csv"


def device_fields(offload_video: bool = False) -> dict[str, Any]:
    """GPU and library versions; empty strings when torch (or a GPU) is not there."""
    fields: dict[str, Any] = {name: "" for name in DEVICE_COLUMNS}
    fields["offload_video"] = int(offload_video)
    # torch is optional: on a CPU-only machine the device fields stay empty.
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return fields
    fields["torch_version"] = str(torch.__version__)
    fields["cuda_version"] = str(torch.version.cuda or "")
    if torch.cuda.is_available():
        fields["gpu_name"] = str(torch.cuda.get_device_name(0))
    return fields


def timing_row(
    video: str, frames: int, objects: int, compute_s: float, wall_s: float, offload_video: bool
) -> dict[str, Any]:
    """Build the timing row of one video.

    Args:
        video: Video name.
        frames: Number of frames in the video.
        objects: Number of objects tracked.
        compute_s: Seconds of SAM 2 compute (see the module docstring).
        wall_s: Seconds for the whole call.
        offload_video: Whether the frames were kept on the CPU.

    Returns:
        Dict with the core columns (fps = frames / compute_s, NaN if compute_s is 0) and the
        device fields.
    """
    fps = frames / compute_s if compute_s > 0 else float("nan")
    row: dict[str, Any] = dict(
        zip(CORE_COLUMNS, (video, frames, objects, compute_s, wall_s, fps), strict=True)
    )
    return {**row, **device_fields(offload_video)}


def write_timing(path: Path, row: dict[str, Any]) -> None:
    """Write a one-row timing CSV, creating its folder if needed.

    Args:
        path: Destination CSV.
        row: Row dict from timing_row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=list(TIMING_COLUMNS)).to_csv(path, index=False)


def read_timing(path: Path, video: str) -> pd.DataFrame | None:
    """The one-row timing table, or None if the file is missing or unusable."""
    try:
        table = pd.read_csv(path, dtype={"video": str})
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if list(table.columns) != list(TIMING_COLUMNS) or len(table) != 1:
        return None
    if table["video"].iloc[0] != video:
        return None
    return table


def collect_timing(
    output_root: Path, method: str, split: str, videos: Sequence[str]
) -> pd.DataFrame:
    """All listed videos' timing rows in order; any missing or unusable one is an error."""
    tables = {v: read_timing(timing_path(output_root, method, split, v), v) for v in videos}
    missing = [v for v, t in tables.items() if t is None]
    if missing:
        raise FileNotFoundError(f"no timing for {len(missing)} video(s): {missing[:10]}")
    return pd.concat([t for t in tables.values() if t is not None], ignore_index=True)


def read_video_list(path: Path) -> list[str]:
    """Read video names from a text file, one per line; blank lines are ignored.

    Args:
        path: Text file to read.

    Raises:
        ValueError: If the file has no names.
    """
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise ValueError(f"{path}: no video names")
    return names


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point: collect per-video timing files into one CSV.

    Reads the timing file of every video listed in --videos-file from the run folder, writes the
    combined table to --out and prints the video count and the overall frames per second.

    Args:
        argv: Command-line arguments; None means sys.argv.

    Raises:
        SystemExit: Status 2 from argparse on bad arguments or when the data-root environment
            variable is not set.
        FileNotFoundError: If a listed video has no usable timing file.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", choices=["tiny", "large"], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--videos-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", default=None)
    args = parser.parse_args(argv)
    # Fail early with a clear message when the data folder is not configured.
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    from src.sam2_runner.smoke import load_data_config

    cfg = load_data_config()
    method = validate_method(args.method or f"sam2_{args.model}")
    # Every listed video must have timing; a missing one stops the collection.
    table = collect_timing(
        cfg.data_root / cfg.output_root,
        method,
        validate_split(args.split),
        read_video_list(args.videos_file),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    # Overall speed is total frames over total compute time, not a mean of per-video rates.
    total_fps = table["frames"].sum() / table["compute_s"].sum()
    print(f"{len(table)} videos, {total_fps:.1f} fps overall -> {args.out}")


if __name__ == "__main__":
    main()
