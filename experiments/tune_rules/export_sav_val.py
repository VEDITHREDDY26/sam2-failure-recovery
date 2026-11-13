"""Export per-frame SA-V val tables for tuning the rule detector (run next to the masks).

    python -m experiments.tune_rules.export_sav_val --models tiny large [--out-dir DIR]

Only the SA-V val split is ever read. Per model, two files are written into the output folder
(default $SAM2FR_DATA_ROOT/tune_rules):

    labels_<model>_sav_val.csv   video, object_id, frame_index, iou, failure
    boxes_<model>_sav_val.csv    video, object_id, frame_index, height, width,
                                 box_x0, box_y0, box_x1, box_y1  (xyxy; empty mask -> empty cells)

Boxes cover every frame from each object's prompt frame on. Finished videos are kept as part
files so an interrupted export resumes. A run that fails the completeness check is refused.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.common.boxes import box_from_mask
from src.common.config import DATA_ROOT_ENV
from src.common.layout import mask_path, validate_method
from src.common.masks import read_label_map
from src.data.video import NeverVisibleError, VideoSample
from src.detection.labels import LABEL_COLUMNS, video_labels
from src.eval.check_complete import MODELS, SUBSET_FILE, check_run, load_video, split_videos
from src.eval.score import load_data_config

SPLIT = "sav_val"
METHOD_PREFIX = "sam2"
BOX_COLUMNS = (
    "video",
    "object_id",
    "frame_index",
    "height",
    "width",
    "box_x0",
    "box_y0",
    "box_x1",
    "box_y1",
)


def video_boxes(video: VideoSample, output_root: Path, method: str) -> pd.DataFrame:
    """Tight predicted box per (object, frame) from the object's prompt frame on."""
    starts: dict[int, int] = {}
    for obj_id in video.object_ids:
        try:
            starts[obj_id] = video.first_box(obj_id).frame_index
        except NeverVisibleError:
            continue
    rows: list[tuple[object, ...]] = []
    if starts:
        for frame_index in range(min(starts.values()), video.num_frames):
            stem = video.frame_stems[frame_index]
            label = read_label_map(mask_path(output_root, method, SPLIT, video.name, stem))
            height, width = label.shape
            for obj_id, start in starts.items():
                if frame_index < start:
                    continue
                box = box_from_mask(label == obj_id)
                corners = (box.x0, box.y0, box.x1, box.y1) if box else (None,) * 4
                rows.append((video.name, obj_id, frame_index, height, width, *corners))
    frame = pd.DataFrame(rows, columns=list(BOX_COLUMNS))
    return frame.astype({c: "Int64" for c in BOX_COLUMNS[3:]})


def write_atomic(frame: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, out)


def _part(parts_dir: Path, kind: str, video: str) -> Path:
    return parts_dir / f"{kind}_{video}.csv"


def export_model(model: str, out_dir: Path, subset_file: Path = SUBSET_FILE) -> bool:
    """Write both tables for one model; False (after saying why) if the run is incomplete."""
    cfg = load_data_config()
    method = validate_method(f"{METHOD_PREFIX}_{model}")
    check = check_run(cfg, method, SPLIT, subset_file)
    if check.missing:
        sys.stdout.write(
            "\n".join(check.lines()) + f"\n{method}/{SPLIT}: refused, run is not complete\n"
        )
        return False
    output_root = cfg.data_root / cfg.output_root
    names = split_videos(cfg, SPLIT, subset_file)
    parts_dir = out_dir / f".parts_{model}_{SPLIT}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names, 1):
        label_part, box_part = _part(parts_dir, "labels", name), _part(parts_dir, "boxes", name)
        if label_part.is_file() and box_part.is_file():
            sys.stdout.write(f"{model} {i}/{len(names)} {name}: already done\n")
            continue
        video = load_video(cfg, SPLIT, name)
        write_atomic(video_labels(video, output_root, method, SPLIT), label_part)
        write_atomic(video_boxes(video, output_root, method), box_part)
        sys.stdout.write(f"{model} {i}/{len(names)} {name}\n")
        sys.stdout.flush()
    for kind, columns in (("labels", LABEL_COLUMNS), ("boxes", BOX_COLUMNS)):
        tables = [pd.read_csv(_part(parts_dir, kind, n)) for n in names]
        table = pd.concat(tables, ignore_index=True)[list(columns)]
        target = out_dir / f"{kind}_{model}_{SPLIT}.csv"
        write_atomic(table, target)
        sys.stdout.write(f"{target}: {len(table)} rows\n")
    shutil.rmtree(parts_dir)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--subset-file", type=Path, default=SUBSET_FILE)
    args = parser.parse_args(argv)
    root = os.environ.get(DATA_ROOT_ENV)
    if not root:
        parser.error(f"{DATA_ROOT_ENV} is not set")
    out_dir = args.out_dir or Path(root) / "tune_rules"
    try:
        done = [export_model(model, out_dir, args.subset_file) for model in args.models]
    except Exception as error:  # noqa: BLE001
        sys.stdout.write(f"export_sav_val: failed: {type(error).__name__}: {error}\n")
        return 1
    if not all(done):
        sys.stdout.write("export_sav_val: some runs are incomplete\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
