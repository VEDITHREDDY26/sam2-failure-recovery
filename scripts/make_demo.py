"""Side-by-side demo video (SAM 2 alone | SAM 2 + fix | ground truth) of the selected clips.

    python -m scripts.make_demo --selection results/e10_selection.csv \
        --out <path outside the repo>.mp4 [--before 15] [--after 30] [--fps 10]

Each clip is a window around its selected frame (rule in experiments/e10_overlays/select.py) and starts with a
title card naming the video, the role, both J&F values and the results file it traces to. Clips
of an oracle method are captioned ORACLE. The video is never written inside the repository: the
output path is refused if it lies in it, and it is meant for Drive.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from experiments.e10_overlays.overlays import (
    LoadVideo,
    default_loader,
    is_oracle_row,
    load_clip_frame,
    overlay_mask,
    read_selection,
)
from src.common.config import DATA_ROOT_ENV
from src.data.video import VideoSample

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_HEIGHT = 360
CAPTION_HEIGHT = 56
CARD_SECONDS = 2
Image = npt.NDArray[np.uint8]


def check_output_path(out: Path) -> Path:
    resolved = out.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise SystemExit(f"{out} is inside the repository; write the demo video to Drive instead")
    if resolved.suffix.lower() != ".mp4":
        raise SystemExit(f"{out}: the demo video must be an .mp4 file")
    return resolved


def _text(image: Image, lines: Sequence[str], top: int = 18, scale: float = 0.5) -> None:
    for i, line in enumerate(lines):
        cv2.putText(
            image, line, (6, top + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1
        )


def _fit(image: Image, height: int) -> Image:
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    return np.asarray(cv2.resize(image, (width, height)), dtype=np.uint8)


def clip_label(row: pd.Series) -> list[str]:
    fix = f"ORACLE {row['fix_method']}" if is_oracle_row(row) else str(row["fix_method"])
    return [
        f"{row['split']} / {row['video']}  ({row['role']})",
        f"alone {row['alone_method']}  J&F {float(row['jf_alone']):.3f}   "
        f"fix {fix}  J&F {float(row['jf_fix']):.3f}",
        f"results: {row['alone_file']} and {row['fix_file']}",
    ]


def compose_frame(row: pd.Series, video: VideoSample, index: int, output_root: Path) -> Image:
    clip = load_clip_frame(output_root, row, video, index)
    gt = overlay_mask(clip.frame, clip.gt) if clip.gt is not None else np.zeros_like(clip.frame)
    if clip.gt is None:
        _text(gt, ["no annotation"])
    panels = [overlay_mask(clip.frame, clip.alone), overlay_mask(clip.frame, clip.fix), gt]
    fix = f"ORACLE {row['fix_method']}" if is_oracle_row(row) else str(row["fix_method"])
    for panel, name in zip(panels, (str(row["alone_method"]), fix, "ground truth"), strict=True):
        _text(panel, [name])
    strip = np.hstack([_fit(p, PANEL_HEIGHT) for p in panels])
    bar = np.zeros((CAPTION_HEIGHT, strip.shape[1], 3), dtype=np.uint8)
    _text(
        bar,
        [f"frame {index}" + ("  (re-prompt frame)" if index == row["frame_index"] else "")]
        + clip_label(row)[1:2],
        top=16,
        scale=0.45,
    )
    return np.vstack([bar, strip])


def title_card(row: pd.Series, size: tuple[int, int]) -> Image:
    card = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    _text(
        card,
        clip_label(row) + [f"selected frame {int(row['frame_index'])} ({row['frame_source']})"],
        top=size[1] // 2 - 30,
        scale=0.6,
    )
    return card


def clip_indices(frame_index: int, num_frames: int, before: int, after: int) -> range:
    return range(max(0, frame_index - before), min(num_frames, frame_index + after + 1))


def make_demo(
    selection: pd.DataFrame,
    output_root: Path,
    load_video: LoadVideo,
    out: Path,
    before: int = 15,
    after: int = 30,
    fps: int = 10,
) -> int:
    """Write the clips one after another; returns the number of video frames written."""
    out = check_output_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    size = (0, 0)
    written = 0
    for _, row in selection.iterrows():
        video = load_video(str(row["split"]), str(row["video"]))
        indices = clip_indices(int(row["frame_index"]), video.num_frames, before, after)
        frames = [compose_frame(row, video, i, output_root) for i in indices]
        if writer is None:
            size = (frames[0].shape[1], frames[0].shape[0])
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                raise OSError(f"cannot open {out} for writing")
        card = title_card(row, size)
        for image in [card] * (CARD_SECONDS * fps) + frames:
            writer.write(np.asarray(cv2.resize(image, size), dtype=np.uint8))
            written += 1
    if writer is None:
        raise ValueError("the selection has no clips")
    writer.release()
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--selection", type=Path, default=Path("results/e10_selection.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--before", type=int, default=15)
    parser.add_argument("--after", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args(argv)
    check_output_path(args.out)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    from src.sam2_runner.smoke import load_data_config

    cfg = load_data_config()
    count = make_demo(
        read_selection(args.selection),
        cfg.data_root / cfg.output_root,
        default_loader(cfg),
        args.out,
        args.before,
        args.after,
        args.fps,
    )
    print(f"{count} frames -> {args.out}")


if __name__ == "__main__":
    main()
