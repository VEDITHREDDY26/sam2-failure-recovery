"""E10 overlay figures: SAM 2 alone | SAM 2 + fix | ground truth at the selected frame.

    python -m experiments.e10_overlays.overlays --selection results/e10_selection.csv \
        [--figures-dir figures]

One figure per selected clip, figures/e10_<split>_<video>.png (e10_oracle_<split>_<video>.png and
an ORACLE title when the fix method is oracle-based). The title carries the per-video J&F of both
runs as read from the results CSVs by experiments.e10_overlays.select. Runs where the data and the
masks are (Newton, or a Mac holding the selected videos' masks and frames).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.config import DATA_ROOT_ENV, DataConfig  # noqa: E402
from src.common.layout import is_oracle_method, mask_path  # noqa: E402
from src.common.masks import (
    LabelMap,
    davis_palette,
    masks_to_label_map,
    read_label_map,
)  # noqa: E402
from src.data.video import VideoSample  # noqa: E402

FIGURE_SIZE = (13.0, 4.6)
DPI = 150
ALPHA = 0.5
ORACLE_TAG = "ORACLE"
SAV_SPLITS = {"sav_val": "val", "sav_test": "test"}  # result split name -> SA-V loader split
LoadVideo = Callable[[str, str], VideoSample]
Image = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class ClipFrame:
    """One frame with the masks to draw on it.

    Attributes:
        frame: The video frame as a BGR uint8 image.
        alone: Label map predicted by SAM 2 alone at this frame.
        fix: Label map predicted by SAM 2 with the fix at this frame.
        gt: Ground-truth label map, or None when the frame has no annotation.
    """

    frame: Image  # BGR
    alone: LabelMap
    fix: LabelMap
    gt: LabelMap | None  # None: no annotation at this frame


def default_loader(cfg: DataConfig) -> LoadVideo:
    """Build the function that loads a video of any supported test set.

    Args:
        cfg: Data configuration with the dataset root and layout.

    Returns:
        A function (split, video name) -> VideoSample for davis_val, sav_val and sav_test.
    """
    from src.data.davis import load_davis_video
    from src.data.sav import load_sav_video

    def load(split: str, name: str) -> VideoSample:
        """Load one video of a test set.

        Raises:
            ValueError: If the split name is not davis_val, sav_val or sav_test.
        """
        if split == "davis_val":
            return load_davis_video(cfg, name)
        if split in SAV_SPLITS:
            return load_sav_video(cfg, SAV_SPLITS[split], name)
        expected = ["davis_val", *sorted(SAV_SPLITS)]
        raise ValueError(f"unknown split '{split}': expected one of {expected}")

    return load


def overlay_mask(frame_bgr: Image, label: LabelMap, alpha: float = ALPHA) -> Image:
    """Colour each object (DAVIS palette) over the frame and draw its outline; BGR in and out."""
    out = frame_bgr.astype(np.float32)
    palette = np.array(davis_palette(), dtype=np.float32).reshape(-1, 3)[:, ::-1]
    for obj_id in (int(i) for i in np.unique(label) if i != 0):
        mask = label == obj_id
        out[mask] = (1 - alpha) * out[mask] + alpha * palette[obj_id]
    result = np.clip(out, 0, 255).astype(np.uint8)
    for obj_id in (int(i) for i in np.unique(label) if i != 0):
        found, _ = cv2.findContours(
            (label == obj_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, found, -1, tuple(int(c) for c in palette[obj_id]), 2)
    return result


def gt_label_map(video: VideoSample, frame_index: int, shape: tuple[int, int]) -> LabelMap | None:
    """Return the ground truth of one frame as a label map, or None if it has no annotation.

    Args:
        video: The loaded video.
        frame_index: Zero-based frame index.
        shape: (height, width) of the frame.
    """
    masks = video.gt_masks(frame_index)
    if not masks:
        return None
    return masks_to_label_map(masks, shape)


def load_clip_frame(
    output_root: Path, row: pd.Series, video: VideoSample, frame_index: int | None = None
) -> ClipFrame:
    """Frame and the two methods' masks (read only) and the ground truth at one frame index."""
    index = int(row["frame_index"]) if frame_index is None else frame_index
    stem = video.frame_stems[index]
    image = cv2.imread(str(video.frame_path(index)), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read frame {video.frame_path(index)}")
    frame = np.asarray(image, dtype=np.uint8)
    maps = [
        read_label_map(mask_path(output_root, str(row[key]), str(row["split"]), video.name, stem))
        for key in ("alone_method", "fix_method")
    ]
    return ClipFrame(
        frame, maps[0], maps[1], gt_label_map(video, index, (frame.shape[0], frame.shape[1]))
    )


def is_oracle_row(row: pd.Series) -> bool:
    """Tell whether a selection row uses an oracle (ground-truth based) fix method."""
    return bool(row["oracle"]) or is_oracle_method(str(row["fix_method"]))


def figure_name(row: pd.Series) -> str:
    """Return the figure file name for a selection row; oracle rows get an oracle_ tag."""
    tag = "oracle_" if is_oracle_row(row) else ""
    return f"e10_{tag}{row['split']}_{row['video']}.png"


def figure_title(row: pd.Series) -> str:
    """Return the figure title: test set, video, frame, role and J&F of both runs."""
    tag = f"{ORACLE_TAG}: " if is_oracle_row(row) else ""
    return (
        f"{tag}{row['split']} / {row['video']}, frame {int(row['frame_index'])} "
        f"({row['role']}, {row['frame_source']}): J&F {float(row['jf_alone']):.3f} alone, "
        f"{float(row['jf_fix']):.3f} with fix ({row['fix_file']})"
    )


def render_figure(row: pd.Series, clip: ClipFrame, path: Path) -> Path:
    # Oracle runs are labelled as such in the panel title; a missing ground truth gives a black
    # panel.
    """Draw the three panels (alone, with fix, ground truth) and save them as one image.

    Args:
        row: Selection row naming the methods, split, video and scores.
        clip: Frame and masks to draw.
        path: Output image path; parent folders are created.

    Returns:
        The path that was written.
    """
    fix_name = f"{ORACLE_TAG} {row['fix_method']}" if is_oracle_row(row) else str(row["fix_method"])
    gt_panel: Any = (
        overlay_mask(clip.frame, clip.gt) if clip.gt is not None else np.zeros_like(clip.frame)
    )
    panels = [
        (f"SAM 2 alone ({row['alone_method']})", overlay_mask(clip.frame, clip.alone)),
        (f"SAM 2 + fix ({fix_name})", overlay_mask(clip.frame, clip.fix)),
        ("ground truth" if clip.gt is not None else "no annotation at this frame", gt_panel),
    ]
    # Panels are BGR (OpenCV); matplotlib needs RGB.
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, dpi=DPI)
    for ax, (label, image) in zip(axes, panels, strict=True):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.suptitle(figure_title(row), fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def build_overlays(
    selection: pd.DataFrame, output_root: Path, load_video: LoadVideo, figures_dir: Path
) -> list[Path]:
    """Render one overlay figure per selected clip.

    Args:
        selection: Selection table from experiments.e10_overlays.select.
        output_root: Folder holding the saved SAM 2 masks of every method.
        load_video: Function (split, name) -> VideoSample.
        figures_dir: Folder for the figures.

    Returns:
        Paths of the figures, in selection order.
    """
    paths: list[Path] = []
    for _, row in selection.iterrows():
        video = load_video(str(row["split"]), str(row["video"]))
        clip = load_clip_frame(output_root, row, video)
        paths.append(render_figure(row, clip, figures_dir / figure_name(row)))
    return paths


def read_selection(path: Path) -> pd.DataFrame:
    """Read the selection CSV, keeping video names as text.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"selection file not found: {path} (run experiments.e10_overlays.select)"
        )
    return pd.read_csv(path, dtype={"video": str})


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point: draw the E10 overlay figures.

    Reads the --selection CSV (default results/e10_selection.csv) and, through the data
    configuration, the dataset frames, ground truth and saved masks. Writes one PNG per clip
    to --figures-dir (default figures) and prints the figure count. Exits through the argument
    parser with an error if the data-root environment variable is not set.

    Args:
        argv: Command line arguments; defaults to sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--selection", type=Path, default=Path("results/e10_selection.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    from src.sam2_runner.smoke import load_data_config

    cfg = load_data_config()
    paths = build_overlays(
        read_selection(args.selection),
        cfg.data_root / cfg.output_root,
        default_loader(cfg),
        args.figures_dir,
    )
    print(f"{len(paths)} figures -> {args.figures_dir}")


if __name__ == "__main__":
    main()
