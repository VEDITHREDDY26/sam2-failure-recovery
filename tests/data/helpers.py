from __future__ import annotations

from pathlib import Path

from src.common.config import DataConfig, DavisConfig, SavConfig
from tests.fixtures.synthetic_video import SyntheticVideo, make_synthetic_video

VIDEO_NAMES = ("zebra", "bear")


def make_config(data_root: Path) -> DataConfig:
    return DataConfig(
        data_root=data_root,
        davis=DavisConfig(Path("DAVIS"), "val", "480p", 2017),
        sav=SavConfig(Path("SA-V"), "sav_val", "sav_test"),
        output_root=Path("out"),
    )


def build_davis_root(data_root: Path, names: tuple[str, ...] = VIDEO_NAMES) -> list[SyntheticVideo]:
    davis = data_root / "DAVIS"
    videos = [make_synthetic_video(davis, name) for name in names]
    sets = davis / "ImageSets" / "2017"
    sets.mkdir(parents=True)
    (sets / "val.txt").write_text("".join(f"{name}\n" for name in names))
    return videos
