"""Build the box source a run config names (real-system sources only; never the oracle)."""

from __future__ import annotations

from collections.abc import Callable

from src.recovery.config import RecoveryRunConfig
from src.recovery.settings import RecoverySettings
from src.recovery.sources import BoxSource, CSRTBoxSource, MotionBoxSource, ViTBoxSource
from src.recovery.vit_tracker import load_vit_config, make_vit_factory


def make_box_source(
    run_cfg: RecoveryRunConfig, settings: RecoverySettings
) -> Callable[[str], BoxSource]:
    source = run_cfg.box_source
    if source == "csrt":
        return lambda video: CSRTBoxSource(settings.tracker_mode)
    if source == "motion":
        return lambda video: MotionBoxSource()
    if source.startswith("vit_"):
        vit = load_vit_config()
        factory = make_vit_factory(vit)
        return lambda video: ViTBoxSource(factory, source.removeprefix("vit_"), vit.anchor_iou)
    raise ValueError(f"no real-system box source '{source}'")
