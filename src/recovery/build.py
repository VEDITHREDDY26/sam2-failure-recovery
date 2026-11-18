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
    """Choose the box source class a run config names and return a per-video factory for it.

    Args:
        run_cfg: Run config; its box_source is "csrt", "motion" or "vit_<mode>".
        settings: Recovery settings (supplies the CSRT tracker mode).

    Returns:
        A function from video name to a fresh BoxSource, so each video starts with no history.

    Raises:
        ValueError: If the box source is not a real-system one (the oracle is refused).
        ConfigError: For ViT sources, if the model file is missing or has a different checksum.
    """
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
