"""Four rule signals and the streaming rule detector (frames <= t only).

Each rule looks only at the observation at t and the same object's observation at t-1:
  low_score      object score below a threshold (NaN score never fires)
  area_change    both frames non-empty and max(a_t/a_prev, a_prev/a_t) above area_ratio
  centre_jump    both non-empty and centre distance / sqrt(a_prev) above jump_ratio
  disappearance  mask empty at t, large at t-1 (area / frame area >= min_area_fraction) and
                 its box at least border_margin_px from every image edge (so an object leaving
                 through the border is not a failure)
The object's first frame has no history and never flags; with no previous observation only
low_score can fire; a re-appearance (empty at t-1) can fire only low_score. Every rule is
"evidence compared with a threshold", so a tuning sweep can reuse the evidence columns.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import ConfigError, load_config, register
from src.detection.signals import FrameObs, join_signals_boxes, obs_from_row  # noqa: F401

LOW_SCORE = "low_score"
AREA_CHANGE = "area_change"
CENTRE_JUMP = "centre_jump"
DISAPPEARANCE = "disappearance"
ALL_RULES: tuple[str, ...] = (LOW_SCORE, AREA_CHANGE, CENTRE_JUMP, DISAPPEARANCE)
EVIDENCE_COLUMNS = ("score", "area_ratio", "jump", "prev_area_fraction", "prev_margin")
DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "detector" / "rules.yaml"
MODELS = ("tiny", "large")


@dataclasses.dataclass(frozen=True)
class RuleThresholds:
    score_threshold: float
    area_ratio: float
    jump_ratio: float
    min_area_fraction: float
    border_margin_px: float

    def __post_init__(self) -> None:
        values = dataclasses.astuple(self)
        if not all(math.isfinite(v) for v in values):
            raise ConfigError(f"rule thresholds must be finite: {self}")
        if self.area_ratio <= 1:
            raise ConfigError("area_ratio must be greater than 1")
        if self.jump_ratio <= 0:
            raise ConfigError("jump_ratio must be positive")
        if not 0 < self.min_area_fraction <= 1:
            raise ConfigError("min_area_fraction must be in (0, 1]")
        if self.border_margin_px < 0:
            raise ConfigError("border_margin_px must be non-negative")


@register("rules")
@dataclasses.dataclass(frozen=True)
class RulesConfig:
    tiny: RuleThresholds
    large: RuleThresholds

    def for_model(self, model: str) -> RuleThresholds:
        if model not in MODELS:
            raise ConfigError(f"unknown model '{model}' (known: {list(MODELS)})")
        return self.tiny if model == "tiny" else self.large


def load_rules_config(model: str, path: str | Path = DEFAULT_RULES_PATH) -> RuleThresholds:
    config = load_config(path)
    if not isinstance(config, RulesConfig):
        raise ConfigError(f"{path}: not a rules config")
    return config.for_model(model)


@dataclasses.dataclass(frozen=True)
class Evidence:
    """NaN means the rule does not apply at this frame (and so never fires)."""

    score: float
    area_ratio: float
    jump: float
    prev_area_fraction: float
    prev_margin: float


_NAN = float("nan")


def evidence(obs: FrameObs, prev: FrameObs | None) -> Evidence:
    area_ratio = jump = fraction = margin = _NAN
    if prev is not None and not prev.empty:
        if not obs.empty:
            hi, lo = max(obs.area, prev.area), min(obs.area, prev.area)
            area_ratio = hi / lo
            dist = math.hypot(obs.centre_x - prev.centre_x, obs.centre_y - prev.centre_y)
            jump = dist / math.sqrt(prev.area)
        else:
            box = prev.box
            fraction = prev.area / (prev.height * prev.width)
            if box is not None:
                margin = float(min(box.x0, box.y0, prev.width - box.x1, prev.height - box.y1))
    return Evidence(obs.object_score, area_ratio, jump, fraction, margin)


def fired_rules(
    ev: Evidence, cfg: RuleThresholds, enabled: Iterable[str] = ALL_RULES
) -> tuple[str, ...]:
    on = set(enabled)
    tests = {
        LOW_SCORE: ev.score < cfg.score_threshold,
        AREA_CHANGE: ev.area_ratio > cfg.area_ratio,
        CENTRE_JUMP: ev.jump > cfg.jump_ratio,
        DISAPPEARANCE: ev.prev_area_fraction >= cfg.min_area_fraction
        and ev.prev_margin >= cfg.border_margin_px,
    }
    return tuple(name for name in ALL_RULES if name in on and tests[name])


@dataclasses.dataclass(frozen=True)
class Decision:
    flag: bool
    fired: tuple[str, ...]


class RuleDetector:
    """Streaming detector for one object; keeps only the previous observation."""

    def __init__(self, config: RuleThresholds, enabled: Iterable[str] = ALL_RULES) -> None:
        self.config = config
        self.enabled = tuple(enabled)
        unknown = set(self.enabled) - set(ALL_RULES)
        if unknown:
            raise ValueError(f"unknown rules {sorted(unknown)}")
        self._prev: FrameObs | None = None

    def update(self, obs: FrameObs) -> Decision:
        prev, self._prev = self._prev, obs
        if prev is None:
            return Decision(False, ())
        fired = fired_rules(evidence(obs, prev), self.config, self.enabled)
        return Decision(bool(fired), fired)


def batch_evidence(table: pd.DataFrame) -> pd.DataFrame:
    """Evidence columns for a joined per-frame table (sorted, consecutive frames per object).

    The first row of each object has all evidence NaN (no history, never flags), so low_score
    is also silent there, matching RuleDetector.
    """
    group = table.groupby(["video", "object_id"], sort=False)
    has_prev = group["frame_index"].shift(1).notna().to_numpy()
    area = table["area"].to_numpy(dtype=float)
    prev_area = group["area"].shift(1).to_numpy(dtype=float)
    cx, cy = table["centre_x"].to_numpy(), table["centre_y"].to_numpy()
    pcx = group["centre_x"].shift(1).to_numpy(dtype=float)
    pcy = group["centre_y"].shift(1).to_numpy(dtype=float)
    both = has_prev & (area > 0) & (prev_area > 0)
    gone = has_prev & (area == 0) & (prev_area > 0)
    with np.errstate(all="ignore"):
        ratio = np.maximum(area / prev_area, prev_area / area)
        jump = np.hypot(cx - pcx, cy - pcy) / np.sqrt(prev_area)
        frame_area = (table["height"] * table["width"]).to_numpy(dtype=float)
        fraction = prev_area / frame_area
    margins = pd.concat(
        [
            group["box_x0"].shift(1),
            group["box_y0"].shift(1),
            table["width"] - group["box_x1"].shift(1),
            table["height"] - group["box_y1"].shift(1),
        ],
        axis=1,
    ).min(axis=1, skipna=False)
    return pd.DataFrame(
        {
            "score": np.where(has_prev, table["object_score"].to_numpy(dtype=float), np.nan),
            "area_ratio": np.where(both, ratio, np.nan),
            "jump": np.where(both, jump, np.nan),
            "prev_area_fraction": np.where(gone, fraction, np.nan),
            "prev_margin": np.where(gone, margins.to_numpy(dtype=float), np.nan),
        },
        index=table.index,
    )


def batch_flags(
    table: pd.DataFrame, cfg: RuleThresholds, enabled: Iterable[str] = ALL_RULES
) -> pd.DataFrame:
    """Per-rule bool columns plus `flag` (OR of the enabled rules), same as RuleDetector."""
    ev = batch_evidence(table)
    on = set(enabled)
    out = pd.DataFrame(
        {
            LOW_SCORE: ev["score"] < cfg.score_threshold,
            AREA_CHANGE: ev["area_ratio"] > cfg.area_ratio,
            CENTRE_JUMP: ev["jump"] > cfg.jump_ratio,
            DISAPPEARANCE: (ev["prev_area_fraction"] >= cfg.min_area_fraction)
            & (ev["prev_margin"] >= cfg.border_margin_px),
        },
        index=table.index,
    )
    for name in ALL_RULES:
        if name not in on:
            out[name] = False
    out["flag"] = out[list(ALL_RULES)].any(axis=1)
    return out
