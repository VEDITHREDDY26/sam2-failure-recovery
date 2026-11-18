"""Learned failure detector: a logistic regression on per-frame features of SAM 2's own outputs.

Features at frame t use only the observation at t and the same object's observation at t-1 (the
evidence of the rule detector plus the log area fraction and an empty flag); NaN evidence is
replaced by 0 and flagged by an indicator feature. Training lives in experiments/e5/train.py; this
module only holds the fitted model (standardisation, coefficients, decision threshold), its JSON
file with a sha256, and the streaming detector for the recovery loop. The object's first frame has
no history and never flags, as for the rules.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.detection.rules import Evidence, batch_evidence, evidence
from src.detection.signals import FrameObs, obs_from_mask
from src.recovery.sources import Decision

FloatArray = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]

FIRED = "learned"
JUMP_CAP = 20.0
FEATURE_NAMES: tuple[str, ...] = (
    "score",
    "score_missing",
    "log_area_ratio",
    "jump",
    "pair_missing",
    "gone",
    "prev_area_fraction",
    "prev_margin_frac",
    "log_area_fraction",
    "empty",
)
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "configs" / "learned"


class ModelFileError(ValueError):
    """A model file is malformed or does not match its pinned hash."""


def feature_matrix(
    ev: Mapping[str, FloatArray], area: FloatArray, height: FloatArray, width: FloatArray
) -> FloatArray:
    """One row per frame; the single code path for streaming (1 row) and batch (all rows)."""
    # Missing evidence becomes 0 plus an indicator column, so the model can tell 'unknown' from 0.
    score, ratio, jump = ev["score"], ev["area_ratio"], ev["jump"]
    fraction, margin = ev["prev_area_fraction"], ev["prev_margin"]
    # log / divide of NaN or empty areas is expected; those entries are replaced below.
    with np.errstate(all="ignore"):
        columns = [
            np.nan_to_num(score, nan=0.0),
            np.isnan(score),
            np.nan_to_num(np.log(ratio), nan=0.0),
            np.nan_to_num(np.minimum(jump, JUMP_CAP), nan=0.0),
            np.isnan(ratio),
            ~np.isnan(fraction),
            np.nan_to_num(fraction, nan=0.0),
            np.nan_to_num(margin / np.minimum(height, width), nan=0.0),
            np.log((area + 1.0) / (height * width)),
            area == 0,
        ]
    return np.column_stack(columns).astype(np.float64)


def table_features(table: pd.DataFrame) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Features and has-history flag for a joined per-frame table (sorted, consecutive frames)."""
    # Reuse the rule evidence, then mark each object's first frame as having no history.
    ev = batch_evidence(table)
    arrays = {c: ev[c].to_numpy(dtype=float) for c in ev.columns}
    first = table.groupby(["video", "object_id"], sort=False)["frame_index"].cumcount() == 0
    matrix = feature_matrix(
        arrays,
        table["area"].to_numpy(dtype=float),
        table["height"].to_numpy(dtype=float),
        table["width"].to_numpy(dtype=float),
    )
    return matrix, ~first.to_numpy()


def obs_features(ev: Evidence, obs: FrameObs) -> FloatArray:
    """Build the feature row of a single frame for the streaming detector.

    Args:
        ev: Rule evidence for the frame (see src.detection.rules.evidence).
        obs: The observation at the same frame.

    Returns:
        A vector with one value per name in FEATURE_NAMES, computed by the same code as the batch
        path so streaming and batch features agree.
    """
    # Wrap the single frame as one-row arrays so it goes through the same feature code as batch.
    arrays = {
        "score": np.array([ev.score]),
        "area_ratio": np.array([ev.area_ratio]),
        "jump": np.array([ev.jump]),
        "prev_area_fraction": np.array([ev.prev_area_fraction]),
        "prev_margin": np.array([ev.prev_margin]),
    }
    row: FloatArray = feature_matrix(
        arrays,
        np.array([float(obs.area)]),
        np.array([float(obs.height)]),
        np.array([float(obs.width)]),
    )[0]
    return row


@dataclasses.dataclass(frozen=True)
class LearnedModel:
    """A fitted logistic-regression failure detector with its standardisation and threshold.

    Attributes:
        sam2_model: SAM 2 model size the detector was fitted for ("tiny" or "large").
        mean: Per-feature mean used to standardise features.
        std: Per-feature standard deviation used to standardise features (positive).
        coef: Logistic-regression weight of each standardised feature.
        intercept: Logistic-regression intercept.
        threshold: Failure probability at or above which a frame is flagged, in (0, 1).
        feature_names: Feature order; must equal FEATURE_NAMES.
    """

    sam2_model: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    threshold: float
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        """Check feature names, lengths, finiteness, positive std and the threshold range."""
        n = len(self.feature_names)
        if self.feature_names != FEATURE_NAMES:
            raise ModelFileError(f"feature names differ from {FEATURE_NAMES}")
        if not (len(self.mean) == len(self.std) == len(self.coef) == n):
            raise ModelFileError("mean, std and coef must have one entry per feature")
        values = [*self.mean, *self.std, *self.coef, self.intercept, self.threshold]
        if not all(math.isfinite(v) for v in values) or min(self.std) <= 0:
            raise ModelFileError("model numbers must be finite and std positive")
        if not 0 < self.threshold < 1:
            raise ModelFileError("threshold must be in (0, 1)")

    def probabilities(self, features: FloatArray) -> FloatArray:
        """Return the failure probability of each feature row.

        Args:
            features: Array of shape (frames, len(FEATURE_NAMES)).

        Returns:
            Probabilities in [0, 1], one per row (standardise, linear score, logistic function).
        """
        z = (features - np.array(self.mean)) / np.array(self.std)
        logit = (z * np.array(self.coef)).sum(axis=1) + self.intercept
        return np.asarray(1.0 / (1.0 + np.exp(-logit)), dtype=np.float64)

    def table_probabilities(self, table: pd.DataFrame) -> FloatArray:
        """Failure probability per row; 0 on each object's first frame (no history)."""
        features, has_prev = table_features(table)
        return np.where(has_prev, self.probabilities(features), 0.0)


def model_text(model: LearnedModel) -> str:
    """Serialise a model to a stable JSON string (sorted keys, indented, trailing newline)."""
    # Fields are written with sorted keys so the file, and its hash, are stable.
    body = {
        "kind": "logreg",
        "sam2_model": model.sam2_model,
        "feature_names": list(model.feature_names),
        "mean": list(model.mean),
        "std": list(model.std),
        "coef": list(model.coef),
        "intercept": model.intercept,
        "threshold": model.threshold,
    }
    return json.dumps(body, indent=2, sort_keys=True) + "\n"


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_model(path: Path, model: LearnedModel) -> str:
    """Write the JSON file and its pinned hash next to it (<name>.sha256); returns the hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model_text(model))
    digest = sha256_of(path)
    # Pin the hash so a changed model file is detected when loaded.
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def load_model(path: Path, expected_sha256: str | None = None) -> LearnedModel:
    """Read a model from its JSON file.

    Args:
        path: Model JSON file.
        expected_sha256: If given, the file's SHA-256 must equal it.

    Raises:
        ModelFileError: If the file is missing, does not match the hash or is malformed.
    """
    if not path.is_file():
        raise ModelFileError(f"missing model file {path}")
    if expected_sha256 is not None and sha256_of(path) != expected_sha256:
        raise ModelFileError(f"{path}: sha256 does not match the pinned hash")
    try:
        # Rebuild the model from the file; any missing or malformed field becomes ModelFileError.
        body = json.loads(path.read_text())
        return LearnedModel(
            str(body["sam2_model"]),
            tuple(float(v) for v in body["mean"]),
            tuple(float(v) for v in body["std"]),
            tuple(float(v) for v in body["coef"]),
            float(body["intercept"]),
            float(body["threshold"]),
            tuple(body["feature_names"]),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ModelFileError(f"{path}: not a learned-detector file ({error})") from error


def model_path(sam2_model: str, directory: Path = DEFAULT_DIR) -> Path:
    """Return the JSON file path of the logistic-regression model for a SAM 2 model size."""
    return directory / f"logreg_{sam2_model}.json"


def pinned_hash(path: Path) -> str:
    """Read the SHA-256 recorded next to a model file.

    Raises:
        ModelFileError: If the .sha256 file does not exist.
    """
    pin = path.with_suffix(".sha256")
    if not pin.is_file():
        raise ModelFileError(f"no pinned hash {pin}: train the model first")
    return pin.read_text().split()[0]


def load_pinned(sam2_model: str, directory: Path = DEFAULT_DIR) -> LearnedModel:
    """Load configs/learned/logreg_<model>.json only if it matches its pinned sha256."""
    path = model_path(sam2_model, directory)
    return load_model(path, pinned_hash(path))


class OnlineLearnedDetector:
    """Streaming detector (Detector protocol): one previous observation per object id."""

    def __init__(self, model: LearnedModel) -> None:
        """Store the model; per-object history starts empty."""
        self.model = model
        self._prev: dict[int, FrameObs] = {}

    @classmethod
    def from_pinned(cls, sam2_model: str, directory: Path = DEFAULT_DIR) -> OnlineLearnedDetector:
        """Build a detector from the shipped model of a SAM 2 size, checking its pinned hash."""
        return cls(load_pinned(sam2_model, directory))

    def reset(self) -> None:
        """Forget all per-object history (call before a new video)."""
        self._prev.clear()

    def probability(self, obj_id: int, obs: FrameObs) -> float | None:
        """Return the failure probability of an observation given the object's previous one.

        Returns:
            The probability, or None on the object's first frame (no history).
        """
        prev = self._prev.get(obj_id)
        if prev is None:
            return None
        features = obs_features(evidence(obs, prev), obs)
        return float(self.model.probabilities(features[None, :])[0])

    def update(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> Decision:
        """Feed one frame of one object and return the decision for it.

        Args:
            obj_id: Object id.
            frame_index: 0-based frame index (not used; only the previously fed frame matters).
            mask: Boolean HxW mask of the object on this frame.
            object_score: SAM 2's object score for this frame.
            frame_shape: Frame (height, width) in pixels; must equal the mask shape.

        Returns:
            A flagged Decision (fired = "learned") if the probability reaches the model threshold,
            otherwise an unflagged one; the first frame of an object never flags.

        Raises:
            ValueError: If the mask shape differs from the frame shape.
        """
        obs = self._obs(mask, object_score, frame_shape)
        # Score against the previous frame first, then make this frame the new previous one.
        probability = self.probability(obj_id, obs)
        self._prev[obj_id] = obs
        if probability is not None and probability >= self.model.threshold:
            return Decision(True, (FIRED,))
        return Decision(False, ())

    def observe(
        self,
        obj_id: int,
        frame_index: int,
        mask: Mask,
        object_score: float,
        frame_shape: tuple[int, int],
    ) -> None:
        """Record the frame as the object's previous observation without producing a decision."""
        self._prev[obj_id] = self._obs(mask, object_score, frame_shape)

    @staticmethod
    def _obs(mask: Mask, object_score: float, frame_shape: tuple[int, int]) -> FrameObs:
        """Build a FrameObs from a mask and score, checking the mask matches the frame shape."""
        if mask.shape != frame_shape:
            raise ValueError(f"mask shape {mask.shape} does not match frame shape {frame_shape}")
        return obs_from_mask(mask, object_score)


def weights_table(model: LearnedModel) -> Sequence[tuple[str, float, float, float]]:
    """(feature, standardised weight, mean, std) per feature."""
    return list(zip(model.feature_names, model.coef, model.mean, model.std, strict=True))
