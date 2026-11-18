"""Strict YAML config loader; each file's `kind` selects a frozen dataclass schema."""

from __future__ import annotations

import dataclasses
import os
import re
import types
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import yaml

DATA_ROOT_ENV = "SAM2FR_DATA_ROOT"
KIND_KEY = "kind"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

T = TypeVar("T")
REGISTRY: dict[str, type[Any]] = {}


class ConfigError(ValueError):
    """Raised for any invalid config file."""


def register(kind: str) -> Callable[[type[T]], type[T]]:
    """Register a frozen dataclass as the schema for configs with `kind: <kind>`."""

    def decorator(cls: type[T]) -> type[T]:
        """Class decorator that records the schema class under the kind and returns it unchanged.

        Raises:
            ConfigError: If the kind is already registered.
        """
        if kind in REGISTRY:
            raise ConfigError(f"config kind '{kind}' is already registered")
        REGISTRY[kind] = cls
        return cls

    return decorator


@dataclasses.dataclass(frozen=True)
class DavisConfig:
    """Location and variant of the DAVIS 2017 dataset.

    Attributes:
        root: Dataset folder, relative to the data root.
        split: Split name, for example "val".
        resolution: Image resolution folder name, for example "480p".
        year: DAVIS release year.
    """

    root: Path
    split: str
    resolution: str
    year: int


@dataclasses.dataclass(frozen=True)
class SavConfig:
    """Location and split names of the SA-V dataset.

    Attributes:
        root: Dataset folder, relative to the data root.
        val_split: Name of the SA-V validation split (used to fit detector settings).
        test_split: Name of the held-out SA-V test subset.
    """

    root: Path
    val_split: str
    test_split: str


@register("data")
@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Schema of the "data" config: where datasets and run outputs live.

    Attributes:
        data_root: Root folder that holds the datasets, checkpoints and outputs.
        davis: DAVIS dataset settings.
        sav: SA-V dataset settings.
        output_root: Folder for run outputs, relative to the data root.
    """

    data_root: Path
    davis: DavisConfig
    sav: SavConfig
    output_root: Path

    def resolve(self, relative: Path) -> Path:
        """Join a sub-path from this config onto the data root."""
        return self.data_root / relative


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """One SAM 2 model variant.

    Attributes:
        model_cfg: Name of the SAM 2 model config file.
        checkpoint: Checkpoint file name inside the checkpoint folder.
    """

    model_cfg: str
    checkpoint: str


@register("models")
@dataclasses.dataclass(frozen=True)
class ModelsConfig:
    """Schema of the "models" config: checkpoint folder and the two SAM 2 variants.

    Attributes:
        checkpoint_dir: Checkpoint folder, relative to the data root.
        tiny: Spec of the Hiera-Tiny model.
        large: Spec of the Hiera-Large model.
    """

    checkpoint_dir: Path
    tiny: ModelSpec
    large: ModelSpec

    def checkpoint_path(self, data_root: Path, spec: ModelSpec) -> Path:
        """Full path of a model's checkpoint file.

        Args:
            data_root: Root folder of the data.
            spec: Model variant whose checkpoint is wanted.

        Returns:
            data_root / checkpoint_dir / spec.checkpoint.
        """
        return data_root / self.checkpoint_dir / spec.checkpoint


def _expand(text: str, where: str) -> str:
    """Replace every ${NAME} in a string with the value of that environment variable.

    Args:
        text: Raw string from the YAML file.
        where: Config location used in error messages.

    Returns:
        The string with all variables substituted.

    Raises:
        ConfigError: If a variable is not set or is empty.
    """

    def sub(match: re.Match[str]) -> str:
        """Look up one matched ${NAME} in the environment and return its value."""
        name = match.group(1)
        if name not in os.environ:
            raise ConfigError(f"{where}: environment variable {name} is not set")
        value = os.environ[name]
        if not value.strip():
            raise ConfigError(f"{where}: environment variable {name} is set but empty")
        return value

    return _ENV_PATTERN.sub(sub, text)


def _convert(value: Any, hint: Any, where: str) -> Any:
    """Convert a raw YAML value to the type a schema field declares.

    Args:
        value: Value parsed from YAML.
        hint: Type annotation of the field (dataclass, list of items, or a scalar type).
        where: Config location used in error messages.

    Returns:
        The converted value (a nested dataclass, a list, or a scalar).

    Raises:
        ConfigError: If the value has the wrong shape or the annotation is a union.
    """
    # Dispatch on the annotation: nested dataclass, list, union (refused) or scalar.
    origin = typing.get_origin(hint)
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return _build(hint, value, where)
    if origin is list:
        (item,) = typing.get_args(hint)
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        return [_convert(v, item, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin in (typing.Union, types.UnionType):
        raise ConfigError(f"{where}: union types are not supported in schemas")
    return _convert_scalar(value, hint, where)


def _convert_scalar(value: Any, hint: Any, where: str) -> Any:
    """Check and convert a str, Path, bool, int or float value.

    Strings and paths get ${VAR} expansion; ints are accepted for float fields; bools are never
    accepted as numbers.

    Args:
        value: Value parsed from YAML.
        hint: The scalar type the field declares.
        where: Config location used in error messages.

    Returns:
        The value in the declared type.

    Raises:
        ConfigError: If the value has the wrong type or the type is unsupported.
    """
    # Text fields must be strings in the YAML; ${VAR} references are expanded first.
    if hint is Path or hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {type(value).__name__}")
        text = _expand(value, where)
        return Path(text) if hint is Path else text
    # Numeric and bool fields: bools never count as numbers, and ints are widened to float.
    if hint is bool:
        ok = isinstance(value, bool)
    elif hint is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif hint is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    else:
        raise ConfigError(f"{where}: unsupported schema type {hint}")
    if not ok:
        raise ConfigError(f"{where}: expected {hint.__name__}, got {type(value).__name__}")
    return value


def _build(cls: type[T], raw: Any, where: str) -> T:
    """Build one schema dataclass from a mapping, rejecting unknown and missing keys.

    Args:
        cls: Frozen dataclass to instantiate.
        raw: Mapping parsed from YAML for this dataclass.
        where: Config location used in error messages.

    Returns:
        An instance of cls with every field converted.

    Raises:
        ConfigError: If raw is not a mapping, has an unknown key, or lacks a required key.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    # Strict schema: unknown keys are errors, so typos cannot silently change a run.
    for key in raw:
        if key not in names:
            raise ConfigError(f"{where}: unknown key '{key}'")
    # Convert each field with its declared type; every field is required.
    values: dict[str, Any] = {}
    for name in names:
        if name not in raw:
            raise ConfigError(f"{where}: missing required key '{name}'")
        values[name] = _convert(raw[name], hints[name], f"{where}.{name}")
    return cls(**values)


def load_config(path: str | Path) -> Any:
    """Load and validate a YAML config; the return type is the schema registered for its kind."""
    path = Path(path)
    # Any read or YAML parse error is reported as a ConfigError naming the file.
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: cannot read config ({exc})") from exc
    # The top-level kind picks the registered schema; the remaining keys are built into it.
    if not isinstance(raw, dict) or KIND_KEY not in raw:
        raise ConfigError(f"{path}: missing top-level key '{KIND_KEY}'")
    kind = raw.pop(KIND_KEY)
    if kind not in REGISTRY:
        raise ConfigError(f"{path}: unknown kind '{kind}' (known: {sorted(REGISTRY)})")
    return _build(REGISTRY[kind], raw, str(path))
