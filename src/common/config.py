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
        if kind in REGISTRY:
            raise ConfigError(f"config kind '{kind}' is already registered")
        REGISTRY[kind] = cls
        return cls

    return decorator


@dataclasses.dataclass(frozen=True)
class DavisConfig:
    root: Path
    split: str
    resolution: str
    year: int


@dataclasses.dataclass(frozen=True)
class SavConfig:
    root: Path
    val_split: str
    test_split: str


@register("data")
@dataclasses.dataclass(frozen=True)
class DataConfig:
    data_root: Path
    davis: DavisConfig
    sav: SavConfig
    output_root: Path

    def resolve(self, relative: Path) -> Path:
        """Join a sub-path from this config onto the data root."""
        return self.data_root / relative


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    model_cfg: str
    checkpoint: str


@register("models")
@dataclasses.dataclass(frozen=True)
class ModelsConfig:
    checkpoint_dir: Path
    tiny: ModelSpec
    large: ModelSpec

    def checkpoint_path(self, data_root: Path, spec: ModelSpec) -> Path:
        return data_root / self.checkpoint_dir / spec.checkpoint


def _expand(text: str, where: str) -> str:
    def sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConfigError(f"{where}: environment variable {name} is not set")
        value = os.environ[name]
        if not value.strip():
            raise ConfigError(f"{where}: environment variable {name} is set but empty")
        return value

    return _ENV_PATTERN.sub(sub, text)


def _convert(value: Any, hint: Any, where: str) -> Any:
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
    if hint is Path or hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {type(value).__name__}")
        text = _expand(value, where)
        return Path(text) if hint is Path else text
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
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(cls)
    fields = dataclasses.fields(cls)  # type: ignore[arg-type]
    names = {f.name for f in fields}
    optional = {
        f.name
        for f in fields
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
    }
    for key in raw:
        if key not in names:
            raise ConfigError(f"{where}: unknown key '{key}'")
    values: dict[str, Any] = {}
    for name in names:
        if name not in raw and name in optional:
            continue
        if name not in raw:
            raise ConfigError(f"{where}: missing required key '{name}'")
        values[name] = _convert(raw[name], hints[name], f"{where}.{name}")
    return cls(**values)


def load_config(path: str | Path) -> Any:
    """Load and validate a YAML config; the return type is the schema registered for its kind."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: cannot read config ({exc})") from exc
    if not isinstance(raw, dict) or KIND_KEY not in raw:
        raise ConfigError(f"{path}: missing top-level key '{KIND_KEY}'")
    kind = raw.pop(KIND_KEY)
    if kind not in REGISTRY:
        raise ConfigError(f"{path}: unknown kind '{kind}' (known: {sorted(REGISTRY)})")
    return _build(REGISTRY[kind], raw, str(path))
