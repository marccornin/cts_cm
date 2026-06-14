from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast, get_args, get_origin, get_type_hints

import yaml

from cts_cm.almanac.tables import Settings


def _coerce(value: Any, annotation: Any) -> Any:
    if dataclasses.is_dataclass(annotation) and isinstance(value, dict):
        return _build(annotation, value)
    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        elem = args[0] if args else Any
        return tuple(_coerce(v, elem) for v in value)
    if origin is list:
        args = get_args(annotation)
        elem = args[0] if args else Any
        return [_coerce(v, elem) for v in value]
    return value


def _build(cls: Any, data: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(data[f.name], hints[f.name])
    return cls(**kwargs)


def _parse_scalar(raw: str, reference: Any) -> Any:
    if isinstance(reference, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(reference, int):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    if isinstance(reference, tuple):
        parts = [p for p in raw.split(",") if p != ""]
        if reference and isinstance(reference[0], int):
            return tuple(int(p) for p in parts)
        if reference and isinstance(reference[0], float):
            return tuple(float(p) for p in parts)
        return tuple(parts)
    return raw


def _set_path(obj: Any, parts: list[str], raw: str) -> Any:
    name = parts[0]
    current = getattr(obj, name)
    if len(parts) == 1:
        return dataclasses.replace(obj, **{name: _parse_scalar(raw, current)})
    return dataclasses.replace(obj, **{name: _set_path(current, parts[1:], raw)})


def apply_override(settings: Settings, expr: str) -> Settings:
    key, _, raw = expr.partition("=")
    return cast(Settings, _set_path(settings, key.split("."), raw))


def load_settings(path: str | Path, overrides: Iterable[str] = ()) -> Settings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    settings = cast(Settings, _build(Settings, raw))
    for expr in overrides:
        settings = apply_override(settings, expr)
    return settings
