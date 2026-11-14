"""Configuration loading.

The whole study is driven by one YAML file (see ``configs/default.yaml``).
This module loads it, applies environment-variable expansion to paths and
exposes a small helper to resolve output sub-directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict with attribute access (``cfg.paths.raw_dir``) for convenience."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    # Make Config picklable / copyable like a plain dict.
    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | os.PathLike) -> Config:
    """Load a YAML config file and expand ``$VARS`` / ``~`` inside strings."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(_expand(raw))
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def out_dir(cfg: Config, *parts: str, create: bool = True) -> Path:
    """Return ``<paths.out_dir>/<parts...>``, creating it when requested."""
    path = Path(cfg["paths"]["out_dir"]).joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
