"""Read recipe parameters.

AIR writes the `parameters:` block of `workload.yaml` to a YAML file and exports its path as
`HYPERPARAMETERS_PATH`. Recipes declare defaults in a dataclass; values from the file override them.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any


def load_parameters(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Return the AIR parameters as a dict (empty if no file is available)."""
    path = path or os.environ.get("HYPERPARAMETERS_PATH")
    if not path or not Path(path).is_file():
        return {}
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def build_config[T](cls: type[T], overrides: dict[str, Any]) -> T:
    """Instantiate dataclass `cls`, applying `overrides`; unknown keys are an error (catches typos)."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(overrides) - names)
    if unknown:
        raise ValueError(f"Unknown parameters for {cls.__name__}: {unknown}")
    return cls(**overrides)
