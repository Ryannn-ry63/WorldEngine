#!/usr/bin/env python3
"""Load a project-local MMCV CUDA extension without modifying AlgEngine."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_isolated_mmcv(extension_path: str | Path) -> ModuleType:
    extension = Path(extension_path).expanduser().resolve()
    if not extension.is_file():
        raise FileNotFoundError(f"isolated MMCV extension is missing: {extension}")

    import torch  # noqa: F401

    module_name = "mmcv._ext"
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path != extension:
            raise RuntimeError(
                "mmcv._ext was imported before the isolated loader: "
                f"expected {extension}, got {existing_path}"
            )
        return existing

    spec = importlib.util.spec_from_file_location(module_name, extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
