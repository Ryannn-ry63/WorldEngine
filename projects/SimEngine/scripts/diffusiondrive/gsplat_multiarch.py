#!/usr/bin/env python3
"""Load an isolated gsplat CUDA extension without modifying its installation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_isolated_gsplat(extension_path: str | Path) -> ModuleType:
    """Load ``gsplat.csrc`` from an explicit project-local shared library."""
    extension = Path(extension_path).expanduser().resolve()
    if not extension.is_file():
        raise FileNotFoundError(f"isolated gsplat extension is missing: {extension}")

    # Import torch first so libtorch/libc10 are available to the extension.
    import torch  # noqa: F401

    module_name = "gsplat.csrc"
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path != extension:
            raise RuntimeError(
                "gsplat.csrc was imported before the isolated loader: "
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
