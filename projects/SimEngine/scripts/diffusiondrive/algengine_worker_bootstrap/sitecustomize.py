"""Opt-in H100 extension injection for DiffusionDrive workers only."""

import importlib.util
import os
import sys
from pathlib import Path


def _active_environment():
    return Path(sys.prefix).name


def _bootstrap_mmcv_extension():
    if os.environ.get("WORLDENGINE_DIFFUSIONDRIVE_MMCV_BOOTSTRAP") != "1":
        return
    if _active_environment() != "algengine":
        return
    extension_path = Path(
        os.environ["WORLDENGINE_MMCV_EXTENSION"]
    ).expanduser().resolve()
    if not extension_path.is_file():
        raise FileNotFoundError(
            f"DiffusionDrive mmcv extension does not exist: {extension_path}"
        )

    # Import torch first so libtorch symbols needed by the extension are live.
    import torch  # noqa: F401
    import mmcv

    module_name = "mmcv._ext"
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load mmcv extension from {extension_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    setattr(mmcv, "_ext", module)


def _bootstrap_gsplat_extension():
    if os.environ.get("WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_BOOTSTRAP") != "1":
        return
    if _active_environment() != "simengine":
        return
    extension = os.environ.get("WORLDENGINE_GSPLAT_EXTENSION")
    if not extension:
        raise RuntimeError(
            "WORLDENGINE_GSPLAT_EXTENSION is required when the "
            "DiffusionDrive gsplat bootstrap is enabled"
        )

    script_dir = Path(__file__).resolve().parent.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from gsplat_multiarch import load_isolated_gsplat

    module = load_isolated_gsplat(extension)
    os.environ["WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_LOADED"] = str(
        Path(module.__file__).resolve()
    )


_bootstrap_mmcv_extension()
_bootstrap_gsplat_extension()
