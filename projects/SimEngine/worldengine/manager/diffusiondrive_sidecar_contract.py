"""Naming contract shared by DiffusionDrive rollout sidecar consumers."""

from __future__ import annotations


def sidecar_prefixes(scene: dict) -> tuple[str, ...]:
    """Return explicit BWM and legacy scene prefixes in priority order."""
    values: list[str] = []
    metadata = scene.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    explicit = metadata.get("rollout_sidecar_prefix")
    if explicit:
        explicit = str(explicit)
        if "/" in explicit or "\\" in explicit or explicit in (".", ".."):
            raise RuntimeError(f"unsafe rollout sidecar prefix: {explicit!r}")
        values.append(explicit)
    for key in ("id", "token"):
        value = scene.get(key)
        if value:
            value = str(value)
            values.extend((value, value.rsplit("-", 1)[-1]))
    return tuple(dict.fromkeys(values))
