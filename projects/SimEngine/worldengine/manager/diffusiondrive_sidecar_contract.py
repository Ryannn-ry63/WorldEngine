"""Identity helpers shared by DiffusionDrive sidecar consumers.

The planner names a rollout sidecar after the ``log_token`` of the frame it
consumes. Reproduced WorldEngine scenes use ``<origin-token>-<replica>`` for
that token, while the scenario ``id`` and ``token`` fields use longer names.
Keep all accepted spellings in one small, dependency-free contract so a
consumer cannot silently fall back to matching only the replica suffix.
"""

from __future__ import annotations

from collections.abc import Mapping


def _append(values: list[str], value) -> None:
    if value is None:
        return
    value = str(value).strip()
    if value:
        values.append(value)


def _producer_prefix_from_scene_id(scene_id) -> str | None:
    """Mirror DataManager's non-synthetic ``log_token`` construction."""

    if scene_id is None:
        return None
    parts = str(scene_id).split("-")
    if not parts:
        return None
    if len(parts[-1]) == 3 and len(parts) >= 2:
        return "-".join(parts[-2:])
    return parts[-1]


def scene_sidecar_prefixes(scene: Mapping) -> tuple[str, ...]:
    """Return exact producer identities followed by legacy lookup aliases."""

    if not isinstance(scene, Mapping):
        raise TypeError("DiffusionDrive scene must be a mapping")
    metadata = scene.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise TypeError("DiffusionDrive scene metadata must be a mapping")

    values: list[str] = []

    # Prepared rollout sources freeze this explicit producer identity.
    _append(values, metadata.get("rollout_sidecar_prefix"))
    _append(values, scene.get("rollout_sidecar_prefix"))

    # Synthetic scenes override DataManager's normal token construction.
    synthetic = metadata.get("synthetic_scene_info") or {}
    if isinstance(synthetic, Mapping):
        synthetic_metadata = synthetic.get("scene_metadata") or {}
        if isinstance(synthetic_metadata, Mapping):
            _append(values, synthetic_metadata.get("scene_token"))

    # Reproduce the ordinary DataManager identity even for older sources that
    # predate the explicit rollout_sidecar_prefix metadata field.
    _append(values, _producer_prefix_from_scene_id(scene.get("id")))

    # Preserve prior aliases for backward compatibility. They remain
    # fail-closed because the loader rejects multiple matching files.
    for key in ("id", "token"):
        value = scene.get(key)
        if value:
            value = str(value)
            _append(values, value)
            _append(values, value.rsplit("-", 1)[-1])
    return tuple(dict.fromkeys(values))
