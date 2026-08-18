#!/usr/bin/env python3
"""Shared provenance helpers for DiffusionDrive selector cache extraction.

The original diagnostic extractor was retired after V3 cache selection, but
the schema-v2 context extractor still imports these small, format-defining
helpers. Keep them dependency-light so all cache extractors share one contract.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import mmcv
import yaml


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_filter_tokens(path):
    path = Path(path)
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"filter is not a mapping: {path}")
    tokens = payload.get("tokens") or payload.get("scenario_tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"filter has no tokens list: {path}")
    values = [str(token) for token in tokens]
    if len(values) != len(set(values)):
        raise ValueError(f"filter contains duplicate tokens: {path}")
    return values


def load_scene_map(annotation_file, eligible_tokens):
    payload = mmcv.load(str(annotation_file), file_format="pkl")
    infos = payload["infos"] if isinstance(payload, dict) else payload
    eligible = set(str(token) for token in eligible_tokens)
    mapping = {}
    for info in infos:
        token = str(info["token"])
        if token in eligible:
            mapping[token] = str(
                info.get("scene_token") or info.get("log_token") or info["log_name"]
            )
    missing = eligible - set(mapping)
    if missing:
        examples = ", ".join(sorted(missing)[:10])
        raise RuntimeError(
            f"annotation omitted {len(missing)} filtered tokens; examples: {examples}"
        )
    return mapping


def configure_dataset(cfg, nav_filter):
    dataset_cfg = copy.deepcopy(cfg.data.train)
    dataset_cfg.nav_filter_path = str(nav_filter)
    dataset_cfg.test_mode = False
    return dataset_cfg
