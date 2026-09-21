#!/usr/bin/env python3
"""Audit the immutable three-lane SimEngine input for rare-policy rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError(f"input audit did not pass: {path}")
    return payload


def load_pairs(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("rare/common pair manifest is empty")
    rare = [str(row["rare_token"]) for row in rows]
    if len(set(rare)) != len(rare):
        raise RuntimeError("rare/common pair manifest repeats a rare token")
    return rows


def scenario_origin_token(scenario_id: str) -> str:
    if "-" not in scenario_id:
        raise RuntimeError(f"scenario id has no origin token: {scenario_id}")
    token = scenario_id.rsplit("-", 1)[-1]
    if not token:
        raise RuntimeError(f"scenario id has an empty origin token: {scenario_id}")
    return token


def asset_token_coverage(config_root: Path) -> set[str]:
    tokens: set[str] = set()
    paths = sorted(config_root.glob("*.yaml"))
    if not paths:
        raise RuntimeError(f"no DigitalTwin asset configs under {config_root}")
    for path in paths:
        # DigitalTwin configs intentionally carry Python object/tuple tags.
        # BaseLoader reads the scalar/list contract without importing the
        # senior-only config class or executing a Python-object constructor.
        payload = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for token in payload.get("central_tokens", []):
            token = str(token)
            if token in tokens:
                raise RuntimeError(f"asset configs repeat central token {token}")
            tokens.add(token)
    return tokens


def audit_scenarios(
    converter_manifest: Path,
    rare_pairs: Path,
    rare_audit: Path,
    rare_filter: Path,
    asset_config_root: Path,
    expected_lanes: int,
    expected_scenarios: int,
    expected_imputed_scenarios: int,
    expected_imputed_frames: int,
) -> dict:
    converter = load_json(converter_manifest)
    rare = load_json(rare_audit)
    pairs = load_pairs(rare_pairs)
    filter_payload = yaml.safe_load(rare_filter.read_text())
    filter_tokens = [str(token) for token in filter_payload.get("tokens", [])]
    pair_tokens = [str(row["rare_token"]) for row in pairs]
    expected = set(pair_tokens)

    if rare.get("method") != "diffusiondrive_v3_full_navtrain_rare_original_v1":
        raise RuntimeError("unexpected rare-original audit method")
    if rare.get("outputs", {}).get("pairs_sha256") != sha256_file(rare_pairs):
        raise RuntimeError("rare/common pair manifest SHA256 drifted")
    if int(rare.get("rare_count", -1)) != expected_scenarios:
        raise RuntimeError("rare-original count drifted")
    if len(pair_tokens) != expected_scenarios or len(expected) != expected_scenarios:
        raise RuntimeError("rare/common pairs do not cover the expected rare population")
    if len(filter_tokens) != expected_scenarios or set(filter_tokens) != expected:
        raise RuntimeError("rare token filter disagrees with the audited pair manifest")
    if int(converter.get("num_splits", -1)) != expected_lanes:
        raise RuntimeError("converter lane count drifted")
    if int(converter.get("num_scenarios", -1)) != expected_scenarios:
        raise RuntimeError("converter scenario count drifted")
    if converter.get("all_scenarios_path") is not None:
        raise RuntimeError("rare rollout must use shards-only conversion")

    assets = asset_token_coverage(asset_config_root)
    missing_assets = expected - assets
    if missing_assets:
        raise RuntimeError(
            f"{len(missing_assets)} rare tokens have no DigitalTwin asset config"
        )

    seen_scenarios: set[str] = set()
    seen_tokens: set[str] = set()
    imputed_scenario_tokens: set[str] = set()
    imputed_frame_count = 0
    imputation_method = "nearest_available_sampled_frame_tie_earlier_v1"
    lane_rows = []
    shards = converter.get("shards", [])
    if len(shards) != expected_lanes:
        raise RuntimeError("converter manifest does not contain every lane")
    for expected_index, row in enumerate(shards):
        if int(row.get("index", -1)) != expected_index:
            raise RuntimeError("scenario lane indices are not contiguous")
        path = Path(row["path"]).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise RuntimeError(f"scenario shard provenance drifted: {path}")
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict):
            raise RuntimeError(f"scenario shard is not a dictionary: {path}")
        if len(payload) != int(row.get("num_scenarios", -1)):
            raise RuntimeError(f"scenario shard count drifted: {path}")
        lane_tokens = []
        for scenario_id, scenario in payload.items():
            scenario_id = str(scenario_id)
            if scenario_id in seen_scenarios:
                raise RuntimeError(f"scenario is repeated across lanes: {scenario_id}")
            if str(scenario.get("id")) != scenario_id:
                raise RuntimeError(f"scenario key/id mismatch: {scenario_id}")
            token = scenario_origin_token(scenario_id)
            if token not in expected:
                raise RuntimeError(f"non-rare scenario entered rollout: {scenario_id}")
            if token in seen_tokens:
                raise RuntimeError(f"rare token produced multiple scenarios: {token}")

            metadata = scenario.get("metadata")
            if not isinstance(metadata, dict):
                raise RuntimeError(f"scenario metadata is not a dictionary: {scenario_id}")
            imputations = metadata.get("openscene_context_imputations", [])
            if not isinstance(imputations, list):
                raise RuntimeError(
                    f"OpenScene context imputations are not a list: {scenario_id}"
                )
            if imputations:
                if metadata.get("openscene_context_imputation_method") != imputation_method:
                    raise RuntimeError(
                        f"unexpected OpenScene context imputation method: {scenario_id}"
                    )
                lidar_tokens = [
                    str(value)
                    for value in metadata.get("nuplan_lidar_pc_tokens", [])
                ]
                openscene_infos = metadata.get("openscene_data_infos_dict", {})
                if not lidar_tokens or not isinstance(openscene_infos, dict):
                    raise RuntimeError(
                        f"imputed scenario has incomplete OpenScene metadata: {scenario_id}"
                    )
                if not all(isinstance(value, dict) for value in imputations):
                    raise RuntimeError(
                        f"invalid OpenScene context imputation row: {scenario_id}"
                    )
                targets = {str(value.get("target_token")) for value in imputations}
                for value in imputations:
                    target = str(value.get("target_token"))
                    source = str(value.get("source_token"))
                    offset = value.get("sample_offset")
                    if (
                        target == token
                        or target == source
                        or target not in lidar_tokens
                        or source not in lidar_tokens
                        or source in targets
                        or target not in openscene_infos
                        or source not in openscene_infos
                        or not isinstance(offset, int)
                        or offset == 0
                    ):
                        raise RuntimeError(
                            f"invalid OpenScene context imputation contract: {scenario_id}"
                        )
                    if str(openscene_infos[target].get("token")) != target:
                        raise RuntimeError(
                            f"imputed OpenScene frame token drifted: {scenario_id}"
                        )
                imputed_scenario_tokens.add(token)
                imputed_frame_count += len(imputations)
            seen_scenarios.add(scenario_id)
            seen_tokens.add(token)
            lane_tokens.append(token)
        lane_rows.append(
            {
                "lane": expected_index,
                "scenario_file": str(path),
                "scenario_file_sha256": row["sha256"],
                "num_scenarios": len(payload),
                "tokens_sha256": hashlib.sha256(
                    "\n".join(sorted(lane_tokens)).encode("utf-8")
                ).hexdigest(),
            }
        )

    if seen_tokens != expected:
        raise RuntimeError(
            "scenario shards do not exactly cover rare tokens: "
            f"missing={len(expected - seen_tokens)} extra={len(seen_tokens - expected)}"
        )
    if len(imputed_scenario_tokens) != expected_imputed_scenarios:
        raise RuntimeError(
            "OpenScene context imputed-scenario count drifted: "
            f"expected={expected_imputed_scenarios} "
            f"observed={len(imputed_scenario_tokens)}"
        )
    if imputed_frame_count != expected_imputed_frames:
        raise RuntimeError(
            "OpenScene context imputed-frame count drifted: "
            f"expected={expected_imputed_frames} observed={imputed_frame_count}"
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_rollout_scenario_contract_v1",
        "source_kind": "full_navtrain_rare_original_v1",
        "num_scenarios": len(seen_scenarios),
        "num_origin_rare_tokens": len(seen_tokens),
        "num_lanes": expected_lanes,
        "openscene_context_imputation_method": imputation_method,
        "openscene_context_imputed_scenarios": len(imputed_scenario_tokens),
        "openscene_context_imputed_frames": imputed_frame_count,
        "converter_manifest": str(converter_manifest),
        "converter_manifest_sha256": sha256_file(converter_manifest),
        "rare_pairs": str(rare_pairs),
        "rare_pairs_sha256": sha256_file(rare_pairs),
        "rare_data_audit": str(rare_audit),
        "rare_data_audit_sha256": sha256_file(rare_audit),
        "rare_filter": str(rare_filter),
        "rare_filter_sha256": sha256_file(rare_filter),
        "asset_config_root": str(asset_config_root),
        "asset_config_tokens": len(assets),
        "lanes": lane_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter-manifest", type=Path, required=True)
    parser.add_argument("--rare-pairs", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--rare-filter", type=Path, required=True)
    parser.add_argument("--asset-config-root", type=Path, required=True)
    parser.add_argument("--expected-lanes", type=int, default=3)
    parser.add_argument("--expected-scenarios", type=int, default=6271)
    parser.add_argument("--expected-imputed-scenarios", type=int, default=82)
    parser.add_argument("--expected-imputed-frames", type=int, default=84)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: value.expanduser().resolve()
        for name, value in {
            "converter_manifest": args.converter_manifest,
            "rare_pairs": args.rare_pairs,
            "rare_audit": args.rare_data_audit,
            "rare_filter": args.rare_filter,
            "asset_config_root": args.asset_config_root,
        }.items()
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    report = audit_scenarios(
        paths["converter_manifest"],
        paths["rare_pairs"],
        paths["rare_audit"],
        paths["rare_filter"],
        paths["asset_config_root"],
        args.expected_lanes,
        args.expected_scenarios,
        args.expected_imputed_scenarios,
        args.expected_imputed_frames,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
