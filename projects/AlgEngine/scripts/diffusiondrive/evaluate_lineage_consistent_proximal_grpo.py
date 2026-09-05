#!/usr/bin/env python3
"""Evaluate an LC-PGRPO checkpoint on its locked log-disjoint split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common
import lcpgrpo_protocol as protocol
import lineage_consistent_proximal_grpo as lcp
import proposal_aware_full_feedback_grpo as categorical
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer


SPLIT_CONTRACTS = {
    "cv": ("train", protocol.FRESH_NOISE_SEEDS),
    "development": ("development", protocol.DEVELOPMENT_NOISE_SEEDS),
    "certification": ("certification", protocol.CERTIFICATION_NOISE_SEEDS),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--mechanism-gate", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_CONTRACTS), required=True)
    parser.add_argument("--heldout-fold", type=int, default=-1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def verify_mechanism_gate(path):
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 2
        or payload.get("status") != "PASS"
        or payload.get("method") != protocol.MECHANISM_METHOD
        or payload.get("decision") != "AUTHORIZE_LCPGRPO_LOG_CV"
        or payload.get("noise_seeds") != list(protocol.FRESH_NOISE_SEEDS)
        or payload.get("prediction_metric_contract", {})
        .get("gate", {})
        .get("metric")
        != "predicted_top1_is_in_heldout_top4_fraction"
    ):
        raise RuntimeError("evaluation requires the locked LC-PGRPO mechanism gate")
    return path, common.sha256_file(path)


def load_models(path, expected_sha, device):
    path = path.expanduser().resolve()
    actual_sha = common.sha256_file(path)
    if expected_sha is not None and actual_sha != expected_sha:
        raise RuntimeError("candidate checkpoint SHA256 drifted")
    payload = torch.load(path, map_location="cpu")
    if payload.get("method") != protocol.METHOD or payload.get("arm") not in lcp.ARMS:
        raise RuntimeError("invalid LC-PGRPO checkpoint")
    candidate = common.model_from_config(payload["scene_selector_config"])
    candidate.load_state_dict(payload["scene_selector_state"], strict=True)
    candidate.to(device).eval()

    anchor_path = Path(payload["v3_anchor"]).expanduser().resolve()
    anchor_sha = common.sha256_file(anchor_path)
    if anchor_sha != payload["v3_anchor_sha256"]:
        raise RuntimeError("V3 anchor SHA256 drifted")
    anchor_payload = torch.load(anchor_path, map_location="cpu")
    anchor = common.model_from_config(anchor_payload["scene_selector_config"])
    anchor.load_state_dict(anchor_payload["scene_selector_state"], strict=True)
    anchor.to(device).eval()
    return (
        candidate,
        payload,
        path,
        actual_sha,
        anchor,
        anchor_payload,
        anchor_path,
        anchor_sha,
    )


def token_logs(pair_rows, tokens):
    mapping = {}
    for row in pair_rows:
        for key in ("rare_token", "common_token"):
            token = str(row[key])
            log_name = str(row["log_name"])
            previous = mapping.setdefault(token, log_name)
            if previous != log_name:
                raise RuntimeError("one token maps to multiple logs")
    try:
        return [mapping[str(token)] for token in tokens]
    except KeyError as error:
        raise RuntimeError(f"cache token absent from pair manifest: {error}")


def evaluate_draw(candidate, anchor, cache, indices, device, temperature, batch_size):
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = torch.as_tensor(
                indices[start : start + batch_size], dtype=torch.long
            )
            current_logits, _ = common.current_logits(
                candidate, cache, batch_indices, device
            )
            anchor_logits, _ = common.current_logits(
                anchor, cache, batch_indices, device
            )
            rewards = cache["candidate_rewards"][batch_indices].to(
                device=device, dtype=torch.float32
            )
            valid = cache["candidate_reward_valid_mask"][batch_indices].to(
                device=device, dtype=torch.bool
            )
            valid = valid & torch.isfinite(rewards)
            current_logp, current_probability = categorical.masked_log_policy(
                current_logits, valid, temperature
            )
            anchor_logp, anchor_probability = categorical.masked_log_policy(
                anchor_logits, valid, temperature
            )
            current_index = current_probability.masked_fill(~valid, -1.0).argmax(
                dim=-1
            )
            anchor_index = anchor_probability.masked_fill(~valid, -1.0).argmax(
                dim=-1
            )
            oracle_reward, oracle_index = rewards.masked_fill(
                ~valid, -torch.inf
            ).max(dim=-1)

            def gather(values, index):
                return values.gather(-1, index[:, None]).squeeze(-1)

            current_reward = gather(rewards, current_index)
            anchor_reward = gather(rewards, anchor_index)
            oracle_probability = gather(anchor_probability, oracle_index)
            headroom = oracle_reward - anchor_reward
            current_kl = (
                current_probability
                * torch.where(
                    valid,
                    current_logp - anchor_logp,
                    torch.zeros_like(current_logp),
                )
            ).sum(dim=-1)
            batch_output = {
                "hard_gain": current_reward - anchor_reward,
                "current_reward": current_reward,
                "anchor_reward": anchor_reward,
                "oracle_reward": oracle_reward,
                "anchor_oracle_headroom": headroom,
                "current_oracle_headroom": oracle_reward - current_reward,
                "anchor_oracle_probability": oracle_probability,
                "recoverable": (headroom > 0.005).float(),
                "suppressed_recoverable": (
                    (headroom > 0.005) & (oracle_probability < 1e-4)
                ).float(),
                "solved": (headroom <= 0.005).float(),
                "selection_changed": current_index.ne(anchor_index).float(),
                "current_to_v3_kl": current_kl,
            }
            output.append({key: value.cpu() for key, value in batch_output.items()})
    return {
        key: torch.cat([batch[key] for batch in output]) for key in output[0]
    }


def make_records(stacked, cache, indices, logs, labels, noise_seeds):
    keys = tuple(stacked)
    rows = []
    for local_index, cache_index in enumerate(indices):
        for draw, noise_seed in enumerate(noise_seeds):
            row = {
                "token": str(cache["tokens"][cache_index]),
                "scene": str(cache["scenes"][cache_index]),
                "log": str(logs[cache_index]),
                "stratum": str(labels[cache_index]),
                "noise_seed": int(noise_seed),
            }
            for key in keys:
                row[key] = float(stacked[key][local_index, draw])
            rows.append(row)
    return rows


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_repetitions <= 0:
        raise ValueError("evaluation budgets must be positive")
    cache_split, expected_seeds = SPLIT_CONTRACTS[args.split]
    if args.split == "cv" and args.heldout_fold not in range(protocol.NUM_FOLDS):
        raise ValueError("CV evaluation requires heldout fold 0..4")
    if args.split != "cv" and args.heldout_fold != -1:
        raise ValueError("development/certification evaluate the complete split")
    gate_path, gate_sha = verify_mechanism_gate(args.mechanism_gate)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    loaded = [common.load_cache(path, cache_split) for path in args.cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    standard.assert_cache_group(caches, manifests, cache_split, expected_seeds)
    pair_rows, rare_audit, pair_path, audit_path = rare_trainer.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    rare_indices, common_indices = rare_trainer.validate_cache_pair_alignment(
        caches, manifests, pair_rows, rare_audit
    )
    fold_contract = protocol.build_fold_contract(pair_rows)
    positions = protocol.pair_positions_for_evaluation(
        fold_contract, args.heldout_fold
    )
    selected_rare = [rare_indices[position] for position in positions]
    selected_common = [common_indices[position] for position in positions]
    selected_rare_set = set(selected_rare)
    selected_common_set = set(selected_common)
    if selected_rare_set.intersection(selected_common_set):
        raise RuntimeError("rare/common evaluation strata overlap")
    # A matched common token may legitimately serve two rare pairs. Evaluate
    # each token once; equal-stratum aggregation below preserves 50/50 weight.
    indices = np.asarray(
        sorted(selected_rare_set.union(selected_common_set)), dtype=np.int64
    )

    labels = np.asarray(["common"] * len(caches[0]["tokens"]), dtype=object)
    labels[np.asarray(rare_indices, dtype=np.int64)] = "rare"
    logs = token_logs(pair_rows, caches[0]["tokens"])
    (
        candidate,
        checkpoint,
        checkpoint_path,
        checkpoint_sha,
        anchor,
        anchor_payload,
        anchor_path,
        anchor_sha,
    ) = load_models(args.checkpoint, args.checkpoint_sha256, device)
    if checkpoint.get("mechanism_gate_sha256") != gate_sha:
        raise RuntimeError("checkpoint/evaluation mechanism gates differ")
    checkpoint_fold_path = Path(checkpoint["fold_contract"]).expanduser().resolve()
    if not checkpoint_fold_path.is_file():
        raise FileNotFoundError(checkpoint_fold_path)
    checkpoint_fold_sha = common.sha256_file(checkpoint_fold_path)
    if checkpoint_fold_sha != checkpoint.get("fold_contract_sha256"):
        raise RuntimeError("checkpoint fold contract SHA256 drifted")
    checkpoint_fold = json.loads(checkpoint_fold_path.read_text())
    if args.split == "cv":
        if checkpoint.get("pair_manifest_sha256") != common.sha256_file(pair_path):
            raise RuntimeError("checkpoint/evaluation pair manifests differ")
        if checkpoint.get("rare_data_audit_sha256") != common.sha256_file(audit_path):
            raise RuntimeError("checkpoint/evaluation rare-data audits differ")
        if checkpoint_fold != fold_contract:
            raise RuntimeError("checkpoint/evaluation fold contracts differ")
        if checkpoint.get("heldout_fold") != args.heldout_fold:
            raise RuntimeError("checkpoint was trained for a different heldout fold")
    elif checkpoint.get("heldout_fold") != -1:
        raise RuntimeError("development/certification require the all-train-log model")

    draws = [
        evaluate_draw(
            candidate,
            anchor,
            cache,
            indices,
            device,
            float(checkpoint["temperature"]),
            args.batch_size,
        )
        for cache in caches
    ]
    stacked = {
        key: torch.stack([draw[key] for draw in draws], dim=1)
        for key in draws[0]
    }
    records = make_records(
        stacked, caches[0], indices, logs, labels, expected_seeds
    )
    aggregate = protocol.aggregate_records(
        records,
        expected_noise_seeds=expected_seeds,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    summary = aggregate["summary"]
    summary.update(
        {
            "equal_stratum_current_pdm": 0.5
            * sum(
                np.mean(
                    [
                        row["current_reward"]
                        for row in records
                        if row["stratum"] == label
                    ],
                    dtype=np.float64,
                )
                for label in ("common", "rare")
            ),
            "equal_stratum_anchor_pdm": 0.5
            * sum(
                np.mean(
                    [
                        row["anchor_reward"]
                        for row in records
                        if row["stratum"] == label
                    ],
                    dtype=np.float64,
                )
                for label in ("common", "rare")
            ),
        }
    )

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records_path = output.with_name(output.stem + "_records.jsonl")
    if records_path.exists():
        raise FileExistsError(records_path)
    with records_path.open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": protocol.EVALUATION_METHOD,
        "arm": checkpoint["arm"],
        "arm_contract": checkpoint["arm_contract"],
        "lineage_control": checkpoint["lineage_control"],
        "target_kl": checkpoint["target_kl"],
        "epoch": checkpoint["epoch"],
        "split": args.split,
        "cache_split": cache_split,
        "heldout_fold": args.heldout_fold,
        "noise_seeds": list(expected_seeds),
        "summary": summary,
        "strata": aggregate["strata"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "v3_anchor": str(anchor_path),
        "v3_anchor_sha256": anchor_sha,
        "v3_anchor_method": anchor_payload.get("method"),
        "mechanism_gate": str(gate_path),
        "mechanism_gate_sha256": gate_sha,
        "cache_manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "fold_contract": fold_contract,
        "num_evaluation_pairs": len(positions),
        "num_evaluation_tokens": len(indices),
        "num_evaluation_rare_tokens": len(selected_rare_set),
        "num_evaluation_common_tokens": len(selected_common_set),
        "checkpoint_fold_contract": str(checkpoint_fold_path),
        "checkpoint_fold_contract_sha256": checkpoint_fold_sha,
        "records": str(records_path),
        "records_sha256": common.sha256_file(records_path),
        "scientific_contract": {
            "selector_only": True,
            "generator_frozen": True,
            "official_scalar_pdm_only": True,
            "new_data_kind_consumed": False,
            "log_disjoint_from_checkpoint_training": args.split == "cv",
            "development_consumed": args.split == "development",
            "certification_consumed": args.split == "certification",
        },
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
            str(Path(protocol.__file__).resolve()): common.sha256_file(
                Path(protocol.__file__).resolve()
            ),
            str(Path(lcp.__file__).resolve()): common.sha256_file(
                Path(lcp.__file__).resolve()
            ),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "arm": checkpoint["arm"],
                "epoch": checkpoint["epoch"],
                "heldout_fold": args.heldout_fold,
                "output": str(output),
                **summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
