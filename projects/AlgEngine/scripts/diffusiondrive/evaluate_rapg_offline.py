#!/usr/bin/env python3
"""Evaluate RAPG checkpoints on log-disjoint common/real-rare/synthetic strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from build_grpo_selector_v3_rare_rollout_data import parse_seed_path
import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached_rare_rollout as trainer


def load_state(path: Path, device: torch.device):
    path = path.expanduser().resolve()
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") not in (3, 4, 5, 6):
        raise RuntimeError(f"invalid selector state schema: {path}")
    config = payload.get("scene_selector_config")
    state = payload.get("scene_selector_state")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise RuntimeError(f"incomplete selector state: {path}")
    model = common.model_from_config(config)
    model.load_state_dict(state, strict=True)
    return path, payload, model.to(device).eval()


def append_metrics(output, logits, reference, rewards, components, valid, temperature):
    metrics = common.selector_metrics(
        logits, reference, rewards, components, valid, temperature
    )
    oracle = rewards.masked_fill(~valid, -torch.inf).max(dim=-1).values
    current = metrics["current_reward"]
    reference_reward = metrics["reference_reward"]
    current_masked = (logits / temperature).masked_fill(~valid, -1e4)
    reference_masked = (reference / temperature).masked_fill(~valid, -1e4)
    current_top2 = current_masked.topk(k=2, dim=-1).values
    reference_top2 = reference_masked.topk(k=2, dim=-1).values
    metrics.update(
        degraded=(current < reference_reward - 1e-8).float(),
        improved=(current > reference_reward + 1e-8).float(),
        epsilon_optimal_001=(oracle - current <= 0.01 + 1e-8).float(),
        epsilon_optimal_003=(oracle - current <= 0.03 + 1e-8).float(),
        epsilon_optimal_005=(oracle - current <= 0.05 + 1e-8).float(),
        selection_margin=current_top2[:, 0] - current_top2[:, 1],
        reference_margin=reference_top2[:, 0] - reference_top2[:, 1],
        _current_index=current_masked.argmax(dim=-1),
        _reference_index=reference_masked.argmax(dim=-1),
    )
    for key, value in metrics.items():
        output.setdefault(key, []).append(value.detach().cpu())


def append_override_metrics(output, diagnostics, rewards):
    proposal = diagnostics["proposal_index"]
    incumbent = diagnostics["incumbent_index"]
    override = diagnostics["override_mask"]
    proposal_reward = rewards.gather(1, proposal[:, None]).squeeze(1)
    incumbent_reward = rewards.gather(1, incumbent[:, None]).squeeze(1)
    gain = proposal_reward - incumbent_reward
    changed_proposal = proposal.ne(incumbent)
    proposal_evidence = diagnostics.get("proposal_evidence")
    if proposal_evidence is None:
        proposal_evidence = diagnostics["verification_logits"].gather(
            1, proposal[:, None]
        ).squeeze(1)
    values = {
        "override": override.float(),
        "proposal_changed": changed_proposal.float(),
        "proposal_evidence": proposal_evidence,
        "override_improved": (override & gain.gt(1e-8)).float(),
        "override_degraded": (override & gain.lt(-1e-8)).float(),
        "proposal_reward": proposal_reward,
        "proposal_top1_gain": gain,
        "beneficial_proposal": (changed_proposal & gain.gt(1e-8)).float(),
        "missed_beneficial_proposal": (
            ~override & changed_proposal & gain.gt(1e-8)
        ).float(),
        "oracle_verified_reward": torch.where(
            changed_proposal & gain.gt(1e-8), proposal_reward, incumbent_reward
        ),
        "accepted_scalar_gain": torch.where(override, gain, torch.zeros_like(gain)),
        "recovered_positive_gain": torch.where(
            override, gain.clamp_min(0.0), torch.zeros_like(gain)
        ),
        "available_positive_gain": torch.where(
            changed_proposal, gain.clamp_min(0.0), torch.zeros_like(gain)
        ),
        "false_accept_regret": torch.where(
            override, (-gain).clamp_min(0.0), torch.zeros_like(gain)
        ),
        "false_reject_regret": torch.where(
            changed_proposal & ~override, gain.clamp_min(0.0), torch.zeros_like(gain)
        ),
    }
    if "opportunity" in diagnostics:
        values.update(
            predicted_opportunity=diagnostics["opportunity"],
            predicted_risk=diagnostics["risk"],
            opportunity_squared_error=(
                diagnostics["opportunity"] - gain.clamp_min(0.0)
            ).square(),
            risk_squared_error=(
                diagnostics["risk"] - (-gain).clamp_min(0.0)
            ).square(),
            signed_gain_squared_error=(
                proposal_evidence - gain
            ).square(),
        )
    for key, value in values.items():
        output.setdefault(key, []).append(value.detach().cpu())


def evaluate_rows(
    model,
    row_specs,
    hard_rows,
    real_cache,
    token_map,
    synthetic,
    device,
    temperature,
    batch_size,
):
    output = {}
    with torch.no_grad():
        for start in range(0, len(row_specs), batch_size):
            inputs, reference, rewards, components, valid, _ = trainer.mixed_batch(
                row_specs[start : start + batch_size],
                hard_rows,
                real_cache,
                token_map,
                common_cache=None,
                common_token_map=None,
                common_tokens=None,
                synthetic=synthetic,
                device=device,
            )
            if getattr(model, "requires_reference_logits", False):
                inputs["reference_logits"] = reference
            if getattr(model, "requires_candidate_mask", False):
                inputs["candidate_mask"] = valid
            if getattr(model, "supports_override_diagnostics", False):
                residual, override_diagnostics = model(
                    **inputs, return_override_diagnostics=True
                )
                logits = reference + residual
            else:
                override_diagnostics = None
                logits = reference + model(**inputs)
            append_metrics(
                output,
                logits,
                reference,
                rewards,
                components,
                valid,
                temperature,
            )
            if override_diagnostics is not None:
                append_override_metrics(output, override_diagnostics, rewards)
    return {key: torch.cat(values) for key, values in output.items()}


def average_ranks(values):
    values = values.detach().cpu().to(torch.float64)
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def pearson_correlation(left, right):
    left = left.detach().cpu().to(torch.float64)
    right = right.detach().cpu().to(torch.float64)
    if len(left) < 2:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) <= 0.0:
        return None
    return float((left * right).sum() / denominator)


def binary_roc_auc(scores, targets):
    scores = scores.detach().cpu().to(torch.float64)
    targets = targets.detach().cpu().bool()
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = ranks[targets].sum()
    return float(
        (
            positive_rank_sum - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


def binary_average_precision(scores, targets):
    scores = scores.detach().cpu().to(torch.float64)
    targets = targets.detach().cpu().bool()
    positives = int(targets.sum())
    if not positives:
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    ordered = targets[order].to(torch.float64)
    precision = ordered.cumsum(0) / torch.arange(
        1, len(ordered) + 1, dtype=torch.float64
    )
    return float(precision[ordered.bool()].mean())


def proposal_ranking_statistics(values):
    if "proposal_evidence" not in values:
        return {}, None
    gain = values["proposal_top1_gain"]
    active = values["proposal_changed"].bool() & gain.abs().gt(1e-8)
    scores = values["proposal_evidence"][active]
    gains = gain[active]
    targets = gains.gt(0.0)
    statistics = {
        "proposal_ranking_examples": int(active.sum()),
        "proposal_benefit_roc_auc": binary_roc_auc(scores, targets),
        "proposal_benefit_pr_auc": binary_average_precision(scores, targets),
        "proposal_evidence_gain_pearson": pearson_correlation(scores, gains),
        "proposal_evidence_gain_spearman": pearson_correlation(
            average_ranks(scores), average_ranks(gains)
        ),
        "proposal_beneficial_ranking_examples": int(targets.sum()),
        "proposal_degrading_ranking_examples": int((~targets).sum()),
    }
    records = {
        "evidence": [float(value) for value in scores],
        "gain": [float(value) for value in gains],
    }
    return statistics, records


def summarize(values):
    public_values = {
        key: value for key, value in values.items() if not key.startswith("_")
    }
    summary = common.summarize(public_values)
    summary.update(
        degraded_fraction=float(values["degraded"].mean()),
        improved_fraction=float(values["improved"].mean()),
        epsilon_optimal_hit_001=float(values["epsilon_optimal_001"].mean()),
        epsilon_optimal_hit_003=float(values["epsilon_optimal_003"].mean()),
        epsilon_optimal_hit_005=float(values["epsilon_optimal_005"].mean()),
        selection_margin_cdf_le_001=float(
            (values["selection_margin"] <= 0.01).float().mean()
        ),
        selection_margin_cdf_le_003=float(
            (values["selection_margin"] <= 0.03).float().mean()
        ),
        selection_margin_cdf_le_005=float(
            (values["selection_margin"] <= 0.05).float().mean()
        ),
        examples=int(len(values["current_reward"])),
    )
    if "override" in values:
        override_count = float(values["override"].sum())
        available_positive_gain = float(values["available_positive_gain"].sum())
        recovered_positive_gain = float(values["recovered_positive_gain"].sum())
        false_accept_regret = float(values["false_accept_regret"].sum())
        false_reject_regret = float(values["false_reject_regret"].sum())
        accepted_harm = values["false_accept_regret"]
        accepted_harm = accepted_harm[accepted_harm.gt(0.0)]
        worst_count = max(1, (len(accepted_harm) + 9) // 10)
        worst_decile_harm = (
            float(accepted_harm.topk(worst_count).values.mean())
            if len(accepted_harm)
            else 0.0
        )
        summary.update(
            override_coverage=float(values["override"].mean()),
            override_precision=(
                float(values["override_improved"].sum()) / override_count
                if override_count
                else 1.0
            ),
            override_degradation_rate=(
                float(values["override_degraded"].sum()) / override_count
                if override_count
                else 0.0
            ),
            positive_oracle_gain_recovery=(
                recovered_positive_gain / available_positive_gain
                if available_positive_gain
                else 1.0
            ),
            net_accepted_scalar_gain=float(values["accepted_scalar_gain"].sum()),
            false_accept_regret=false_accept_regret,
            false_reject_regret=false_reject_regret,
            mean_false_accept_regret=float(values["false_accept_regret"].mean()),
            mean_false_reject_regret=float(values["false_reject_regret"].mean()),
            accepted_harm_examples=int(len(accepted_harm)),
            worst_decile_accepted_harm=worst_decile_harm,
            available_positive_gain=available_positive_gain,
            recovered_positive_gain=recovered_positive_gain,
            mean_proposal_evidence=float(values["proposal_evidence"].mean()),
        )
        if "predicted_opportunity" in values:
            changed = values["proposal_changed"].bool()
            summary.update(
                mean_predicted_opportunity=float(
                    values["predicted_opportunity"].mean()
                ),
                mean_predicted_risk=float(values["predicted_risk"].mean()),
                changed_opportunity_mse=(
                    float(values["opportunity_squared_error"][changed].mean())
                    if changed.any() else 0.0
                ),
                changed_risk_mse=(
                    float(values["risk_squared_error"][changed].mean())
                    if changed.any() else 0.0
                ),
                changed_signed_gain_mse=(
                    float(values["signed_gain_squared_error"][changed].mean())
                    if changed.any() else 0.0
                ),
            )
    return summary


def noise_view_agreement(groups):
    if len(groups) < 2:
        return None
    lengths = {len(group["_current_index"]) for group in groups}
    if len(lengths) != 1:
        raise RuntimeError("noise-view agreement groups are not aligned")

    def agreement(key):
        values = [group[key] for group in groups]
        pairwise = []
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                pairwise.append(values[left].eq(values[right]).float().mean())
        all_views = torch.stack(values).eq(values[0]).all(dim=0).float().mean()
        return {
            "all_views": float(all_views),
            "mean_pairwise": float(torch.stack(pairwise).mean()),
        }

    return {
        "current_selection": agreement("_current_index"),
        "reference_selection": agreement("_reference_index"),
        "num_noise_views": len(groups),
    }


def evaluate_checkpoint(
    model,
    hard_rows,
    real_caches,
    token_maps,
    synthetic,
    device,
    temperature,
    batch_size,
):
    positions = list(range(len(hard_rows)))
    real_positions = [
        index for index, row in enumerate(hard_rows) if row["hard_kind"] == "real_rare"
    ]
    synthetic_positions = [
        index
        for index, row in enumerate(hard_rows)
        if row["hard_kind"] == "synthetic_rollout"
    ]
    strata_values = {"common": [], "real_rare": [], "synthetic": []}
    for real_cache, token_map in zip(real_caches, token_maps):
        strata_values["common"].append(
            evaluate_rows(
                model,
                [("common", index) for index in positions],
                hard_rows,
                real_cache,
                token_map,
                synthetic,
                device,
                temperature,
                batch_size,
            )
        )
        strata_values["real_rare"].append(
            evaluate_rows(
                model,
                [("hard", index) for index in real_positions],
                hard_rows,
                real_cache,
                token_map,
                synthetic,
                device,
                temperature,
                batch_size,
            )
        )
    strata_values["synthetic"].append(
        evaluate_rows(
            model,
            [("hard", index) for index in synthetic_positions],
            hard_rows,
            real_caches[0],
            token_maps[0],
            synthetic,
            device,
            temperature,
            batch_size,
        )
    )
    strata = {}
    for name, groups in strata_values.items():
        pooled = {key: torch.cat([group[key] for group in groups]) for key in groups[0]}
        ranking_statistics, ranking_records = proposal_ranking_statistics(pooled)
        strata[name] = {
            **summarize(pooled),
            **ranking_statistics,
            "proposal_ranking_records": ranking_records,
            "noise_view_agreement": noise_view_agreement(groups),
        }
    equal_weight_selected_reward = sum(
        row["current_reward"] for row in strata.values()
    ) / len(strata)
    equal_weight_top1_gain = sum(
        row["top1_reward_gain"] for row in strata.values()
    ) / len(strata)
    return {
        "strata": strata,
        "equal_weight_selected_reward": equal_weight_selected_reward,
        "equal_weight_top1_gain": equal_weight_top1_gain,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-cache", action="append", type=parse_seed_path, required=True
    )
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--split", choices=("development", "certification"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.temperature <= 0.0 or args.batch_size <= 0:
        raise ValueError("invalid evaluation temperature/batch size")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    (
        manifest_path,
        manifest,
        hard_pool_path,
        all_rows,
        real_caches,
        real_manifests,
        token_maps,
        synthetic_path,
        synthetic,
    ) = trainer.load_contract(args)
    hard_rows = [row for row in all_rows if row["split"] == args.split]
    expected = manifest["split_counts"][args.split]
    counts = {
        "hard": len(hard_rows),
        "real_rare": sum(row["hard_kind"] == "real_rare" for row in hard_rows),
        "synthetic": sum(row["hard_kind"] == "synthetic_rollout" for row in hard_rows),
        "logs": len({row["log_name"] for row in hard_rows}),
    }
    if counts != {key: int(expected[key]) for key in counts}:
        raise RuntimeError(f"held-out split count drifted: {counts} != {expected}")

    checkpoints = []
    for checkpoint in args.checkpoint:
        path, payload, model = load_state(checkpoint, device)
        checkpoints.append(
            {
                "checkpoint": str(path),
                "checkpoint_sha256": common.sha256_file(path),
                "method": payload["method"],
                "selector_architecture": payload.get("selector_architecture"),
                "ablation": payload.get("ablation"),
                "objective": payload.get("objective"),
                "preference_weight": float(payload.get("preference_weight", 0.0)),
                "arbiter_loss": payload.get("arbiter_loss"),
                "arbiter_risk": payload.get("arbiter_risk", "global"),
                "use_decision_context": payload.get("use_decision_context"),
                "arbiter_target": payload.get("arbiter_target"),
                "counterfactual_loss": payload.get("counterfactual_loss"),
                "train_evaluator_encoder": payload.get("train_evaluator_encoder"),
                "source_risk": payload.get("source_risk"),
                "evaluator_initialization_max_abs_delta": payload.get(
                    "evaluator_initialization_max_abs_delta"
                ),
                "proposal_checkpoint": payload.get("proposal_checkpoint"),
                "proposal_checkpoint_sha256": payload.get("proposal_checkpoint_sha256"),
                "reward_components_consumed": payload.get("reward_components_consumed"),
                "training_data_split": payload.get("training_data_split"),
                "training_epochs": payload.get("training_epochs"),
                "training_examples_per_cache_epoch": payload.get(
                    "training_examples_per_cache_epoch"
                ),
                "training_batch_size": payload.get("training_batch_size"),
                "formal_contract": payload.get("formal_contract"),
                "temperature": payload.get("temperature"),
                "learning_rate": payload.get("learning_rate"),
                "kl_weight": payload.get("kl_weight"),
                "sampling_mode": payload.get("sampling_mode"),
                "common_arm": payload.get("common_arm"),
                "common_unique_tokens": payload.get("common_unique_tokens"),
                "common_data_manifest": payload.get("common_data_manifest"),
                "common_data_manifest_sha256": payload.get(
                    "common_data_manifest_sha256"
                ),
                "common_caches": payload.get("common_caches"),
                "verifier_reward_margin": payload.get("verifier_reward_margin"),
                "verifier_reward_temperature": payload.get(
                    "verifier_reward_temperature"
                ),
                "override_threshold": payload.get("override_threshold"),
                "train_seed": int(payload["train_seed"]),
                "epoch": int(payload["epoch"]),
                **evaluate_checkpoint(
                    model,
                    hard_rows,
                    real_caches,
                    token_maps,
                    synthetic,
                    device,
                    args.temperature,
                    args.batch_size,
                ),
            }
        )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "rapg_log_disjoint_offline_evaluation_v1",
        "split": args.split,
        "certification_consumed": args.split == "certification",
        "three_equal_weight_strata": ["common", "real_rare", "synthetic"],
        "split_counts": counts,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(hard_pool_path),
        "hard_pool_sha256": common.sha256_file(hard_pool_path),
        "synthetic_cache": str(synthetic_path),
        "synthetic_cache_sha256": common.sha256_file(synthetic_path),
        "real_cache_manifests": list(real_manifests),
        "checkpoints": checkpoints,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
