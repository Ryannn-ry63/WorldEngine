#!/usr/bin/env python3
"""Train one frozen CPV E2 common-coverage arm."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_grpo_selector_v3_rare_rollout_data import parse_seed_path
import cpv_e2_common_contract as e2_common
import train_grpo_selector_v3_cached_rare_rollout as trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-cache", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--common-cache", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--common-data-manifest", type=Path, required=True)
    parser.add_argument("--common-arm", choices=e2_common.ARMS, required=True)
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=trainer.FORMAL_EPOCHS)
    parser.add_argument(
        "--examples-per-cache-epoch",
        type=int,
        default=trainer.FORMAL_EXAMPLES_PER_CACHE_EPOCH,
    )
    parser.add_argument("--checkpoint-epochs", default=str(trainer.FORMAL_EPOCHS))
    parser.add_argument("--batch-size", type=int, default=trainer.FORMAL_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--formal-contract", action="store_true")
    args = parser.parse_args()
    args.architecture = "proposal_conditioned_regret_arbitration"
    args.ablation = "relational_only"
    args.objective = trainer.OBJECTIVE_PROPOSAL_REGRET
    args.preference_weight = 0.0
    args.verifier_reward_margin = 0.0
    args.verifier_reward_temperature = 0.05
    args.override_threshold = 0.0
    args.arbiter_loss = "regret"
    args.arbiter_risk = "source_sign"
    args.use_decision_context = False
    args.counterfactual_loss = None
    args.train_evaluator_encoder = False
    args.source_risk = "original_mixture"
    args.data_split = "train"
    args.method_name = None
    args.smoke_limit_hard_pool = None
    return args


def main():
    trainer.train(parse_args())


if __name__ == "__main__":
    main()
