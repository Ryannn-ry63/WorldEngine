#!/usr/bin/env python3
"""Train PCRA V2 with a frozen proposal and decoupled pair evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_grpo_selector_v3_rare_rollout_data import parse_seed_path
import train_grpo_selector_v3_cached_rare_rollout as trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-cache", action="append", type=parse_seed_path, required=True
    )
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--counterfactual-loss",
        choices=trainer.COUNTERFACTUAL_LOSSES,
        required=True,
    )
    parser.add_argument(
        "--train-evaluator-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--source-risk", choices=trainer.SOURCE_RISKS, default="original_mixture"
    )
    parser.add_argument("--override-threshold", type=float, default=0.0)
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
    parser.add_argument("--method-name")
    parser.add_argument("--data-split", choices=("train", "all"), default="train")
    parser.add_argument("--formal-contract", action="store_true")
    parser.add_argument("--smoke-limit-hard-pool", type=int)
    args = parser.parse_args()
    args.architecture = "proposal_conditioned_counterfactual_evaluator"
    args.ablation = "relational_only"
    args.objective = trainer.OBJECTIVE_COUNTERFACTUAL_EVALUATION
    args.preference_weight = 0.0
    args.verifier_reward_margin = 0.0
    args.verifier_reward_temperature = 0.05
    args.arbiter_loss = None
    args.use_decision_context = False
    return args


def main():
    trainer.train(parse_args())


if __name__ == "__main__":
    main()
