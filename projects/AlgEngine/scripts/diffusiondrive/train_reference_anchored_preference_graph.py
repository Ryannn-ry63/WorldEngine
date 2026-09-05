#!/usr/bin/env python3
"""Train RAPG and its causal controls on the audited rare-rollout mixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_grpo_selector_v3_rare_rollout_data import parse_seed_path
import train_grpo_selector_v3_cached_rare_rollout as trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-cache", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--preference-weight", type=float, default=0.5)
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
    parser.add_argument(
        "--architecture",
        choices=(
            "trajectory_set_reasoner",
            "reference_anchored_preference_graph",
            "capacity_matched_unary",
        ),
        default="reference_anchored_preference_graph",
    )
    parser.add_argument(
        "--ablation",
        choices=("relational_only", "full", "temporal_only", "no_scene_context"),
        default="relational_only",
    )
    parser.add_argument(
        "--objective",
        choices=(trainer.OBJECTIVE_OFFICIAL, trainer.OBJECTIVE_SCALAR_PREFERENCE),
        default=trainer.OBJECTIVE_SCALAR_PREFERENCE,
    )
    parser.add_argument(
        "--data-split",
        choices=("train", "all"),
        default="train",
        help="use train for search; use all only after method/hyperparameters are locked",
    )
    parser.add_argument("--formal-contract", action="store_true")
    parser.add_argument("--smoke-limit-hard-pool", type=int)
    return parser.parse_args()


def main() -> None:
    trainer.train(parse_args())


if __name__ == "__main__":
    main()
