import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "diffusiondrive"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RESCORE = load_module(
    "rescore_grpo_selector_v2_cache",
    "rescore_grpo_selector_v2_cache.py",
)
TRAIN = load_module(
    "train_grpo_selector_v2_cached_progress_fix",
    "train_grpo_selector_v2_cached.py",
)


def source_cache(count=4):
    generator = torch.Generator().manual_seed(17)
    return {
        "schema_version": 1,
        "tokens": [f"token-{index}" for index in range(count)],
        "scenes": [f"scene-{index // 2}" for index in range(count)],
        "candidate_features": torch.randn(
            count, 20, 256, generator=generator
        ).half(),
        "candidate_trajectories_8": torch.randn(
            count, 20, 8, 3, generator=generator
        ),
        "candidate_rewards": torch.randn(count, 20, generator=generator),
        "candidate_reward_components": torch.randn(
            count, 20, 6, generator=generator
        ),
        "candidate_reward_valid_mask": torch.ones(
            count, 20, dtype=torch.bool
        ),
        "reference_logits": torch.randn(count, 20, generator=generator),
        "current_logits": torch.randn(count, 20, generator=generator),
        "baseline_selector_state": {
            "0.weight": torch.randn(2, 2, generator=generator),
            "0.bias": torch.randn(2, generator=generator),
        },
    }


def reward_part(source, rank, world_size):
    indices = RESCORE.expected_rank_indices(len(source["tokens"]), rank, world_size)
    return {
        "indices": indices,
        "rewards": source["candidate_rewards"][indices] + 0.25,
        "components": source["candidate_reward_components"][indices] + 0.5,
        "valid": source["candidate_reward_valid_mask"][indices].clone(),
    }


def test_reward_only_merge_preserves_every_immutable_value():
    source = source_cache()
    parts = [reward_part(source, rank, 2) for rank in range(2)]
    corrected = RESCORE.build_corrected_cache(source, parts, 4)
    RESCORE.assert_immutable_equal(source, corrected, 4)
    assert RESCORE.immutable_digest(
        RESCORE.subset_immutable_cache(source, 4)
    ) == RESCORE.immutable_digest(corrected)
    assert torch.equal(
        corrected["candidate_rewards"], source["candidate_rewards"] + 0.25
    )


def test_reward_only_merge_rejects_duplicate_or_missing_indices():
    source = source_cache()
    duplicated = reward_part(source, 0, 2)
    with pytest.raises(RuntimeError, match="coverage incomplete"):
        RESCORE.build_corrected_cache(source, [duplicated], 4)
    with pytest.raises(RuntimeError, match="duplicate/out-of-range"):
        RESCORE.build_corrected_cache(source, [duplicated, duplicated], 4)


def test_merge_writes_audited_schema2_cache(tmp_path):
    source = source_cache()
    source_cache_path = tmp_path / "source" / "cache.pt"
    source_cache_path.parent.mkdir()
    torch.save(source, source_cache_path)
    source_cache_sha = RESCORE.sha256_file(source_cache_path)
    source_manifest_path = source_cache_path.parent / "manifest.json"
    source_manifest = {
        "schema_version": 1,
        "status": "PASS",
        "method": RESCORE.SOURCE_METHOD,
        "checkpoint_sha256": "baseline",
        "nav_filter_sha256": "filter",
        "split": "train",
        "noise_seed": 0,
        "num_tokens": 4,
        "cache_sha256": source_cache_sha,
    }
    source_manifest_path.write_text(json.dumps(source_manifest) + "\n")
    source_manifest_sha = RESCORE.sha256_file(source_manifest_path)
    metric_cache = tmp_path / "metric-cache"
    metric_cache.mkdir()
    validator = tmp_path / "validator.py"
    validator.write_text("# pinned validator\n")
    output_dir = tmp_path / "output"
    (output_dir / "parts").mkdir(parents=True)
    reward_sha = RESCORE.sha256_file(RESCORE.reward_implementation_path())
    for rank in range(2):
        part = reward_part(source, rank, 2)
        part.update(
            {
                "status": "PASS",
                "rank": rank,
                "world_size": 2,
                "source_cache_sha256": source_cache_sha,
                "reward_contract": RESCORE.REWARD_CONTRACT,
                "reward_implementation_sha256": reward_sha,
            }
        )
        torch.save(part, output_dir / "parts" / f"rank{rank}.pt")
    args = SimpleNamespace(
        source_cache=source_cache_path,
        source_manifest=source_manifest_path,
        expected_source_cache_sha256=source_cache_sha,
        expected_source_manifest_sha256=source_manifest_sha,
        expected_checkpoint_sha256="baseline",
        expected_nav_filter_sha256="filter",
        expected_split="train",
        expected_noise_seed=0,
        expected_num_tokens=4,
        metric_cache_path=metric_cache,
        validator=validator,
        output_dir=output_dir,
        world_size=2,
        limit=0,
    )
    RESCORE.run_merge(args)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "PASS"
    assert manifest["schema_version"] == 2
    assert manifest["immutable_source_sha256"] == manifest[
        "immutable_output_sha256"
    ]


def test_cached_trainer_rejects_old_schema_before_training(tmp_path):
    cache = source_cache()
    cache_path = tmp_path / "cache.pt"
    torch.save(cache, cache_path)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "method": RESCORE.SOURCE_METHOD,
        "cache_sha256": TRAIN.sha256_file(cache_path),
        "split": "train",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(RuntimeError, match="not corrected schema v2"):
        TRAIN.load_cache(cache_path, "train")


def test_rank_partition_is_complete_and_deterministic():
    partitions = [RESCORE.expected_rank_indices(17, rank, 8) for rank in range(8)]
    assert sorted(index for part in partitions for index in part) == list(range(17))
    assert partitions == [
        RESCORE.expected_rank_indices(17, rank, 8) for rank in range(8)
    ]


def test_dynamic_materialization_contract_accepts_selected_hparams(tmp_path):
    materialize = load_module(
        "materialize_grpo_selector_v2_progress_fix_test",
        "materialize_grpo_selector_v2_replica.py",
    )
    state = tmp_path / "selector.pt"
    payload = {
        "schema_version": 2,
        "method": "exact_group_grpo",
        "selector_state": {
            f"tensor_{index}": torch.tensor([float(index)])
            for index in range(10)
        },
        "temperature": 4.0,
        "learning_rate": 3e-4,
        "kl_weight": 1e-4,
        "train_seed": 2,
        "epoch": 64,
        "reward_contract": RESCORE.REWARD_CONTRACT,
        "reward_implementation_sha256": "reward-sha",
    }
    torch.save(payload, state)
    assert materialize.validate_selector_payload(
        state,
        materialize.sha256_file(state),
        2,
        temperature=4.0,
        learning_rate=3e-4,
        kl_weight=1e-4,
        epoch=64,
        expected_reward_contract=RESCORE.REWARD_CONTRACT,
        expected_reward_sha256="reward-sha",
    ) == materialize.sha256_file(state)


def test_integrated_runner_combines_prepare_and_exposes_parallel_formal_lanes():
    runner = (
        SCRIPT_DIR / "run_grpo_selector_v2_progress_fix_h100.sh"
    ).read_text()
    assert "prepare-seed0" in runner
    assert 'run_grpo_selector_v2_progress_fix_formal_seed_h100.sh\" 0' in runner
    assert 'formal_seed${FORMAL_SEED}' in runner
    assert "wait" not in runner.split('case "${MODE}" in', 1)[-1]


def test_formal_runner_uses_fresh_model_name_and_all_four_blocks():
    runner = (
        SCRIPT_DIR / "run_grpo_selector_v2_progress_fix_formal_seed_h100.sh"
    ).read_text()
    assert "e2e_diffusiondrive_grpo_selector_v2_progress_fix_v1_s${SEED}" in runner
    assert "run_grpo_selector_formal_table_h100.sh" in runner
    assert "selection_mode" in runner


def test_closedloop_runner_propagates_split_count_to_merger():
    ray_runner = (
        SCRIPT_DIR / "run_ray_distributed_testing_diffusiondrive_h100.sh"
    ).read_text()
    merge_script = (
        Path(__file__).resolve().parents[2]
        / "SimEngine/scripts/merge_simulation_results.py"
    ).read_text()
    assert '--num-splits "$SPLIT_COUNT"' in ray_runner
    assert 'parser.add_argument("--num-splits"' in merge_script
    assert "range(NUM_SPLITS)" in merge_script
