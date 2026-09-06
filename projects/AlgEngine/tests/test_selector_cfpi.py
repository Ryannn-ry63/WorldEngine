"""CFPI semantics, isolation and synthetic pipeline tests (no simulator/GPU)."""
import ast
import csv
import copy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/diffusiondrive"
sys.path.insert(0, str(SCRIPTS))
import cfpi_common as c
import prepare_selector_cfpi as prep
import selector_cfpi_objectives as obj
import selector_cfpi_model as model_api
import report_selector_cfpi as reports
import run_selector_cfpi as runner
from materialize_selector_cfpi import replace_selector
import assemble_selector_cfpi_cache as assembler
import train_selector_cfpi as trainer

torch.set_num_threads(2)


def test_weighted_ce_low_probability_gradient():
    z = torch.tensor([[12., -12., 0.]], requires_grad=True)
    q = torch.tensor([[0., 1., .1]])
    b = torch.tensor([0])
    loss = obj.return_weighted_classification(z, q, b, delta=0)
    loss.backward()
    assert z.grad[0, 1] < -.99
    other = z.detach().requires_grad_()
    obj.exact_group(other, q).backward()
    assert abs(other.grad[0, 1]) < 1e-8


def test_ce_shift_permutation_and_training_constant():
    torch.manual_seed(0)
    z, q = torch.randn(4, 20), torch.rand(4, 20)
    b = z.argmax(1)
    p = torch.randperm(20)
    inverse = torch.argsort(p)
    a = obj.return_weighted_classification(z, q, b, normalizer=3)
    assert torch.allclose(a, obj.return_weighted_classification(z, q+3, b, normalizer=3))
    assert torch.allclose(a, obj.return_weighted_classification(z[:, p], q[:, p], inverse[b], normalizer=3))
    w = obj.classification_weights(q, b, delta=0)
    assert torch.allclose(obj.classification_weights(2*q, b, delta=0), 2*w)
    assert not torch.allclose(w.sum(1), torch.ones(4))


def test_flat_ties_and_expected_return_ranking():
    z = torch.zeros(2, 3, requires_grad=True)
    q = torch.ones(2, 3)
    b = torch.tensor([0, 1])
    zero = obj.return_weighted_classification(z, q, b, delta=0)
    assert zero.item() == 0
    zero.backward()
    assert torch.equal(z.grad, torch.zeros_like(z))
    z.grad.zero_()
    obj.return_weighted_classification(z, q, b, delta=.01).backward()
    assert z.grad[0, 0] < 0 and z.grad[1, 1] < 0
    # Most samples prefer A by .02; a rarer large loss makes B better in expectation.
    q = torch.tensor([[.52, .50]] * 9 + [[0., 1.]])
    weights = obj.classification_weights(q, torch.zeros(10, dtype=torch.long), delta=0)
    assert weights.mean(0).argmax() == q.mean(0).argmax() == 1
    biased = obj.classification_weights(q, torch.zeros(10, dtype=torch.long), delta=.01)
    assert biased.mean(0).argmax() == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_losses_fail(bad):
    with pytest.raises(ValueError):
        obj.return_weighted_classification(torch.zeros(1, 2), torch.tensor([[bad, 0.]]),
                                           torch.tensor([0]))
    with pytest.raises(ValueError):
        obj.return_weighted_classification(torch.zeros(1, 2), torch.zeros(1, 2),
                                           torch.tensor([0]), normalizer=bad)


def target_rows(per_group=20):
    rows = []
    for f, family in enumerate(c.ALLOWED_FAMILIES):
        for o, outcome in enumerate(("failed", "solved")):
            for index in range(per_group):
                log = f"log-{f}-{o}-{index}"
                rows.append(dict(scene_id=log + "-0123456789abcdef", origin_log=log,
                                 scenario_family=family, outcome_stratum=outcome,
                                 all_decisions=list(range(4, 12)), near_decisions=[9, 10, 11]))
    return rows


def test_strata_are_exact_and_q_blind():
    rows = target_rows()
    a = prep.select_stratified(rows)
    for row in rows:
        row["candidate_rewards"] = np.random.randn(20).tolist()
        row["branch_returns"] = np.random.randn(20).tolist()
    b = prep.select_stratified(list(reversed(rows)))
    fields = lambda values: [(r["scene_id"], r["decision_step"], r["time_stratum"]) for r in values]
    assert fields(a) == fields(b)
    assert len(a) == 64 and len({r["scene_id"] for r in a}) == 64
    for family in c.ALLOWED_FAMILIES:
        near = lambda outcome: sorted(r["decision_step"] for r in a if r["scenario_family"] == family
                                    and r["outcome_stratum"] == outcome and r["time_stratum"] == "near")
        assert near("failed") == near("solved")
    mapping = prep.assign_folds(a)
    assert set(mapping.values()) == set(range(4))
    assert all(mapping[r["origin_log"]] == r["fold"] for r in a)
    with pytest.raises(RuntimeError, match="Insufficient target cell"):
        prep.select_stratified(target_rows(8))


def test_global_log_cap_and_no_silent_source_reassignment():
    rows = target_rows()
    for row in rows:
        row["origin_log"] = "one-log"
    with pytest.raises(RuntimeError):
        prep.select_stratified(rows)
    source = dict(id="log-0123456789abcdef", token="0123456789abcdef", metadata={})
    with pytest.raises(ValueError):
        c.annotate_scenario(source, "rare_union", "validation")


def test_joint_source_repairs_greedy_starvation_without_relaxing(monkeypatch):
    rare, common = c.ALLOWED_FAMILIES
    monkeypatch.setattr(c, "stable_digest", lambda *parts: parts[-1])
    candidates = {
        rare: [f"a-shared-{i:016x}" for i in range(2)] + [f"z-spare-{i:016x}" for i in range(2, 4)],
        common: [f"a-shared-{i:016x}" for i in range(4, 6)],
    }
    selected, audit = prep.select_source_joint(candidates, per_family=2, maximum_per_log=2)
    assert audit["legacy_greedy_counts"] == {rare: 2, common: 0}
    assert audit["selected_counts"] == {rare: 2, common: 2}
    assert audit["actual_maximum_per_log"] == 2 and audit["augmentations"] > 0
    assert all(c.scene_origin_log(s) == "z-spare" for s in selected[rare])
    assert prep.select_source_joint({f: list(reversed(v)) for f, v in reversed(list(candidates.items()))}, 2, 2) == (selected, audit)
    impossible = {rare: candidates[rare][:2], common: candidates[common]}
    with pytest.raises(RuntimeError, match="Joint clean source infeasible"):
        prep.select_source_joint(impossible, 2, 2)
    with pytest.raises(RuntimeError, match="overlapping"):
        prep.select_source_joint({rare: candidates[rare], common: candidates[rare]}, 2, 2)


def test_joint_source_matches_exhaustive_small_capacity_problems():
    # 729 independent capacity patterns, compared with exhaustive feasible totals.
    for numbers in itertools.product(range(3), repeat=6):
        candidates = {family: [f"log-{log}-{(100*f + 10*log + j):016x}"
                              for log in range(3) for j in range(numbers[3*f + log])]
                      for f, family in enumerate(c.ALLOWED_FAMILIES)}
        reachable = {(0, 0)}
        for log in range(3):
            reachable = {(a+x, b+y) for a, b in reachable
                         for x in range(numbers[log] + 1)
                         for y in range(numbers[3+log] + 1)
                         if x+y <= 2 and a+x <= 2 and b+y <= 2}
        if (2, 2) not in reachable:
            with pytest.raises(RuntimeError, match="Joint clean source infeasible"):
                prep.select_source_joint(candidates, 2, 2)
        else:
            selected, audit = prep.select_source_joint(candidates, 2, 2)
            assert all(len(selected[f]) == 2 and set(selected[f]) <= set(candidates[f])
                       for f in c.ALLOWED_FAMILIES)
            assert audit["actual_maximum_per_log"] <= 2


def test_source_pool_joint_allocation_filtering_materialization_and_resume(tmp_path, monkeypatch):
    import prepare_selector_v4_causal_source as upstream
    rare, common = c.ALLOWED_FAMILIES
    candidates = {rare: [("a-shared", 20), ("a-shared", 20), ("z-spare", 20),
                          ("z-spare", 20), ("short", 19), ("excluded", 20)],
                  common: [("a-shared", 20), ("a-shared", 20)]}
    root = tmp_path / "upstream"
    tokens, logs = {}, set()
    for f, family in enumerate(c.ALLOWED_FAMILIES):
        content = {}
        for i, (log, length) in enumerate(candidates[family]):
            token = f"{100*f+i:016x}"
            scene = f"{log}-{token}"
            tokens[token], _ = "train", logs.add(log)
            content[scene] = dict(id=scene, token=token, log_length=length, metadata=dict(
                nuplan_lidar_pc_tokens=[token], openscene_data_infos_dict={token: dict(log_name=log)}))
        shard = root / "p2_shards_v1" / family / "shard_000.pkl"
        c.atomic_pickle(shard, content)
        c.atomic_json(shard.with_name("index.json"), dict(scenario_family=family, scenario_variant="original",
            revision="diffusiondrive_grpo_v4_p2_shards_v1", shards=[dict(path=str(shard), scenario_ids=list(content))]))
    monkeypatch.setattr(upstream, "validate_split_contract", lambda path: ({}, tokens,
        dict(train=logs, validation=set(), test=set())))
    monkeypatch.setattr(prep, "historical_exposure", lambda *args: dict(fully_unseen_claim_authorized=False))
    monkeypatch.setattr(c, "TRAIN_POOL_PER_FAMILY", 2)
    monkeypatch.setattr(c, "MAXIMUM_POOL_SCENES_PER_LOG", 2)
    monkeypatch.setattr(c, "stable_digest", lambda *parts: parts[-1])
    exclusions = tmp_path / "exclusions.json"
    c.atomic_json(exclusions, dict(excluded_origin_logs=["excluded"] + [f"excluded-{i}" for i in range(14)],
                                  historical_exposure_status="synthetic"))
    args = SimpleNamespace(run_root=tmp_path / "run", source_root=root, exclusions=exclusions)
    audit = prep.source_pool(args)
    assert audit["train_scenario_count"] == 4
    assert audit["source_selection"]["eligible_counts"] == {rare: 4, common: 2}
    assert audit["source_selection"]["legacy_greedy_counts"] == {rare: 2, common: 0}
    payload = c.load_pickle(audit["train_scenario_file"])
    assert all(c.source_metadata(row)["split"] == "train" for row in payload.values())
    assert all(row["log_length"] >= 20 and c.scene_origin_log(scene) != "excluded"
               for scene, row in payload.items())
    assert prep.source_pool(args) == audit
    c.atomic_pickle(shard, dict(changed=True))
    with pytest.raises(RuntimeError, match="shard drifted"):
        prep.source_pool(args)


def small_model():
    return model_api.v3.SceneConditionedTrajectorySetSelector(
        model_dim=16, geometry_hidden_dim=16, num_heads=4, feedforward_dim=32)


def visible_row(index=0):
    row = {key: np.zeros(shape, dtype=np.float32) for key, shape in model_api.INPUTS.values()}
    row["reference_logits"] = np.arange(20, dtype=np.float32) / 20
    row["v3_logits"] = row["reference_logits"].copy()
    return row


def test_visible_interface_and_score_modes(tmp_path):
    model = small_model().eval()
    rows = [visible_row()]
    a = model_api.score(model, rows, [0], "cpu", "residual")
    rows[0].update(branch_returns=np.full(20, 1e9), local_official_pdm=np.full(20, -1e9),
                   behavior_success=False, origin_log="privileged-log", fold=99)
    b = model_api.score(model, rows, [0], "cpu", "residual")
    assert torch.equal(a, b)
    q_model = model_api.initialize(model, "direct_q", torch.full((1, 20), .4))
    q = model_api.score(q_model, rows, [0], "cpu", "direct_q")
    assert torch.allclose(q, torch.full((1, 20), .4))
    path = tmp_path / "selector.pt"
    model_api.save_selector(path, model, model.config_dict(), "residual", {})
    restored, payload = model_api.load_selector(path)
    assert torch.equal(a, model_api.score(restored, rows, [0], "cpu", payload["score_mode"]))


def test_export_preserves_nonselector_tensors():
    checkpoint = {"state_dict": {"perception.w": torch.randn(3),
                                 "planning_head.scene_selector.w": torch.zeros(2)}}
    original = checkpoint["state_dict"]["perception.w"].clone()
    output, count = replace_selector(checkpoint, {"w": torch.ones(2)})
    assert count == 1 and torch.equal(output["state_dict"]["perception.w"], original)
    with pytest.raises(RuntimeError):
        replace_selector(checkpoint, {"wrong": torch.ones(2)})


def test_locked_json_and_legacy_cache_rejection(tmp_path):
    path = tmp_path / "contract.json"
    c.locked_json(path, dict(code="a"))
    c.locked_json(path, dict(code="a"))
    with pytest.raises(RuntimeError):
        c.locked_json(path, dict(code="b"))
    with pytest.raises(RuntimeError):
        c.validate_cache(dict(method="diffusiondrive_selector_v4_causal_cache_v1"))
    assert not set(runner.STAGES) & {"all", "expand192", "dev64", "test"}


def test_cfpi_config_is_train_only(monkeypatch):
    env = dict(DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED="0",
               DIFFUSIONDRIVE_CFPI_COLLECTION_ID="repeat8_arm_19",
               DIFFUSIONDRIVE_CFPI_SOURCE_SPLIT="train",
               DIFFUSIONDRIVE_CFPI_INTERVENTION_MODE="one_shot_manifest",
               DIFFUSIONDRIVE_CFPI_TREATMENT_MANIFEST_SHA256="a"*64,
               DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256=c.CHECKPOINT_SHA256,
               DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256="b"*64,
               DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256="c"*64,
               DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE=c.NOISE_NAMESPACE)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    path = SCRIPTS.parents[1] / "configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_cfpi_collect.py"
    config = runpy.run_path(str(path))
    assert not config["selector_rollout_contract"]["inference_uses_reward_or_q"]
    assert config["model"]["planning_head"]["online_reward"] is None
    from mmcv import Config
    # CPU syntax/pretty_text check only. GPU preflight also imports planner plugins.
    parsed = Config.fromfile(str(path), import_custom_modules=False)
    assert parsed.model.planning_head.export_rollout_context
    assert "re.Match" not in parsed.pretty_text
    monkeypatch.setenv("DIFFUSIONDRIVE_CFPI_COLLECTION_ID", "dev64_arm_00")
    with pytest.raises(RuntimeError):
        runpy.run_path(str(path))


def test_one_shot_manager_restores_incumbent():
    # Execute the real before_step implementation with only environment I/O mocked.
    path = runner.ROOT / "projects/SimEngine/worldengine/manager/diffusiondrive_cfpi_cache_manager.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    before = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "before_step")
    scope = dict(np=np, cfpi=c, r15=prep.r15,
                 DiffusionDriveDynamicRewardManager=SimpleNamespace(before_step=lambda self: None),
                 should_score_preaction=lambda *args: True, logger=SimpleNamespace(info=lambda *a: None))
    exec(compile(ast.Module(body=[before], type_ignores=[]), str(path), "exec"), scope)
    trajectories = np.zeros((20, 40, 3))
    trajectories[:, :, 0] = np.arange(20)[:, None]
    candidates = np.zeros((20, 8, 3))
    logits = np.arange(20, dtype=float)
    sidecar = dict(planner_step=4, candidate_trajectories_8=candidates, current_logits=logits,
                   selected_index=19, deployed_trajectory=trajectories[19],
                   selector_rollout_contract=dict(action_score_timing="pre_action",
                       intervention_mode="one_shot_manifest", behavior_policy_train_seed=0,
                       treatment_manifest_sha256="x", collection_id="pilot64_arm_02"))
    reward = np.arange(20, dtype=float) / 20
    engine = SimpleNamespace(episode_step=3, external_actions=trajectories[19])
    state = SimpleNamespace(engine=engine, num_history=3, buffer_size=10,
        intervention_mode="one_shot_manifest", behavior_train_seed=0, target_manifest_sha256="x",
        collection_id="pilot64_arm_02", current_scene={"id": "scene"}, _one_shot_applied=False,
        targets={"scene": dict(decision_step=4, treatment_index=2, candidate_rewards=reward,
                              candidate_trajectories_8=candidates, current_logits=logits)},
        _load_sidecar=lambda: (sidecar, "/unused"),
        _prepare_preaction_scores=lambda c: ((reward, np.zeros((20, 6))), trajectories),
        _candidate_action=lambda t, i: t[i], _write_cfpi_record=lambda *args: None)
    # The actual action container is complex; compare equivalent arrays in this test.
    old = prep.r15.trajectory_action_error
    prep.r15.trajectory_action_error = lambda a,b: float(np.max(np.abs(a-b)))
    try:
        scope["before_step"](state)
        assert state._one_shot_applied and np.array_equal(engine.external_actions, trajectories[2])
        engine.episode_step, sidecar["planner_step"] = 4, 5
        engine.external_actions = trajectories[19]
        scope["before_step"](state)
        assert np.array_equal(engine.external_actions, trajectories[19])
    finally:
        prep.r15.trajectory_action_error = old


def synthetic_cache(tmp_path):
    """A label-bearing synthetic cache, never a real driving experiment."""
    model = small_model().eval()
    rows = []
    for i in range(64):
        row = visible_row(i)
        row["candidate_features"][:, 0] = np.arange(20) / 20
        row["candidate_features"][:, 1] = (i % 4) / 4
        q = np.linspace(.1, .7, 20, dtype=np.float32)
        q[i % 18] = .9
        row.update(scene_id=f"synthetic-log-{i // 2}-{i:016x}", origin_log=f"synthetic-log-{i // 2}",
                   origin_token=f"{i:016x}", split="train", fold=(i // 2) % 4, policy_index=19,
                   scenario_family=c.ALLOWED_FAMILIES[i % 2], behavior_success=bool(i % 2),
                   behavior_score=float(q[19]), first_violation_step=None if i % 2 else 12,
                   behavior_outcome_stratum="solved" if i % 2 else "failed", time_stratum="ordinary",
                   branch_returns=q, local_official_pdm=np.linspace(.1, .7, 20, dtype=np.float32),
                   causal_outcome_components=np.ones((20, 3), dtype=np.float32),
                   branch_success=[bool(i % 2)] * 20,
                   branch_first_violation_steps=[None if i % 2 else 12] * 20)
        rows.append(row)
    payload = dict(schema_version=c.SCHEMA_VERSION, num_candidates=20,
                   method=c.CACHE_METHOD, design_version=c.DESIGN_VERSION, stage="pilot64",
                   status="PASS", checkpoint_sha256=c.CHECKPOINT_SHA256, selector_state_sha256="s"*64,
                   candidate_noise_namespace=c.NOISE_NAMESPACE, num_rows=64, rows=rows,
                   collection_code_sha="c"*64, rollout_implementation_sha256="d"*64,
                   development_consumed=False, test_consumed=False)
    return model, payload


@pytest.mark.parametrize("method", ["q_ce_d001", "q_mse"])
def test_cpu_training_export_and_resume(tmp_path, method, monkeypatch):
    model, payload = synthetic_cache(tmp_path)
    cache = tmp_path / "pilot64_cache.pkl"
    c.atomic_pickle(cache, payload)
    cache_audit = tmp_path / "cache_audit.json"
    c.atomic_json(cache_audit, dict(status="PASS", cache_file=str(cache), cache_file_sha256=c.sha256_file(cache)))
    gate = tmp_path / "pilot_gate.json"
    c.atomic_json(gate, dict(status="PASS", training_authorized=True, cache_sha256=c.sha256_file(cache)))
    incumbent = tmp_path / "incumbent.pt"
    torch.save(dict(scene_selector_config=model.config_dict(), scene_selector_state=model.state_dict()), incumbent)
    manifest = tmp_path / "checkpoint_manifest.json"
    c.atomic_json(manifest, dict(status="PASS", checkpoint_sha256=c.CHECKPOINT_SHA256,
                  scene_selector_state=str(incumbent), scene_selector_state_sha256=c.sha256_file(incumbent)))
    args = SimpleNamespace(cache=cache, cache_audit=cache_audit, pilot_gate=gate,
                           checkpoint_manifest=manifest, output_root=tmp_path / "cv",
                           method=method, lr=3e-5, fold=0, seed=0, device="cpu")
    if method == "q_ce_d001":
        original_score = model_api.score
        calls = [0]

        def interrupted_score(module, *values, **kwargs):
            if module.training:
                calls[0] += 1
                if calls[0] == 81:
                    raise RuntimeError("synthetic training interruption")
            return original_score(module, *values, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(model_api, "score", interrupted_score)
            with pytest.raises(RuntimeError, match="synthetic training interruption"):
                trainer.train(args)
        job = args.output_root / trainer.job_name(method, args.lr, args.fold, args.seed)
        state = torch.load(job / "resume.pt", map_location="cpu")
        assert state["step"] == 75 and not (job / "report.json").exists()
    report = trainer.train(args)
    assert [r["step"] for r in report["checkpoints"]] == [50, 150, 500]
    assert report["heldout_scene_count"] == 16 and report["train_scene_count"] == 48
    assert report["score_mode"] == ("direct_q" if method == "q_mse" else "residual")
    assert trainer.train(args) == report
    for checkpoint in report["checkpoints"]:
        loaded, metadata = model_api.load_selector(checkpoint["checkpoint"])
        ids = [i for i, row in enumerate(payload["rows"]) if row["fold"] == 0]
        actual = model_api.score(loaded, payload["rows"], ids, "cpu", metadata["score_mode"]).argmax(1).tolist()
        assert actual == [checkpoint["predictions"][payload["rows"][i]["scene_id"]] for i in ids]
    if method == "q_ce_d001":
        uninterrupted = trainer.train(SimpleNamespace(**dict(vars(args), output_root=tmp_path / "uninterrupted")))
        for resumed, fresh in zip(report["checkpoints"], uninterrupted["checkpoints"]):
            assert resumed["predictions"] == fresh["predictions"]
            resumed_model, _ = model_api.load_selector(resumed["checkpoint"])
            fresh_model, _ = model_api.load_selector(fresh["checkpoint"])
            assert all(torch.equal(value, fresh_model.state_dict()[key])
                       for key, value in resumed_model.state_dict().items())


def test_pilot_gate_fixed_condition_repeat_and_event_drift(tmp_path, monkeypatch):
    _, payload = synthetic_cache(tmp_path)
    target_path = tmp_path / "targets.json"
    repeat_ids = [row["scene_id"] for row in payload["rows"][:8]]
    c.atomic_json(target_path, dict(repeat_scene_ids=repeat_ids))
    payload.update(target_manifest=str(target_path), target_manifest_sha256=c.sha256_file(target_path))
    repeat = dict(payload, stage="repeat8", num_rows=8, rows=copy.deepcopy(payload["rows"][:8]))
    monkeypatch.setattr(c, "load_target_manifest", lambda path: (path, {"repeat_scene_ids": repeat_ids}))
    for stage, value in (("pilot64", payload), ("repeat8", repeat)):
        path = tmp_path / f"cache/{stage}_cache.pkl"
        c.atomic_pickle(path, value)
        c.atomic_json(tmp_path / f"cache/{stage}_cache_audit.json",
                      dict(status="PASS", cache_file=str(path), cache_file_sha256=c.sha256_file(path)))
    gate = reports.pilot_gate(tmp_path)
    assert gate["training_authorized"] and not gate["expansion_authorized"]
    assert gate["improvable_above_0p02"] == 64
    repeat["rows"][0]["branch_first_violation_steps"][0] = 11
    path = tmp_path / "cache/repeat8_cache.pkl"
    c.atomic_pickle(path, repeat)
    c.atomic_json(tmp_path / "cache/repeat8_cache_audit.json",
                  dict(status="PASS", cache_file=str(path), cache_file_sha256=c.sha256_file(path)))
    with pytest.raises(RuntimeError, match="event timing"):
        reports.pilot_gate(tmp_path)


def test_group_statistics_do_not_count_candidates_as_samples(tmp_path):
    _, payload = synthetic_cache(tmp_path)
    rows = payload["rows"]
    predictions = {r["scene_id"]: int(np.argmax(r["branch_returns"])) for r in rows}
    result = reports.summarize_predictions(rows, predictions)
    assert result["num_origin_logs"] == 32
    assert result["screening_pass"]
    assert result["beneficial_changes"] == 64
    assert result["harmful_changes"] == 0


def test_train_rejects_gate_for_different_cache(tmp_path):
    _, payload = synthetic_cache(tmp_path)
    path = tmp_path / "cache.pkl"
    c.atomic_pickle(path, payload)
    audit = tmp_path / "audit.json"
    c.atomic_json(audit, dict(status="PASS", cache_file=str(path), cache_file_sha256=c.sha256_file(path)))
    gate = tmp_path / "gate.json"
    c.atomic_json(gate, dict(status="PASS", training_authorized=True, cache_sha256="wrong"))
    with pytest.raises(RuntimeError, match="does not authorize"):
        trainer.train(SimpleNamespace(cache_audit=audit, pilot_gate=gate, cache=path))


def test_incumbent_recompute_has_separate_numeric_and_exact_decision_gates(monkeypatch):
    rows = [dict(scene_id=f"scene-{i}", v3_logits=np.array([1., 0.], dtype=np.float32))
            for i in range(3)]
    monkeypatch.setattr(model_api, "score", lambda *args, **kwargs:
                        torch.tensor([[1.00005, 0.]] * 3))
    passed = trainer.validate_incumbent_recompute(object(), rows, "cpu")
    assert passed["max_abs"] < c.MODEL_RECOMPUTE_TOLERANCE
    assert passed["argmax_mismatch_count"] == 0
    monkeypatch.setattr(model_api, "score", lambda *args, **kwargs:
                        torch.tensor([[1.0002, 0.]] * 3))
    with pytest.raises(RuntimeError, match="max_abs"):
        trainer.validate_incumbent_recompute(object(), rows, "cpu")
    rows[0]["v3_logits"] = np.array([0.5, 0.5], dtype=np.float32)
    monkeypatch.setattr(model_api, "score", lambda *args, **kwargs:
                        torch.tensor([[0.49999, 0.50001], [1., 0.], [1., 0.]]))
    with pytest.raises(RuntimeError, match="argmax_mismatch_count"):
        trainer.validate_incumbent_recompute(object(), rows, "cpu")


def test_cv_continuation_lineage_links_only_verified_frozen_inputs(tmp_path):
    _, pilot = synthetic_cache(tmp_path)
    pilot["collection_code_sha"] = "c" * 64
    source = tmp_path / "source_run"
    cache_dir = source / "cache"
    pilot_path = cache_dir / "pilot64_cache.pkl"
    repeat_path = cache_dir / "repeat8_cache.pkl"
    c.atomic_pickle(pilot_path, pilot)
    repeat = dict(pilot, stage="repeat8", num_rows=8, rows=pilot["rows"][:8])
    c.atomic_pickle(repeat_path, repeat)
    c.atomic_json(cache_dir / "pilot64_cache_audit.json",
                  dict(status="PASS", cache_file=str(pilot_path),
                       cache_file_sha256=c.sha256_file(pilot_path)))
    c.atomic_json(cache_dir / "repeat8_cache_audit.json",
                  dict(status="PASS", cache_file=str(repeat_path),
                       cache_file_sha256=c.sha256_file(repeat_path)))
    c.atomic_json(source / "run_contract.json",
                  dict(code_sha="c" * 64, design_version=c.DESIGN_VERSION, gpu_hour_limit=10.0))
    c.atomic_json(source / "decision_ledger.json",
                  dict(active=None, gpu_hours_used=4.25, development_consumed=False,
                       test_consumed=False, expansion_authorized=False))
    c.atomic_json(source / "pilot_gate.json",
                  dict(status="PASS", training_authorized=True, information_gate_pass=True,
                       expansion_authorized=False, development_consumed=False, test_consumed=False,
                       cache_sha256=c.sha256_file(pilot_path),
                       repeat_cache_sha256=c.sha256_file(repeat_path),
                       improvable_above_0p02=20, repeat_max_absolute_error=0.0))
    lineage = runner.validate_cv_source(source)
    assert lineage["remaining_gpu_hours"] == 5.75
    target = tmp_path / "continuation"
    runner.link_cv_inputs(target, lineage)
    assert (target / "pilot_gate.json").is_symlink()
    assert (target / "cache/pilot64_cache.pkl").resolve() == pilot_path.resolve()
    assert runner.validate_cv_source(source) == lineage
    runner.link_cv_inputs(target, lineage)
    changed = json.loads((source / "pilot_gate.json").read_text())
    changed["cache_sha256"] = "wrong"
    c.atomic_json(source / "pilot_gate.json", changed)
    with pytest.raises(RuntimeError, match="lineage drifted"):
        runner.validate_cv_source(source)
    with pytest.raises(RuntimeError, match="changed before linking"):
        runner.link_cv_inputs(target, lineage)


def test_historical_exposure_membership_only(tmp_path):
    logs = {"log-a", "log-b"}
    splits = {}
    for split, names in (("train", ["log-a"]), ("development", ["log-b"]), ("certification", ["log-c"])):
        path = tmp_path / (split + ".jsonl")
        path.write_text("\n".join(json.dumps(dict(log_name=name)) for name in names) + "\n")
        splits[split] = dict(pairs=str(path), pairs_sha256=c.sha256_file(path))
    path = tmp_path / "audit.json"
    c.atomic_json(path, dict(status="PASS", splits=splits))
    audit = prep.historical_exposure(logs, path)
    assert audit["splits"]["train"]["overlapping_log_count"] == 1
    assert audit["splits"]["development"]["overlapping_log_count"] == 1
    assert audit["splits"]["certification"]["overlapping_log_count"] == 0
    assert not audit["fully_unseen_claim_authorized"]


def test_synthetic_freeze_treatments_assembly_and_gate(tmp_path, monkeypatch):
    """Exercise real freeze/41-manifest/20+20 arm assembly code using synthetic returns."""
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic integration contract\n")
    source, frames, outcomes = {}, {}, {}
    records_root = tmp_path / "baseline_records"
    records_root.mkdir()
    for i, candidate in enumerate(target_rows()):
        scene = candidate["scene_id"]
        original = dict(id=scene, token="0123456789abcdef", metadata=dict(
            nuplan_lidar_pc_tokens=["0123456789abcdef"],
            openscene_data_infos_dict={"0123456789abcdef": {"log_name": candidate["origin_log"]}}))
        source[scene] = c.annotate_scenario(original, candidate["scenario_family"], "train")
        success = candidate["outcome_stratum"] == "solved"
        outcomes[scene] = dict(score=.5, success=success, no_at_fault_collisions=float(success),
                               drivable_area_compliance=1., ego_progress=.5,
                               first_violation_step=None if success else 12)
        frames[scene] = {}
        for step in c.DECISION_STEPS:
            record = visible_row()
            record.update(rollout_scene_id=scene, decision_step=step, state_step=step-1,
                          current_logits=record["v3_logits"], policy_selected_index=19,
                          candidate_rewards=np.linspace(.1, .5, 20, dtype=np.float32),
                          candidate_reward_components=np.ones((20, 6), dtype=np.float32),
                          checkpoint_sha256=c.CHECKPOINT_SHA256, candidate_noise_namespace=c.NOISE_NAMESPACE)
            path = records_root / f"{i}_{step}.pkl"
            c.atomic_pickle(path, record)
            frames[scene][step] = (path, record)
    source_path = tmp_path / "source/train_pool_1024.pkl"
    c.atomic_pickle(source_path, source)
    source_audit = dict(train_scenario_file=str(source_path),
                        train_scenario_file_sha256=c.sha256_file(source_path))
    c.atomic_json(tmp_path / "source/source_audit.json", source_audit)
    monkeypatch.setattr(prep, "source_pool", lambda args: source_audit)
    monkeypatch.setattr(prep, "load_records", lambda audit: frames)

    def metrics(path, ids, index=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            fields = ["token", "score", "no_at_fault_collisions", "drivable_area_compliance",
                      "ego_progress", "first_violation_step"]
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for scene in ids:
                original = outcomes[scene]
                values = {key: original[key] for key in fields if key != "token"}
                if index is not None and index != 19:
                    values["score"] = .75
                writer.writerow(dict(token=scene, **values))

    for label in ("a", "b"):
        path = tmp_path / f"collections/baseline_train_{label}/metrics.csv"
        metrics(path, source)
        c.atomic_json(path.parent / "collection_audit.json", dict(
            status="PASS", method=c.COLLECTION_AUDIT_METHOD, layout="merged",
            collection_id=f"baseline_train_{label}", source_split="train", intervention_mode="observe_only",
            intervention_count=0, scenario_file_sha256=c.sha256_file(source_path),
            checkpoint_sha256=c.CHECKPOINT_SHA256, selector_state_sha256="selector",
            candidate_noise_namespace=c.NOISE_NAMESPACE, rollout_implementation_sha256="code", code_sha="code",
            closed_loop_outcome=dict(metrics_csv=str(path), metrics_csv_sha256=c.sha256_file(path))))
    args = SimpleNamespace(run_root=tmp_path, protocol=protocol)
    master = prep.treatments(args)
    assert len(master["collections"]) == 41
    assert set(master["collections"]) == {"sentinel_policy"} | {
        f"{stage}_arm_{index:02d}" for stage in ("pilot64", "repeat8") for index in range(20)}
    # Freezing twice must not change treatment hashes.
    assert prep.treatments(args) == master
    for stage in ("pilot64", "repeat8"):
        for index in range(20):
            collection = f"{stage}_arm_{index:02d}"
            entry = master["collections"][collection]
            _, treatment = c.load_treatment(entry["path"], entry["sha256"])
            path = tmp_path / f"collections/{collection}/metrics.csv"
            metrics(path, [r["scene_id"] for r in treatment["targets"]], index)
            c.atomic_json(path.parent / "collection_audit.json", dict(
                status="PASS", method=c.COLLECTION_AUDIT_METHOD, layout="merged", collection_id=collection,
                intervention_mode="one_shot_manifest", treatment_manifest_sha256=entry["sha256"],
                intervention_count=treatment["target_count"], checkpoint_sha256=c.CHECKPOINT_SHA256,
                selector_state_sha256="selector", candidate_noise_namespace=c.NOISE_NAMESPACE,
                rollout_implementation_sha256="code", code_sha="code",
                closed_loop_outcome=dict(metrics_csv=str(path), metrics_csv_sha256=c.sha256_file(path))))
        assembly_args = SimpleNamespace(run_root=tmp_path, master_manifest=tmp_path / "manifests/master_manifest.json",
            stage=stage, output=tmp_path / f"cache/{stage}_cache.pkl",
            audit_output=tmp_path / f"cache/{stage}_cache_audit.json")
        assembler.assemble(assembly_args)
    gate = reports.pilot_gate(tmp_path)
    assert gate["training_authorized"] and not gate["expansion_authorized"]
    assert gate["repeat_max_absolute_error"] == 0
    # Altering an audited metrics file invalidates arm loading.
    corrupt = tmp_path / "collections/pilot64_arm_00/metrics.csv"
    corrupt.write_text(corrupt.read_text() + "\n")
    with pytest.raises(RuntimeError, match="metrics changed"):
        assembler.load_arm_audit(tmp_path, master, "pilot64", 0)


def test_report_complete_matrix_and_no_auto_promotion(tmp_path):
    _, payload = synthetic_cache(tmp_path)
    cache = tmp_path / "cache/pilot64_cache.pkl"
    c.atomic_pickle(cache, payload)
    c.atomic_json(tmp_path / "pilot_gate.json", dict(status="PASS", training_authorized=True,
                                                   cache_sha256=c.sha256_file(cache)))
    c.atomic_json(tmp_path / "cache/pilot64_cache_audit.json",
                  dict(status="PASS", cache_file=str(cache), cache_file_sha256=c.sha256_file(cache)))
    artifact = tmp_path / "synthetic_checkpoint.pt"
    torch.save(dict(synthetic=True), artifact)
    for method in obj.METHODS:
        for lr in trainer.LEARNING_RATES:
            for seed in trainer.SEEDS:
                for fold in range(4):
                    predictions = {r["scene_id"]: int(np.argmax(r["branch_returns"]))
                                   for r in payload["rows"] if r["fold"] == fold}
                    identity = dict(method=method, learning_rate=lr, seed=seed, fold=fold)
                    report = dict(status="PASS", **identity,
                                  provenance=dict(**identity, cache_sha256=c.sha256_file(cache),
                                      pilot_gate_sha256=c.sha256_file(tmp_path / "pilot_gate.json")),
                                  checkpoints=[dict(step=step, predictions=predictions,
                                                    checkpoint=str(artifact), checkpoint_sha256=c.sha256_file(artifact))
                                               for step in trainer.CHECKPOINT_STEPS])
                    path = tmp_path / "cv" / trainer.job_name(method, lr, fold, seed) / "report.json"
                    c.atomic_json(path, report)
    result = reports.cv_report(tmp_path)
    assert result["decision"] == "REVIEW_EXPANSION"
    assert len(result["configurations"]) == 36
    assert result["selected_configuration"]["step"] == 50
    assert not result["expansion_authorized"] and not result["loss_novelty_established"]
    assert (tmp_path / "PILOT_REPORT.md").is_file()
    path = tmp_path / "cv" / trainer.job_name(next(iter(obj.METHODS)), trainer.LEARNING_RATES[0], 0, 0) / "report.json"
    original = json.loads(path.read_text())
    changed = copy.deepcopy(original)
    changed["fold"] = 1
    c.atomic_json(path, changed)
    with pytest.raises(RuntimeError, match="job identity"):
        reports.cv_report(tmp_path)
    changed = copy.deepcopy(original)
    changed["checkpoints"][0]["predictions"].pop(next(iter(changed["checkpoints"][0]["predictions"])))
    c.atomic_json(path, changed)
    with pytest.raises(RuntimeError, match="held-out fold"):
        reports.cv_report(tmp_path)
    c.atomic_json(path, original)
    torch.save(dict(synthetic=False), artifact)
    with pytest.raises(RuntimeError, match="artifact drifted"):
        reports.cv_report(tmp_path)
    payload["rows"][0]["branch_returns"][0] += .01
    c.atomic_pickle(cache, payload)
    c.atomic_json(tmp_path / "cache/pilot64_cache_audit.json",
                  dict(status="PASS", cache_file=str(cache), cache_file_sha256=c.sha256_file(cache)))
    with pytest.raises(RuntimeError, match="authorized pilot"):
        reports.cv_report(tmp_path)


def test_runner_stops_owned_process_on_budget(tmp_path):
    # No GPU is used; a subprocess sleep exercises accounting and cleanup.
    instance = runner.Runner.__new__(runner.Runner)
    instance.args = SimpleNamespace(gpu_hours=.000001)
    instance.ledger = dict(gpu_hours_used=0.0, events=[])
    instance.ledger_path = tmp_path / "ledger.json"
    instance.children = []
    with pytest.raises(RuntimeError, match="budget reached"):
        instance.execute([([sys.executable, "-c", "import time; time.sleep(30)"],
                           tmp_path, os.environ.copy(), tmp_path / "job.log")],
                         "synthetic_budget", gpus=1)
    assert not instance.children
    assert instance.ledger["active"] is None
    assert instance.ledger["gpu_hours_used"] > 0


def test_preflight_rejects_wrong_allocation_and_persists_failure(tmp_path, monkeypatch):
    from selector_cfpi_preflight import validate_hardware
    fake_torch = SimpleNamespace(__version__="2.0.1+cu118", version=SimpleNamespace(cuda="11.8"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                             get_device_name=lambda i: "NVIDIA GeForce RTX 4090",
                             get_device_capability=lambda i: (8, 9)))
    with pytest.raises(RuntimeError, match="Expected 8 visible H100 GPUs, found 1"):
        validate_hardware(fake_torch, 8)
    with pytest.raises(RuntimeError, match="Expected H100 allocation"):
        validate_hardware(fake_torch, 1)
    fake_torch.cuda.get_device_name = lambda i: "NVIDIA H100 80GB HBM3"
    fake_torch.cuda.get_device_capability = lambda i: (9, 0)
    validate_hardware(fake_torch, 1)
    instance = runner.Runner.__new__(runner.Runner)
    instance.run = tmp_path

    def failed_probe():
        raise RuntimeError("synthetic hardware mismatch")

    monkeypatch.setattr(instance, "_preflight", failed_probe)
    with pytest.raises(RuntimeError, match="hardware mismatch"):
        instance.preflight()
    assert json.loads((tmp_path / "preflight.json").read_text())["status"] == "FAIL"
