"""CPU software tests only; no training, scene rollout or privileged selection."""
import ast
import copy
import csv
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "projects/AlgEngine/scripts/diffusiondrive"))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_deployment_router as router_module
import refit_selector_cfpi_deployment as refit_module
import report_selector_cfpi_deployment as report_module
import prepare_selector_cfpi_deployment as prepare_module
from audit_selector_cfpi_deployment import SHAPES, validate_sidecar, validate_coverage, action_error, action_arrays, audit_collection, load_current_reports
from prepare_selector_cfpi_deployment import natural_membership
from report_selector_cfpi_deployment import report, differences
from run_selector_cfpi_deployment import charge, check_budget, process_identity, owned_session_members, archive_incomplete_collection, Runner


def extracted_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


expand_torch = extracted_function(ROOT / "projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py",
                                  "_expand_to_40", {"torch": torch})
expand_numpy = extracted_function(ROOT / "projects/SimEngine/worldengine/manager/diffusiondrive_dynamic_reward_manager.py",
                                  "expand_candidates_to_40", {"np": np})


def context():
    out = {key: np.zeros(shape, np.float32) for key, shape in SHAPES.items()}
    out["candidate_trajectories_8"][:, :, 0] = np.arange(20)[:, None]
    out["reference_logits"][0] = 10
    out["current_logits"][0] = 10
    out["selected_indices"] = np.array(0)
    return out


class ToySelector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.randn(3))

    def forward(self, **inputs):
        if set(inputs) != {"candidate_features", "candidate_trajectories", "route_bev_features", "status_token", "ego_query", "agents_query"}:
            raise RuntimeError("Non-visible data reached the selector")
        scores = torch.zeros((inputs["candidate_features"].shape[0], 20))
        scores[:, 2] = 1
        return scores


class ToyRefit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.randn(20) * .01)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="cfpi-deployment-test-")
        self.root = Path(self.directory.name)
        self.scene = "log-a-" + "a" * 16
        self.other = "log-b-" + "b" * 16
        self.manifest = dict(status="PASS", method=d.METHOD, policy="q_mse_seed0", phase="phase_a",
            terminal_publication_decision=12,
            inference_uses_reward_or_q=False, incumbent_selector_sha256="incumbent",
            models={"fold0": dict(kind="cfpi", path="unused", sha256="model0", score_mode="direct_q")},
            routes=[dict(scene_id=self.scene, origin_token="a" * 16, fold=0, model_key="fold0", start_decision=6),
                    dict(scene_id=self.other, origin_token="b" * 16, fold=1, model_key="fold0", start_decision=7)])

    def tearDown(self):
        self.directory.cleanup()

    def router(self, manifest=None):
        path = self.root / "routing.json"
        c.atomic_json(path, manifest or self.manifest)
        with patch.object(router_module, "load_bank", side_effect=lambda _: ToySelector().eval()):
            return router_module.DeploymentRouter(path, c.sha256_file(path), lambda t: expand_torch(None, t), "cpu")

    def result(self):
        row = context()
        return dict(diffusiondrive_rollout_context=row, chosen_ind=0, token="frame",
                    trajectory=expand_torch(None, torch.tensor(row["candidate_trajectories_8"][:1]))[0].numpy(),
                    ade_4s=1.0, fde_4s=2.0)

    def sidecar(self, router, step=6):
        result = router.apply(self.result(), "a" * 16, step)
        contract = dict(experiment=d.METHOD, deployment_routing_sha256=router.manifest_sha256,
                        rollout_implementation_sha256="code", diagnostic_split="collection", inference_uses_reward_or_q=False)
        return dict(result["diffusiondrive_rollout_context"], cfpi_deployment=result["cfpi_deployment"],
                    planner_step=step, scene_prefix="a" * 16, selected_index=result["chosen_ind"],
                    deployed_trajectory=result["trajectory"], selector_rollout_contract=contract,
                    checkpoint_sha256=c.CHECKPOINT_SHA256, code_sha="code", candidate_noise_namespace=c.NOISE_NAMESPACE)

    def collection(self, router):
        return dict(routing=dict(sha256=router.manifest_sha256), collection_id="collection", audit_targets={}, sentinel=False)

    def test_takeover_inclusive_and_persistent(self):
        router = self.router()
        for step in c.DECISION_STEPS:
            result = router.apply(self.result(), "a" * 16, step)
            self.assertEqual(result["chosen_ind"], 2 if step >= 6 else 0)
            self.assertEqual(result["cfpi_deployment"]["active"], step >= 6)

    def test_scene_reset_has_no_policy_carry_over(self):
        router = self.router()
        self.assertEqual(router.apply(self.result(), "a" * 16, 11)["chosen_ind"], 2)
        self.assertEqual(router.apply(self.result(), "b" * 16, 4)["chosen_ind"], 0)
        self.assertEqual(router.apply(self.result(), "a" * 16, 4)["chosen_ind"], 0)

    def test_direct_q_not_added_to_base(self):
        router = self.router()
        result = router.apply(self.result(), "a" * 16, 6)
        self.assertEqual(result["chosen_ind"], 2)
        self.assertNotIn("ade_4s", result)
        self.assertNotIn("fde_4s", result)
        self.assertEqual(result["cfpi_deployment"]["score_mode"], "direct_q")

    def test_residual_added_to_base(self):
        self.manifest["models"]["fold0"]["score_mode"] = "residual"
        result = self.router().apply(self.result(), "a" * 16, 6)
        self.assertEqual(result["chosen_ind"], 0)
        self.assertIn("ade_4s", result)

    def test_routing_unknown_and_off_by_one_fail_closed(self):
        for prefix, step in [("unknown", 4), ("a" * 16, 3), ("a" * 16, 12)]:
            with self.assertRaises(RuntimeError):
                d.route_for(self.manifest, prefix, step)

    def test_terminal_publication_is_not_an_executed_decision(self):
        router = self.router()
        result = router.apply(self.result(), "a" * 16, 12)
        self.assertTrue(result["cfpi_deployment"]["terminal_unexecuted"])
        with self.assertRaises(RuntimeError):
            validate_sidecar(self.sidecar(router, 12), self.collection(router), router.manifest, "code")

    def test_ambiguous_scene_fails(self):
        self.manifest["routes"].append(copy.deepcopy(self.manifest["routes"][0]))
        with self.assertRaises(RuntimeError):
            d.route_for(self.manifest, "a" * 16, 6)

    def test_generator_outputs_untouched_and_no_privileged_inputs(self):
        router = self.router()
        result = self.result()
        before = copy.deepcopy(result["diffusiondrive_rollout_context"])
        result["diffusiondrive_rollout_context"]["branch_returns"] = "must never reach forward"
        router.apply(result, "a" * 16, 6)
        for key in set(SHAPES) - {"current_logits"}:
            np.testing.assert_array_equal(result["diffusiondrive_rollout_context"][key], before[key])

    def test_bank_construction_is_rng_neutral(self):
        before = torch.get_rng_state().clone()
        self.router()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_hash_drift_rejected(self):
        path = self.root / "routing.json"
        c.atomic_json(path, self.manifest)
        expected = c.sha256_file(path)
        c.atomic_json(path, dict(self.manifest, policy="different"))
        with self.assertRaises(RuntimeError):
            router_module.DeploymentRouter(path, expected, None, "cpu")

    def test_numpy_simulator_torch_planner_trajectory_parity(self):
        candidates = np.random.default_rng(20260906).uniform(-3, 3, size=(20, 8, 3)).astype(np.float32)
        expected = expand_numpy(candidates)
        actual = expand_torch(None, torch.from_numpy(candidates)).numpy()
        self.assertLess(np.max(np.abs(expected - actual)), c.ACTION_TOLERANCE)

    def test_sidecar_route_score_identity_checks(self):
        router = self.router()
        sidecar = self.sidecar(router)
        collection = self.collection(router)
        result = validate_sidecar(sidecar, collection, router.manifest, "code")
        self.assertEqual(result["selected_index"], 2)
        for key, value in [("fold", 3), ("selector_sha256", "wrong"), ("score_mode", "residual")]:
            bad = copy.deepcopy(sidecar)
            bad["cfpi_deployment"][key] = value
            with self.assertRaises(RuntimeError):
                validate_sidecar(bad, collection, router.manifest, "code")

    def test_prefix_and_exact_oof_target_checked(self):
        router = self.router()
        sidecar = self.sidecar(router)
        baseline = context()
        baseline.update(selected_index=0, deployed_trajectory=self.result()["trajectory"])
        path = self.root / "prefix.pkl"
        c.atomic_pickle(path, baseline)
        collection = self.collection(router)
        target = dict(prefix={"6": d.artifact(path)}, target_decision=6, expected_target_action=2)
        collection["audit_targets"][self.scene] = target
        self.assertTrue(validate_sidecar(sidecar, collection, router.manifest, "code")["target_checked"])
        target["expected_target_action"] = 3
        with self.assertRaises(RuntimeError):
            validate_sidecar(sidecar, collection, router.manifest, "code")
        target["expected_target_action"] = 2
        sidecar["candidate_features"][0, 0] = .01
        with self.assertRaises(RuntimeError):
            validate_sidecar(sidecar, collection, router.manifest, "code")

    def test_actual_action_fields_not_silently_missing(self):
        with self.assertRaises(RuntimeError):
            action_arrays(object())
        action = types.SimpleNamespace(waypoints=np.zeros((4, 2)), velocities=None, headings=None, angular_velocities=None)
        left = action_arrays(action)
        right = copy.deepcopy(left)
        right["waypoints"][0][0] = .1
        self.assertEqual(action_error(left, right), .1)

    def test_resume_coverage_requires_all_frames_and_reports(self):
        rows = [dict(scene_id=self.scene, decision_step=step) for step in c.DECISION_STEPS]
        validate_coverage(rows, {self.scene}, {self.scene}, {self.scene: [False, True]})
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(RuntimeError):
                validate_coverage(bad, {self.scene}, {self.scene}, {self.scene: [True]})
        with self.assertRaises(RuntimeError):
            validate_coverage(rows, {self.scene}, set(), {self.scene: [True]})

    def test_advancement_requires_effect_and_retention(self):
        delta = {k: 0.0 for k in d.METRICS}
        delta["score"] = .006
        self.assertTrue(d.advancement([delta] * 3)["pass_screening"])
        bad = dict(delta, success=-.02)
        self.assertFalse(d.advancement([bad] * 3)["pass_screening"])
        low = dict(delta, score=.0049)
        self.assertFalse(d.advancement([low] * 3)["pass_screening"])
        negative = dict(delta, score=-.001)
        self.assertFalse(d.advancement([dict(delta, score=.03), negative, negative])["pass_screening"])

    def test_budget_phase_and_total_and_recovery_charge(self):
        ledger = dict(gpu_hours_used=0.0, phase_gpu_hours={"phase_a": 0.0, "phase_b": 0.0})
        charge(ledger, "phase_a", 3600, 8)
        self.assertEqual(ledger["gpu_hours_used"], 8)
        self.assertEqual(ledger["phase_gpu_hours"]["phase_a"], 8)
        charge(ledger, "phase_a", 7200, 8)
        with self.assertRaises(RuntimeError):
            check_budget(ledger, "phase_a", 64)
        check_budget(ledger, "phase_b", 64)
        with self.assertRaises(RuntimeError):
            check_budget(ledger, "phase_b", 24)
        with self.assertRaises(RuntimeError):
            check_budget(ledger, "phase_b", 25, reserve=1)

    def test_pid_identity_is_specific_to_owned_process(self):
        identity = process_identity(os.getpid())
        self.assertEqual(identity["pid"], os.getpid())
        self.assertNotEqual(identity, process_identity(os.getppid()))
        self.assertIsNone(process_identity(999999999))

    def test_incomplete_report_cannot_authorize_phase_b(self):
        result = report(self.root, "phase_a")
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result.get("phase_b_authorized", False))
        self.assertFalse((self.root / "phase_a_report.json").exists())

    def test_existing_default_path_inert(self):
        tree = ast.parse((ROOT / "projects/AlgEngine/closed_loop/sim_test.py").read_text())
        loop = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_inference_loop")
        creation = next(n for n in loop.body if isinstance(n, ast.If) and "cfpi_deployment_routing" in ast.unparse(n.test))
        self.assertIn("DeploymentRouter", ast.unparse(creation))
        self.assertIn("if deployment_router is not None", ast.unparse(loop))
        config = (ROOT / "projects/SimEngine/worldengine/configs/default_runner.yaml").read_text()
        self.assertIn("diffusiondrive_cfpi_deployment: false", config)

    def test_incomplete_condition_archived_without_touching_contract(self):
        (self.root / "split_0/frames").mkdir(parents=True)
        c.atomic_pickle(self.root / "split_0/frames/frame.pkl", {"old_frame": True})
        source = self.root / "split_0/frames/frame.pkl"
        original_sha = c.sha256_file(source)
        c.atomic_json(self.root / "routing.json", self.manifest)
        archive = Path(archive_incomplete_collection(self.root))
        self.assertFalse(source.exists())
        self.assertEqual(c.sha256_file(archive / "split_0/frames/frame.pkl"), original_sha)
        self.assertEqual(d.read(self.root / "routing.json"), self.manifest)
        self.assertEqual(d.read(archive / "recovery_map.json")["status"], "ARCHIVED_INCOMPLETE")
        self.assertIsNone(archive_incomplete_collection(self.root))
        c.atomic_json(self.root / "collection_audit.json", {"status": "PASS"})
        with self.assertRaises(RuntimeError):
            archive_incomplete_collection(self.root)

    def test_archived_report_cannot_supply_current_success(self):
        row = dict(scenario_name=self.scene, succeeded=True)
        c.atomic_json(self.root / "attempt_archive/old/runner_report_1.json", [row])
        with self.assertRaises(RuntimeError):
            load_current_reports(self.root)
        c.atomic_json(self.root / "__WORKER_ID__/WE_output/runner_report_2.json", [dict(row, succeeded=False)])
        _, outcomes = load_current_reports(self.root)
        self.assertEqual(outcomes, {self.scene: [False]})

    def test_current_owned_session_detected(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
        try:
            identity = process_identity(process.pid)
            self.assertIn(process.pid, owned_session_members(identity))
            self.assertNotIn(os.getpid(), owned_session_members(identity))
        finally:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()

    def test_complete_simulator_action_audit_and_resume(self):
        self.manifest["routes"] = self.manifest["routes"][:1]
        self.manifest["routes"][0]["start_decision"] = 4
        c.atomic_json(self.root / "toy_model.json", {"synthetic_model": True})
        self.manifest["models"]["fold0"].update(d.artifact(self.root / "toy_model.json"))
        router = self.router()
        worker = self.root / "split_0"
        output = worker / "WE_output/openscene_format"
        plan_dir, sidecar_dir = worker / "plan_traj", worker / "diffusiondrive_candidate_sidecars"
        for folder in (output, plan_dir, sidecar_dir, worker / "completed_scenarios"):
            folder.mkdir(parents=True, exist_ok=True)
        c.atomic_json(self.root / "run_contract.json", dict(code_sha="code", gpu_count=1))
        c.atomic_json(self.root / "inputs.json", dict(status="PASS"))
        c.atomic_pickle(self.root / "scenes.pkl", {self.scene: {"id": self.scene, "log_length": 20}})
        collection = dict(self.collection(router), routing=d.artifact(self.root / "routing.json"),
                          run_contract=d.artifact(self.root / "run_contract.json"),
                          scenario=d.artifact(self.root / "scenes.pkl"), inputs=d.artifact(self.root / "inputs.json"))
        contract_path = self.root / "deployment_collection.json"
        c.atomic_json(contract_path, collection)
        method = extracted_function(ROOT / "projects/SimEngine/worldengine/manager/diffusiondrive_cfpi_deployment_manager.py",
            "before_step", dict(c=c, d=d, np=np, Path=Path, r15=__import__("oracle_r15_common"),
                                validate_sidecar=validate_sidecar, expand_candidates_to_40=expand_numpy,
                                action_arrays=action_arrays, action_error=action_error))
        converter = lambda plan: types.SimpleNamespace(waypoints=np.asarray(plan)[:, :2] + 100,
                                                       velocities=None, headings=None, angular_velocities=None)
        plans = []
        for step in c.DECISION_STEPS:
            sidecar = self.sidecar(router, step)
            path = sidecar_dir / f"{'a' * 16}_{step}.pkl"
            c.atomic_pickle(path, sidecar)
            np.save(plan_dir / f"{'a' * 16}_{step}.npy", sidecar["deployed_trajectory"])
            original_action = converter(sidecar["deployed_trajectory"])
            engine = types.SimpleNamespace(episode_step=step-1, external_actions=original_action,
                       global_config=dict(planner_data_path=str(plan_dir), data_output_dir=str(output)))
            manager = types.SimpleNamespace(engine=engine, current_scene={"id": self.scene},
                _load_sidecar=lambda s=sidecar, p=path: (s, p), collection=collection,
                routing=router.manifest, code_sha="code", collection_sha=c.sha256_file(contract_path),
                agent=types.SimpleNamespace(client=types.SimpleNamespace(trajectory_from_local_plan=converter)))
            method(manager)
            self.assertIs(engine.external_actions, original_action)  # Audit manager never overrides action.
            plans.append(dict(prefix="a" * 16, step=step, plan_idx=2))
        terminal = self.sidecar(router, 12)
        c.atomic_pickle(sidecar_dir / f"{'a' * 16}_12.pkl", terminal)
        np.save(plan_dir / f"{'a' * 16}_12.npy", terminal["deployed_trajectory"])
        plans.append(dict(prefix="a" * 16, step=12, plan_idx=2))
        with (plan_dir / "plan_idx.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["prefix", "step", "plan_idx"])
            writer.writeheader()
            writer.writerows(plans)
        with (output / "all_scenes_pdm_averages_R.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["token", *d.METRICS[:-1]])
            writer.writeheader()
            writer.writerow(dict(token=self.scene, **{k: 1.0 for k in d.METRICS[:-1]}))
        (worker / "completed_scenarios/completed_split_0.txt").write_text(self.scene + "\n")
        c.atomic_json(self.root / "__WORKER_ID__/WE_output/runner_report_0.json",
                      [dict(scenario_name=self.scene, succeeded=True)])
        missing = output / f"cfpi_deployment_records/{'a' * 16}_11.json"
        hidden = missing.with_suffix(".partial")
        missing.rename(hidden)
        with self.assertRaises(RuntimeError):
            audit_collection(contract_path)
        hidden.rename(missing)
        result = audit_collection(contract_path)
        self.assertEqual(result["records"], 8)
        self.assertEqual(result["scenes"], 1)
        self.assertEqual(len(result["terminal_unexecuted_publications"]), 1)
        self.assertEqual(audit_collection(contract_path), result)
        np.save(plan_dir / f"{'a' * 16}_11.npy", np.ones((40, 3)))
        with self.assertRaises(RuntimeError):
            audit_collection(contract_path)

    def test_refit_optimizer_rng_resume_matches_uninterrupted_toy_training(self):
        # Only a synthetic 20-parameter vector is optimized; no real planner or cache is trained.
        labels = np.random.default_rng(5).uniform(size=(64, 20)).astype(np.float32)
        rows = [dict(branch_returns=value) for value in labels]
        outputs = []
        for interrupted in (False, True):
            folder = self.root / ("interrupted" if interrupted else "straight")
            c.atomic_pickle(folder / "cache.pkl", rows)
            c.atomic_json(folder / "scalar.json", {})
            c.atomic_json(folder / "pilot_gate.json", dict(training_authorized=True, cache_sha256=c.sha256_file(folder / "cache.pkl")))
            c.atomic_json(folder / "deployment_inputs.json", dict(artifacts={
                "cache": d.artifact(folder / "cache.pkl"), "pilot_gate": d.artifact(folder / "pilot_gate.json"),
                "scalar_manifest": d.artifact(folder / "scalar.json")}))
            c.atomic_json(folder / "run_contract.json", dict(code_sha="synthetic-test"))
            c.atomic_json(folder / "phase_a_report.json", dict(status="PASS", phase_b_authorized=True))
            def reload_toy(path):
                payload = torch.load(path, map_location="cpu")
                model = ToyRefit()
                model.load_state_dict(payload["scene_selector_state"])
                return model.eval(), payload
            original_save = torch.save
            def stop_at_25(payload, path):
                original_save(payload, path)
                if isinstance(payload, dict) and payload.get("step") == 25 and "optimizer" in payload:
                    raise RuntimeError("injected optimizer checkpoint interruption")
            with patch.object(refit_module.c, "validate_cache", side_effect=lambda value: value), \
                 patch.object(refit_module, "validate_incumbent_recompute", return_value={"synthetic": True}), \
                 patch.object(refit_module.models, "load_incumbent", side_effect=lambda _: (ToyRefit(), {"toy": True}, {})), \
                 patch.object(refit_module.models, "score", side_effect=lambda model, rows, ids, device, mode: model.logits[None].expand(len(ids), -1)), \
                 patch.object(refit_module.models, "load_selector", side_effect=reload_toy), \
                 patch.dict(d.POLICIES, q_grpo_t1=dict(lr=1e-4, steps=50, score_mode="residual")):
                if interrupted:
                    with patch.object(torch, "save", side_effect=stop_at_25), self.assertRaisesRegex(RuntimeError, "injected"):
                        refit_module.train(folder, "q_grpo_t1", 0, "cpu")
                refit_module.train(folder, "q_grpo_t1", 0, "cpu")
                refit_module.train(folder, "q_grpo_t1", 0, "cpu")  # Completed job is immutable/revalidated.
            final = d.read(folder / "refit/q_grpo_t1_seed0/report.json")
            outputs.append(torch.load(final["selector"]["path"], map_location="cpu")["scene_selector_state"]["logits"])
        self.assertTrue(torch.equal(outputs[0], outputs[1]))

    def test_natural_membership_ignores_outcomes_and_csv_order(self):
        inventories, scene_ids = [], []
        for family_index, family in enumerate(c.ALLOWED_FAMILIES):
            ids = [f"family{family_index}-log{i:03d}-{family_index * 1000 + i:016x}" for i in range(80)]
            path = self.root / f"index_{family}.json"
            c.atomic_json(path, dict(shards=[dict(scenario_ids=ids)]))
            inventories.append(dict(family=family, index=str(path), index_sha256=c.sha256_file(path)))
            scene_ids += ids
        # The production source has 1024 IDs; duplicate-free extra IDs all remain in indexes.
        for family_index, entry in enumerate(inventories):
            extra = [f"family{family_index}-log{i:03d}-{family_index * 1000 + i:016x}" for i in range(80, 512)]
            index = d.read(entry["index"])
            index["shards"][0]["scenario_ids"] += extra
            c.atomic_json(entry["index"], index)
            entry["index_sha256"] = c.sha256_file(entry["index"])
            scene_ids += extra
        history = {"splits": {}, "source": "unused"}
        for split, log in [("development", "family1-log000"), ("certification", "family0-log001")]:
            path = self.root / f"{split}.jsonl"
            path.write_text('{"log_name":"' + log + '"}\n')
            history["splits"][split] = dict(source=str(path), sha256=c.sha256_file(path))
        audit = dict(source_inventories=inventories, historical_v3_membership_audit=history)
        baseline = self.root / "source.csv"
        results = []
        for order, score in [(scene_ids, 0), (list(reversed(scene_ids)), 1)]:
            with baseline.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["token", "score", "first_violation_step"])
                writer.writeheader()
                writer.writerows(dict(token=s, score=score, first_violation_step=score * 99) for s in order)
            with patch.object(prepare_module, "historical_exposure", return_value={"fully_unseen_claim_authorized": False}):
                results.append(natural_membership(audit, [{"origin_log": "family0-log000"}], baseline))
        self.assertEqual(results[0], results[1])
        rows = results[0]["rows"]
        self.assertEqual(len({r["origin_log"] for r in rows}), 128)
        self.assertEqual({f: sum(r["scenario_family"] == f for r in rows) for f in c.ALLOWED_FAMILIES},
                         {f: 64 for f in c.ALLOWED_FAMILIES})
        self.assertFalse({r["origin_log"] for r in rows} & {"family0-log000", "family0-log001", "family1-log000"})

    def test_full_phase_a_report_one_shot_comparison_and_gate(self):
        rows = [dict(scene_id=f"scene{i:02d}", origin_log=f"log{i:02d}", scenario_family=c.ALLOWED_FAMILIES[i % 2],
                     outcome_stratum="solved", time_stratum="ordinary") for i in range(64)]
        baseline = {r["scene_id"]: dict.fromkeys(d.METRICS, 1.0) for r in rows}
        for values in baseline.values():
            values["score"] = .5
        def write_metrics(path, values):
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["token", *d.METRICS[:-1]])
                writer.writeheader()
                writer.writerows(dict(token=s, **{k: v[k] for k in d.METRICS[:-1]}) for s, v in values.items())
        baseline_path = self.root / "baseline.csv"
        write_metrics(baseline_path, baseline)
        targets_path = self.root / "targets.json"
        c.atomic_json(targets_path, dict(targets=rows))
        audit_path = self.root / "synthetic_audit.json"
        c.atomic_json(audit_path, dict(status="PASS", synthetic=True))
        arms = []
        for candidate in range(20):
            values = copy.deepcopy(baseline)
            for value in values.values():
                value["score"] = .52 if candidate == 1 else .5
            path = self.root / f"arm_{candidate}.csv"
            write_metrics(path, values)
            arms.append(dict(d.artifact(audit_path), candidate_index=candidate,
                             metrics_csv=str(path), metrics_csv_sha256=c.sha256_file(path)))
        data, predictions, entries = {}, {}, []
        for policy in ["scalar_v3"] + [d.learned_id(m, s) for m in d.POLICIES for s in d.SEEDS]:
            values = copy.deepcopy(baseline)
            for value in values.values():
                value["score"] = .51
            data[policy] = dict(collection=dict(policy=policy), audit=d.artifact(audit_path), metrics=values)
            entries.append(dict(path=policy, sha256="synthetic"))
            predictions[policy] = {r["scene_id"]: 1 for r in rows}
        c.atomic_json(self.root / "phase_a_collections.json", dict(collections=entries))
        c.atomic_json(self.root / "deployment_inputs.json", dict(
            artifacts=dict(targets=d.artifact(targets_path), baseline_csv=d.artifact(baseline_path)),
            one_shot_arms=arms, oof_predictions=predictions))
        with patch.object(report_module, "checked_collection", side_effect=lambda entry: data[entry["path"]]):
            result = report(self.root, "phase_a")
            self.assertEqual(report(self.root, "phase_a"), result)
        self.assertTrue(result["phase_b_authorized"])
        self.assertFalse(result["expansion_authorized"])
        self.assertAlmostEqual(result["configurations"]["q_grpo_t1"]["mean_delta"]["score"], .01)
        self.assertAlmostEqual(result["configurations"]["q_grpo_t1"]["continuous_minus_one_shot"][0]["delta"]["score"], -.01)
        with (self.root / "phase_a_paired_scenes.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 768)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
