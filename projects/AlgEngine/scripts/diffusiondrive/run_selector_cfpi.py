"""Bounded, provenance-locked CFPI pilot orchestration. No expansion entrypoint."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import math
import signal
import subprocess
import tempfile
import time

import cfpi_common as c

SCRIPT = Path(__file__).resolve().parent
ROOT = SCRIPT.parents[3]
CANONICAL = Path("/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine")
STAGES = ("preflight", "source", "baseline_train", "freeze", "sentinel", "pilot",
          "pilot_audit", "train_cv", "report", "first_phase", "cv_continuation")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def environment(source_root=CANONICAL):
    env = os.environ.copy()
    alg_env = Path(env.get("DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE",
                          "/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine"))
    env.update(ALGENGINE_PYTHON=str(alg_env / "bin/python"),
               SIMENGINE_PYTHON=env.get("DIFFUSIONDRIVE_SIMENGINE_PYTHON", "/root/miniconda3/envs/simengine/bin/python"),
               WORLDENGINE_ROOT=str(ROOT), SOURCE_WORLDENGINE_ROOT=str(source_root),
               ALGENGINE_ROOT=str(ROOT / "projects/AlgEngine"), SIMENGINE_ROOT=str(ROOT / "projects/SimEngine"),
               WORLDENGINE_DIFFUSIONDRIVE_MMCV_BOOTSTRAP="1", WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_BOOTSTRAP="1",
               WORLDENGINE_MMCV_EXTENSION=str(source_root / "artifacts/toolchains/mmcv_sm89_sm90_v1/mmcv/_ext.so"),
               WORLDENGINE_GSPLAT_EXTENSION=str(source_root / "artifacts/toolchains/gsplat_sm89_sm90_v1/gsplat/csrc.so"),
               OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "8"))
    env.update(DIFFUSIONDRIVE_GRPO_V3_MODEL_DIM="256", DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE="1.0")
    support = ROOT / "projects/SimEngine/scripts/diffusiondrive"
    paths = [support / "algengine_worker_bootstrap", support, ROOT / "projects/AlgEngine",
             ROOT / "projects/SimEngine", source_root.parent / "DiffusionDrive", source_root.parent / "mmcv",
             Path("/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/nuplan-devkit"),
             Path("/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/E2E/navsim_v1"), SCRIPT]
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = str(alg_env / "bin") + os.pathsep + env["PATH"]
    env.pop("RAY_ADDRESS", None)
    env["RAY_TMPDIR"] = tempfile.mkdtemp(prefix="cfpi-ray-")
    env["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="cfpi-mpl-")
    return env


def implementation_inventory():
    # Explicitly cover imported local helpers, configs, controller and simulator;
    # inherited results/documents are not consulted.
    folders = [SCRIPT, ROOT / "projects/AlgEngine/mmdet3d_plugin/navformer",
               ROOT / "projects/AlgEngine/configs/diffusiondrive",
               ROOT / "projects/AlgEngine/closed_loop", ROOT / "projects/SimEngine/worldengine",
               ROOT / "projects/SimEngine/scripts", ROOT / "research/cfpi"]
    files = {p for folder in folders for p in folder.rglob("*")
             if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".json", ".md"}}
    files.add(ROOT / "run_diffusiondrive_selector_cfpi_8h100.sh")
    return {str(p.relative_to(ROOT)): c.sha256_file(p) for p in sorted(files)}


def validate_cv_source(source_run):
    source_run = Path(source_run).expanduser().resolve()
    if not source_run.is_dir():
        raise RuntimeError(f"Missing frozen pilot source run: {source_run}")
    source_contract = json.loads((source_run / "run_contract.json").read_text())
    ledger = json.loads((source_run / "decision_ledger.json").read_text())
    if not isinstance(source_contract, dict) or not isinstance(ledger, dict):
        raise RuntimeError("Invalid frozen pilot source contract/ledger")
    gate = c.verified_json(source_run / "pilot_gate.json")
    pilot_audit = c.verified_json(source_run / "cache/pilot64_cache_audit.json", "cache_file")
    repeat_audit = c.verified_json(source_run / "cache/repeat8_cache_audit.json", "cache_file")
    pilot = c.load_pickle(pilot_audit["cache_file"])
    rows = c.validate_cache(pilot)
    repeat = c.load_pickle(repeat_audit["cache_file"])
    if (gate.get("status") != "PASS" or gate.get("training_authorized") is not True
            or gate.get("information_gate_pass") is not True
            or gate.get("expansion_authorized") is not False
            or gate.get("development_consumed") is not False
            or gate.get("test_consumed") is not False):
        raise RuntimeError("Frozen pilot gate does not authorize train-only CV")
    if (gate["cache_sha256"] != pilot_audit["cache_file_sha256"]
            or gate["repeat_cache_sha256"] != repeat_audit["cache_file_sha256"]):
        raise RuntimeError("Frozen pilot gate/cache lineage drifted")
    if (len(rows) != 64 or repeat.get("method") != c.CACHE_METHOD
            or repeat.get("stage") != "repeat8" or repeat.get("num_rows") != 8
            or repeat.get("status") != "PASS"):
        raise RuntimeError("Frozen pilot/repeat cache contract is incomplete")
    if (source_contract.get("code_sha") != pilot.get("collection_code_sha")
            or source_contract.get("design_version") != c.DESIGN_VERSION):
        raise RuntimeError("Frozen pilot collection code contract drifted")
    if (ledger.get("active") is not None or ledger.get("development_consumed") is not False
            or ledger.get("test_consumed") is not False
            or ledger.get("expansion_authorized") is not False):
        raise RuntimeError("Frozen pilot source ledger is not safely stopped")
    files = {
        "source_run_contract": source_run / "run_contract.json",
        "source_decision_ledger": source_run / "decision_ledger.json",
        "pilot_gate": source_run / "pilot_gate.json",
        "pilot_cache": Path(pilot_audit["cache_file"]),
        "pilot_cache_audit": source_run / "cache/pilot64_cache_audit.json",
        "repeat_cache": Path(repeat_audit["cache_file"]),
        "repeat_cache_audit": source_run / "cache/repeat8_cache_audit.json",
    }
    return dict(source_run=str(source_run), source_gpu_hour_limit=float(source_contract["gpu_hour_limit"]),
                source_gpu_hours_used=float(ledger["gpu_hours_used"]),
                remaining_gpu_hours=float(source_contract["gpu_hour_limit"]) - float(ledger["gpu_hours_used"]),
                information_gate_improvable_count=int(gate["improvable_above_0p02"]),
                repeat_max_absolute_error=float(gate["repeat_max_absolute_error"]),
                files={key: dict(path=str(path.resolve()), sha256=c.sha256_file(path))
                       for key, path in files.items()})


def link_cv_inputs(target_run, lineage):
    target_run = Path(target_run).resolve()
    for name, artifact in lineage["files"].items():
        if c.sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"Frozen CV source changed before linking: {name}")
    links = {
        target_run / "pilot_gate.json": lineage["files"]["pilot_gate"]["path"],
        target_run / "cache/pilot64_cache.pkl": lineage["files"]["pilot_cache"]["path"],
        target_run / "cache/pilot64_cache_audit.json": lineage["files"]["pilot_cache_audit"]["path"],
        target_run / "cache/repeat8_cache.pkl": lineage["files"]["repeat_cache"]["path"],
        target_run / "cache/repeat8_cache_audit.json": lineage["files"]["repeat_cache_audit"]["path"],
    }
    for target, source in links.items():
        source = Path(source).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source:
                raise RuntimeError(f"Refusing to replace CV continuation input: {target}")
        else:
            target.symlink_to(source)
    c.locked_json(target_run / "cv_continuation_manifest.json", lineage)


class Runner:
    def __init__(self, args):
        self.args = args
        self.run = ROOT / "experiments/diffusiondrive/selector_cfpi_v1/runs" / args.run_id
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run / "runner.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.env = environment(args.source_worldengine_root)
        self.alg = Path(self.env["ALGENGINE_ROOT"])
        self.sim = Path(self.env["SIMENGINE_ROOT"])
        self.config = self.alg / "configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_cfpi_collect.py"
        self.protocol = ROOT / "research/cfpi/PROTOCOL.md"
        self.exclusions = ROOT / "research/cfpi/exclusions.json"
        self.source = args.source_worldengine_root / "experiments/grpo_sources/diffusiondrive_v4_split0"
        self.checkpoint_manifest = args.source_worldengine_root / (
            "experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json")
        self.cv_lineage = validate_cv_source(args.pilot_source_run) if args.stage == "cv_continuation" else None
        if self.cv_lineage is not None and args.gpu_hours > self.cv_lineage["remaining_gpu_hours"] + 1e-9:
            raise RuntimeError("CV continuation GPU budget exceeds frozen source-run remainder")
        self.inventory = implementation_inventory()
        self.code_sha = hashlib.sha256(json.dumps(self.inventory, sort_keys=True).encode()).hexdigest()
        contract = dict(design_version=c.DESIGN_VERSION, code_sha=self.code_sha, implementation=self.inventory,
                        protocol_sha256=c.sha256_file(self.protocol), checkpoint_sha256=c.CHECKPOINT_SHA256,
                        checkpoint_manifest_sha256=c.sha256_file(self.checkpoint_manifest),
                        source_worldengine_root=str(args.source_worldengine_root.resolve()),
                        noise_namespace=c.NOISE_NAMESPACE, gpu_count=args.gpus,
                        gpu_hour_limit=args.gpu_hours, endpoint="pilot_report", expansion_authorized=False)
        if self.cv_lineage is not None:
            contract.update(execution_scope="train_cv_and_report_only",
                            frozen_pilot_lineage=self.cv_lineage)
        c.locked_json(self.run / "run_contract.json", contract)
        self.ledger_path = self.run / "decision_ledger.json"
        self.ledger = (json.loads(self.ledger_path.read_text()) if self.ledger_path.exists()
                       else dict(run_id=args.run_id, gpu_hours_used=0.0, events=[],
                                 development_consumed=False, test_consumed=False, expansion_authorized=False))
        if self.ledger.get("active"):
            # Conservatively charge the unaccounted interval after an unclean stop.
            previous = self.ledger["active"]
            elapsed = max(0.0, time.time() - previous["last_heartbeat_unix"])
            self.ledger["gpu_hours_used"] += elapsed * previous["gpus"] / 3600
            self.ledger["active"] = None
        self.children = []
        signal.signal(signal.SIGTERM, self.interrupt)
        signal.signal(signal.SIGINT, self.interrupt)

    def interrupt(self, signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    def record(self, stage, status, detail=None):
        self.ledger["events"].append(dict(time=utc(), stage=stage, status=status, detail=detail))
        c.atomic_json(self.ledger_path, self.ledger)
        print(json.dumps(dict(time=utc(), stage=stage, status=status, detail=detail)), flush=True)

    def stop_children(self):
        for process, stream in self.children:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 10
        while any(p.poll() is None for p, _ in self.children) and time.monotonic() < deadline:
            time.sleep(.2)
        for process, stream in self.children:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            stream.close()
        self.children = []

    def execute(self, jobs, stage, gpus=0, timeout=21600):
        """Jobs: (argv, cwd, env, log_path). Kill only process groups we own."""
        if self.ledger["gpu_hours_used"] >= self.args.gpu_hours and gpus:
            raise RuntimeError("GPU-hour budget exhausted; preserving partial progress")
        start = last = time.time()
        log_tick = start
        self.record(stage, "RUNNING")
        try:
            for command, cwd, env, log in jobs:
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open("a")
                process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                self.children.append((process, stream))
            while True:
                now = time.time()
                self.ledger["gpu_hours_used"] += max(0, now - last) * gpus / 3600
                last = now
                self.ledger["active"] = dict(stage=stage, gpus=gpus, last_heartbeat_unix=now,
                                              owned_pids=[p.pid for p, _ in self.children])
                c.atomic_json(self.ledger_path, self.ledger)
                failed = [p.returncode for p, _ in self.children if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"{stage} child failed: {failed}; inspect stage logs")
                if all(p.poll() == 0 for p, _ in self.children):
                    break
                if now - start > timeout:
                    raise TimeoutError(f"{stage} timeout; preserving outputs")
                if gpus and self.ledger["gpu_hours_used"] >= self.args.gpu_hours:
                    raise RuntimeError("GPU-hour budget reached; resume checkpoints preserved")
                if now - log_tick >= 30:
                    print(json.dumps(dict(stage=stage, elapsed_seconds=int(now-start),
                                          gpu_hours_used=round(self.ledger["gpu_hours_used"], 3))), flush=True)
                    log_tick = now
                time.sleep(1)
            self.record(stage, "COMPLETE")
        finally:
            self.stop_children()
            self.ledger["active"] = None
            c.atomic_json(self.ledger_path, self.ledger)

    def python(self, script, arguments=(), stage=None, sim=False, env=None, gpus=0, timeout=21600):
        stage = stage or Path(script).stem
        env = env or self.env
        executable = env["SIMENGINE_PYTHON" if sim else "ALGENGINE_PYTHON"]
        self.execute([([executable, script, *arguments], self.sim if sim else ROOT, env,
                       self.run / "logs" / (stage + ".log"))], stage, gpus, timeout)

    def preflight(self):
        try:
            self._preflight()
        except Exception as error:
            c.atomic_json(self.run / "preflight.json",
                          dict(status="FAIL", failures=[str(error)], checked_at=utc()))
            raise

    def _preflight(self):
        failures = []
        for name in ("ALGENGINE_PYTHON", "SIMENGINE_PYTHON", "WORLDENGINE_MMCV_EXTENSION", "WORLDENGINE_GSPLAT_EXTENSION"):
            if not Path(self.env[name]).is_file():
                failures.append(f"Missing {name}: {self.env[name]}")
        try:
            allocation = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
            if allocation.returncode:
                failures.append("GPU driver unavailable: " + allocation.stderr.strip() + allocation.stdout.strip())
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append("GPU driver check failed: " + str(error))
        assets = self.args.source_worldengine_root / "data/sim_engine/assets/navtrain/assets"
        if not assets.is_dir():
            failures.append(f"Missing assets: {assets}")
        if shutil.disk_usage(self.run).free < 200 * 1024**3:
            failures.append("Less than 200 GiB free for the first-phase cache and collections")
        if failures:
            c.atomic_json(self.run / "preflight.json", dict(status="FAIL", failures=failures, checked_at=utc()))
            raise RuntimeError("; ".join(failures))
        # Validate the allocation before linking data or running GPU kernels.
        self.python(SCRIPT / "selector_cfpi_preflight.py", ["--gpus", str(self.args.gpus)],
                    stage="cuda_preflight", timeout=300)
        target = ROOT / "data"
        source = self.args.source_worldengine_root / "data"
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError("Refusing to replace worktree data path")
        else:
            target.symlink_to(source, target_is_directory=True)
        manifest = json.loads(self.checkpoint_manifest.read_text())
        if c.sha256_file(manifest["checkpoint"]) != c.CHECKPOINT_SHA256:
            raise RuntimeError("Incumbent checkpoint bytes changed")
        support = self.sim / "scripts/diffusiondrive"
        self.python(support / "preflight_mmcv_cuda.py",
                    ["--extension", self.env["WORLDENGINE_MMCV_EXTENSION"], "--all-visible",
                     "--expected-capability", "sm_90"], stage="mmcv_preflight", gpus=self.args.gpus, timeout=600)
        self.python(support / "preflight_gsplat_cuda.py",
                    ["--extension", self.env["WORLDENGINE_GSPLAT_EXTENSION"],
                     "--expected-capability", "sm_90", "--ray-workers", str(self.args.gpus)],
                    stage="gsplat_preflight", sim=True, gpus=self.args.gpus, timeout=600)
        c.atomic_json(self.run / "preflight.json", dict(status="PASS", checked_at=utc(), gpus=self.args.gpus))

    def prepare(self, stage):
        self.python(SCRIPT / "prepare_selector_cfpi.py", [stage, "--run-root", self.run,
                    "--source-root", self.source, "--exclusions", self.exclusions, "--protocol", self.protocol],
                    stage=stage)

    def collection(self, collection_id, scenario, treatment=None):
        root = self.run / "collections" / collection_id
        root.mkdir(parents=True, exist_ok=True)
        env = self.env.copy()
        mode = "one_shot_manifest" if treatment else "observe_only"
        manifest = json.loads(self.checkpoint_manifest.read_text())
        treatment_sha = c.sha256_file(treatment) if treatment else "none"
        env.update(DIFFUSIONDRIVE_BEHAVIOR_POLICY_TRAIN_SEED="0", DIFFUSIONDRIVE_CFPI_COLLECTION_ID=collection_id,
                   DIFFUSIONDRIVE_CFPI_SOURCE_SPLIT="train", DIFFUSIONDRIVE_CFPI_INTERVENTION_MODE=mode,
                   DIFFUSIONDRIVE_CFPI_TREATMENT_MANIFEST_SHA256=treatment_sha,
                   DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256=c.CHECKPOINT_SHA256,
                   DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256=c.sha256_file(self.checkpoint_manifest),
                   DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE=c.NOISE_NAMESPACE,
                   DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256=self.code_sha,
                   DIFFUSIONDRIVE_ROLLOUT_CODE_SHA=self.code_sha)
        collection_contract = dict(code_sha=self.code_sha, scenario_sha256=c.sha256_file(scenario),
                                   treatment_sha256=treatment_sha, mode=mode,
                                   checkpoint_sha256=c.CHECKPOINT_SHA256, noise=c.NOISE_NAMESPACE)
        c.locked_json(root / "collection_contract.json", collection_contract)
        final = root / "collection_audit.json"
        audit_args = ["--rollout-root", root, "--scenario-file", scenario,
                      "--checkpoint-manifest", self.checkpoint_manifest, "--collection-id", collection_id,
                      "--source-split", "train", "--mode", mode, "--treatment-sha256", treatment_sha,
                      "--expected-noise-namespace", c.NOISE_NAMESPACE, "--expected-implementation-sha256", self.code_sha,
                      "--expected-code-sha", self.code_sha, "--expected-workers", str(self.args.gpus)]
        if treatment:
            audit_args += ["--treatment-manifest", treatment]
        metrics = root / "WE_output/openscene_format/all_scenes_pdm_averages_R.csv"
        if final.exists():
            self.python(SCRIPT / "audit_selector_cfpi_collection.py",
                        [*audit_args, "--layout", "merged", "--metrics-csv", metrics,
                         "--output", root / "resume_audit.json"], stage=collection_id + "_revalidate", env=env)
            return
        self.python(SCRIPT / "selector_cfpi_preflight.py", ["--config", self.config],
                    stage=collection_id + "_config", env=env, timeout=180)
        # Completion flags from an interrupted prior merge are retained recoverably.
        for flag in root.glob("split_*/**/simulation_completed.flag"):
            flag.rename(flag.with_name("simulation_completed.flag.previous." + str(time.time_ns())))
        we = [env["SIMENGINE_PYTHON"], self.sim / "worldengine/runner/run_simulation.py",
              "debug_mode=True", "debug_scene_name=null", f"data_file_path={scenario}",
              f"asset_folder_path={self.args.source_worldengine_root}/data/sim_engine/assets/navtrain/assets",
              f"output_dir={root}/__WORKER_ID__/WE_output", f"job_name=selector_cfpi_{collection_id}",
              "use_planner_actions=true", "ego_policy=env_input_policy", "ego_client=navformer_client",
              "ego_controller=log_play_controller", "ego_navigation=trajectory_navigation",
              "agent_policy=idm_policy", "agent_navigation=idm_navigation",
              f"planner_data_path={root}/__WORKER_ID__/plan_traj",
              f"planner_client_folder={root}/__WORKER_ID__/frames",
              "with_metric_manager=true", "with_dense_reward_manager=true",
              "diffusiondrive_cfpi_causal_cache=true", "diffusiondrive_v4_causal_cache=false",
              "diffusiondrive_candidate_sweep=false", "diffusiondrive_preaction_oracle=false",
              "diffusiondrive_dynamic_candidate_reward=false", "diffusiondrive_behavior_train_seed=0",
              f"diffusiondrive_cfpi_intervention_mode={mode}", f"diffusiondrive_cfpi_collection_id={collection_id}",
              f"diffusiondrive_cfpi_treatment_manifest_sha256={treatment_sha}",
              f"diffusiondrive_candidate_sidecar_path={root}/__WORKER_ID__/diffusiondrive_candidate_sidecars",
              "diffusiondrive_sidecar_timeout_s=300", "distributed_mode=SCENARIO_BASED",
              "worker=ray_distributed", "worker_id_prefix=split_", "enable_resume=true",
              f"completed_scenarios_dir={root}/__WORKER_ID__/completed_scenarios"]
        if treatment:
            we += [f"diffusiondrive_cfpi_treatment_manifest={treatment}"]
        jobs = [(we, self.sim, env, root / "logs/worldengine.log")]
        devices = env.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(self.args.gpus)))).split(",")
        if len(devices) != self.args.gpus:
            raise RuntimeError("CUDA device mapping does not match worker count")
        for i, device in enumerate(devices):
            worker = root / f"split_{i}"
            for directory in ("plan_traj", "frames", "merged_ann_files", "diffusiondrive_candidate_sidecars"):
                (worker / directory).mkdir(parents=True, exist_ok=True)
            planner_env = dict(env, CUDA_VISIBLE_DEVICES=device)
            cmd = [env["ALGENGINE_PYTHON"], self.alg / "closed_loop/sim_test.py", self.config,
                   manifest["checkpoint"], "--seed", "0", "--log-dir", worker, "--cfg-options",
                   f"sim.monitored_folder={worker}/frames", f"sim.plan_save_path={worker}/plan_traj",
                   f"sim.merged_ann_save_dir={worker}/merged_ann_files",
                   f"sim.diffusiondrive_rollout_sidecar_path={worker}/diffusiondrive_candidate_sidecars",
                   "sim.clean_temp_files=False", "sim.clean_record_data=False",
                   f"data_root={worker}/WE_output/openscene_format/"]
            jobs.append((cmd, self.alg, planner_env, root / f"logs/planner_split{i}.log"))
        # Both sides have file-based readiness waits; monitor all processes together.
        self.execute(jobs, collection_id, self.args.gpus)
        self.python(SCRIPT / "audit_selector_cfpi_collection.py",
                    [*audit_args, "--layout", "split", "--output", root / "premerge_collection_audit.json"],
                    stage=collection_id + "_premerge", env=env)
        merged = root / "WE_output"
        if merged.exists():
            merged.rename(root / ("WE_output.previous." + str(time.time_ns())))
        self.python(self.sim / "scripts/merge_simulation_results.py",
                    ["--test_path", root, "--react_type", "R", "--num-splits", str(self.args.gpus)],
                    stage=collection_id + "_merge", sim=True, env=env)
        self.python(SCRIPT / "audit_selector_cfpi_collection.py",
                    [*audit_args, "--layout", "merged", "--metrics-csv", metrics, "--output", final],
                    stage=collection_id + "_audit", env=env)

    def sentinel(self):
        self.prepare("freeze")
        master = json.loads((self.run / "manifests/master_manifest.json").read_text())
        path = Path(master["collections"]["sentinel_policy"]["path"])
        _, treatment = c.load_treatment(path, master["collections"]["sentinel_policy"]["sha256"])
        self.collection("sentinel_policy", Path(treatment["scenario_file"]), path)
        self.python(SCRIPT / "verify_selector_cfpi_sentinel.py",
                    ["--baseline-audit", self.run / "collections/baseline_train_a/collection_audit.json",
                     "--sentinel-audit", self.run / "collections/sentinel_policy/collection_audit.json",
                     "--treatment-manifest", path, "--output", self.run / "sentinel_gate.json"])

    def pilot(self):
        sentinel = c.verified_json(self.run / "sentinel_gate.json")
        if sentinel.get("decision") != "AUTHORIZE_CFPI_PILOT64_CAUSAL_COLLECTION":
            raise RuntimeError("Sentinel did not authorize pilot collection")
        for key in ("baseline_audit", "sentinel_audit", "treatment_manifest"):
            if c.sha256_file(sentinel[key]) != sentinel[key + "_sha256"]:
                raise RuntimeError("Sentinel provenance changed")
        master = json.loads((self.run / "manifests/master_manifest.json").read_text())
        for stage in ("pilot64", "repeat8"):
            for i in range(20):
                collection_id = f"{stage}_arm_{i:02d}"
                row = master["collections"][collection_id]
                path, treatment = c.load_treatment(row["path"], row["sha256"])
                self.collection(collection_id, Path(treatment["scenario_file"]), path)
            self.python(SCRIPT / "assemble_selector_cfpi_cache.py",
                        ["--run-root", self.run, "--master-manifest", self.run / "manifests/master_manifest.json",
                         "--stage", stage, "--output", self.run / f"cache/{stage}_cache.pkl",
                         "--audit-output", self.run / f"cache/{stage}_cache_audit.json"], stage=stage + "_assemble")

    def train_cv(self):
        from selector_cfpi_objectives import METHODS
        from train_selector_cfpi import LEARNING_RATES, SEEDS, job_name
        gate = c.verified_json(self.run / "pilot_gate.json")
        if not gate.get("training_authorized"):
            self.record("train_cv", "SKIPPED", gate["decision"])
            return
        jobs = []
        devices = self.env.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(self.args.gpus)))).split(",")
        for method in METHODS:
            for lr in LEARNING_RATES:
                for seed in SEEDS:
                    for fold in range(4):
                        name = job_name(method, lr, fold, seed)
                        cmd = [self.env["ALGENGINE_PYTHON"], SCRIPT / "train_selector_cfpi.py",
                               "--cache", self.run / "cache/pilot64_cache.pkl",
                               "--cache-audit", self.run / "cache/pilot64_cache_audit.json",
                               "--pilot-gate", self.run / "pilot_gate.json",
                               "--checkpoint-manifest", self.checkpoint_manifest, "--output-root", self.run / "cv",
                               "--method", method, "--lr", str(lr), "--fold", str(fold), "--seed", str(seed)]
                        jobs.append((cmd, ROOT, dict(self.env, CUDA_VISIBLE_DEVICES=devices[len(jobs) % self.args.gpus]),
                                     self.run / "logs" / (name + ".log")))
        for start in range(0, len(jobs), self.args.gpus):
            batch = jobs[start:start + self.args.gpus]
            self.execute(batch, f"cv_batch_{start // self.args.gpus}", len(batch), timeout=7200)

    def dispatch(self):
        stage = self.args.stage
        if stage in {"preflight", "baseline_train", "sentinel", "pilot", "train_cv",
                     "first_phase", "cv_continuation"}:
            self.preflight()
        if stage == "cv_continuation":
            link_cv_inputs(self.run, self.cv_lineage)
            self.train_cv()
            self.python(SCRIPT / "report_selector_cfpi.py",
                        ["report", "--run-root", self.run], stage="pilot_report")
            self.record(stage, "COMPLETE", "CV-only continuation complete; no expansion authorized")
            return
        if stage == "source":
            self.prepare("source")
        if stage in {"baseline_train", "first_phase"}:
            self.prepare("source")
            scenario = self.run / "source/train_pool_1024.pkl"
            for label in ("a", "b"):
                self.collection(f"baseline_train_{label}", scenario)
        if stage in {"freeze", "first_phase"}:
            self.prepare("freeze")
        if stage in {"sentinel", "first_phase"}:
            self.sentinel()
        if stage in {"pilot", "first_phase"}:
            self.pilot()
        if stage in {"pilot_audit", "pilot", "first_phase"}:
            self.python(SCRIPT / "report_selector_cfpi.py", ["pilot_gate", "--run-root", self.run])
        if stage in {"train_cv", "first_phase"}:
            self.train_cv()
        if stage in {"report", "first_phase"}:
            self.python(SCRIPT / "report_selector_cfpi.py", ["report", "--run-root", self.run], stage="pilot_report")
        self.record(stage, "COMPLETE", "Stopped at requested pilot-stage boundary; no expansion authorized")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("run_id")
    parser.add_argument("--gpus", type=int, default=8, choices=(1, 2, 4, 8))
    parser.add_argument("--gpu-hours", type=float, default=144)
    parser.add_argument("--source-worldengine-root", type=Path, default=CANONICAL)
    parser.add_argument("--pilot-source-run", type=Path)
    args = parser.parse_args()
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id)
            or not math.isfinite(args.gpu_hours) or args.gpu_hours <= 0):
        parser.error("Invalid run ID or GPU budget")
    if (args.stage == "cv_continuation") != (args.pilot_source_run is not None):
        parser.error("--pilot-source-run is required only for cv_continuation")
    runner = Runner(args)
    try:
        runner.dispatch()
    except BaseException as error:
        runner.stop_children()
        runner.record(args.stage, "STOPPED", str(error))
        raise


if __name__ == "__main__":
    main()
