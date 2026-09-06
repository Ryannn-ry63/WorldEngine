"""Independent 8-H100 A/B runner; cumulative 64 GPUh maximum, no expansion."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import cfpi_common as c
import selector_cfpi_deployment_common as d
from run_selector_cfpi import Runner as PilotRunner, ROOT, SCRIPT, CANONICAL, environment, implementation_inventory, utc


def process_identity(pid):
    """Linux PID reuse protection: kernel start tick plus boot ID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=int(pid), start_ticks=fields[19], boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_session_members(identity):
    """Read-only orphan detection; never kill a process inferred from an old PID."""
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != identity["boot_id"]:
        return []
    leader = process_identity(identity["pid"])
    if leader is not None and leader != identity:
        return []  # PID was reused by an unrelated session.
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) == identity["pid"] and fields[0] != "Z":
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def charge(ledger, phase, seconds, gpus):
    amount = max(0.0, float(seconds)) * gpus / 3600.0
    ledger["gpu_hours_used"] += amount
    ledger["phase_gpu_hours"][phase] += amount


def check_budget(ledger, phase, total, reserve=0.0):
    if ledger["gpu_hours_used"] + reserve >= total or ledger["phase_gpu_hours"][phase] + reserve >= d.PHASE_LIMITS[phase]:
        raise RuntimeError(f"GPU-hour budget reached ({phase}); no expansion or budget reset authorized")


def archive_incomplete_collection(root):
    """Collection-level resume: never feed stale frames/actions back to the planner.

    Completed audited conditions are reused. An interrupted condition is retried
    from scene zero, with its entire previous output retained recoverably.
    """
    root = Path(root).resolve()
    if (root / "collection_audit.json").exists():
        raise RuntimeError("Refusing to archive an audited condition")
    targets = [p for p in root.iterdir() if p.is_dir() and
               (re.fullmatch(r"split_[0-7]", p.name) or p.name in {"__WORKER_ID__", "logs"})]
    targets += [p for p in root.glob("runner_report_*.json") if p.is_file()]
    if not targets:
        return None
    destination = root / "attempt_archive" / str(time.time_ns())
    destination.mkdir(parents=True)
    mapping = {str(p): str(destination / p.name) for p in targets}
    c.atomic_json(destination / "recovery_map.json", dict(status="ARCHIVED_INCOMPLETE", paths=mapping,
                  note="Stored payload paths retain the original prefix; use this map for historical inspection"))
    for source in targets:
        source.rename(destination / source.name)
    return str(destination)


class Runner(PilotRunner):
    def __init__(self, args):
        self.args = args
        self.run = ROOT / "experiments/diffusiondrive/selector_cfpi_deployment_v1/runs" / args.run_id
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run / "runner.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.env = environment(args.source_worldengine_root)
        self.alg, self.sim = Path(self.env["ALGENGINE_ROOT"]), Path(self.env["SIMENGINE_ROOT"])
        self.config = self.alg / "configs/diffusiondrive/e2e_diffusiondrive_selector_cfpi_deployment.py"
        self.checkpoint_manifest = args.source_worldengine_root / (
            "experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json")
        self.gate_manifest = args.source_worldengine_root / (
            "experiments/diffusiondrive/grpo_selector_v3_gate_conditioned_v1/models/seed0/checkpoint_manifest.json")
        self.phase = "phase_b" if args.stage == "phase_b" else "phase_a"
        self.inventory = implementation_inventory()
        self.inventory["run_diffusiondrive_selector_cfpi_deployment_8h100.sh"] = c.sha256_file(
            ROOT / "run_diffusiondrive_selector_cfpi_deployment_8h100.sh")
        self.code_sha = hashlib.sha256(json.dumps(self.inventory, sort_keys=True).encode()).hexdigest()
        contract = dict(method=d.METHOD, code_sha=self.code_sha, implementation=self.inventory,
                        pilot_run=str(args.pilot_run.resolve()), cv_run=str(args.cv_run.resolve()),
                        source_worldengine_root=str(args.source_worldengine_root.resolve()),
                        checkpoint_sha256=c.CHECKPOINT_SHA256, noise_namespace=c.NOISE_NAMESPACE,
                        gpu_count=args.gpus, gpu_hour_limit=args.gpu_hours, phase_limits=d.PHASE_LIMITS,
                        policies=d.POLICIES, seeds=list(d.SEEDS), advancement=d.ADVANCEMENT,
                        phase_a_core_rollouts=768, phase_a_sentinel_rollouts=8, phase_b_rollouts=1792,
                        allocation_accounting="all 8 GPUs reserved during every launched stage, including CPU preparation",
                        endpoint="phase_b_report", expansion_authorized=False,
                        development_consumed=False, test_consumed=False)
        # A new run ID cannot overwrite or mutate old pilot contracts.
        c.locked_json(self.run / "run_contract.json", contract)
        self.ledger_path = self.run / "decision_ledger.json"
        self.ledger = (d.read(self.ledger_path) if self.ledger_path.exists() else
                       dict(run_id=args.run_id, gpu_hours_used=0.0, phase_gpu_hours={k: 0.0 for k in d.PHASE_LIMITS},
                            events=[], active=None, expansion_authorized=False, development_consumed=False, test_consumed=False))
        self.children = []
        self.recover_accounting()
        signal.signal(signal.SIGINT, self.interrupt)
        signal.signal(signal.SIGTERM, self.interrupt)

    def recover_accounting(self):
        previous = self.ledger.get("active")
        if not previous:
            return
        charge(self.ledger, previous["phase"], time.time() - previous["last_heartbeat_unix"], previous["gpus"])
        previous["last_heartbeat_unix"] = time.time()
        c.atomic_json(self.ledger_path, self.ledger)
        live = [pid for item in previous.get("owned_processes", []) for pid in owned_session_members(item)]
        if live:
            raise RuntimeError(f"Previous owned session members still live: {live}; stop them before resume")
        self.ledger["events"].append(dict(time=utc(), stage="crash_recovery", status="CHARGED_UNKNOWN_INTERVAL",
                                           detail="Conservative elapsed allocation charge; no free budget reset"))
        self.ledger["active"] = None
        c.atomic_json(self.ledger_path, self.ledger)

    def execute(self, jobs, stage, gpus=0, timeout=21600):
        # Charge the entire allocated instance, including small refit batches.
        allocated = self.args.gpus
        reserve = 15.0 * allocated / 3600.0  # polling + owned-group shutdown grace
        check_budget(self.ledger, self.phase, self.args.gpu_hours, reserve)
        start = last = tick = time.monotonic()
        self.record(stage, "RUNNING")
        self.ledger["active"] = dict(stage=stage, phase=self.phase, gpus=allocated,
                                     last_heartbeat_unix=time.time(), owned_processes=[])
        c.atomic_json(self.ledger_path, self.ledger)
        try:
            for command, cwd, env, log in jobs:
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open("a")
                try:
                    process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close()
                    raise
                self.children.append((process, stream))
                identity = process_identity(process.pid)
                if identity:
                    self.ledger["active"]["owned_processes"].append(identity)
                c.atomic_json(self.ledger_path, self.ledger)
            while True:
                now = time.monotonic()
                charge(self.ledger, self.phase, now - last, allocated)
                last = now
                self.ledger["active"]["last_heartbeat_unix"] = time.time()
                c.atomic_json(self.ledger_path, self.ledger)
                failed = [p.returncode for p, _ in self.children if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"{stage} child failed: {failed}; inspect stage logs")
                if all(p.poll() == 0 for p, _ in self.children):
                    break
                check_budget(self.ledger, self.phase, self.args.gpu_hours, reserve)
                if now - start > timeout:
                    raise TimeoutError(f"{stage} timed out; preserving partial outputs")
                if now - tick >= 30:
                    print(json.dumps(dict(stage=stage, elapsed_seconds=int(now-start),
                          gpu_hours_used=self.ledger["gpu_hours_used"], phase_gpu_hours=self.ledger["phase_gpu_hours"])), flush=True)
                    tick = now
                time.sleep(1)
        finally:
            self.stop_children()
            charge(self.ledger, self.phase, time.monotonic() - last, allocated)
            self.ledger["active"] = None
            c.atomic_json(self.ledger_path, self.ledger)
        self.record(stage, "COMPLETE")

    def stop_children(self):
        # A failed parent may have living descendants in its original session.
        # These groups were created by this Runner, never global Ray/user jobs.
        groups = [process.pid for process, _ in self.children]
        def send(sig):
            for group in groups:
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        send(signal.SIGTERM)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            live = False
            for group in groups:
                try:
                    os.killpg(group, 0)
                    live = True
                except ProcessLookupError:
                    pass
            for process, _ in self.children:
                process.poll()
            if not live:
                break
            time.sleep(.1)
        send(signal.SIGKILL)
        for process, stream in self.children:
            process.wait()
            stream.close()
        self.children = []

    def prepare(self, stage):
        self.python(SCRIPT / "prepare_selector_cfpi_deployment.py", [stage, "--run-root", self.run,
                    "--pilot-run", self.args.pilot_run, "--cv-run", self.args.cv_run,
                    "--checkpoint-manifest", self.checkpoint_manifest, "--gate-manifest", self.gate_manifest],
                    stage="prepare_" + stage)

    def collection(self, entry):
        path = d.verify(entry)
        contract = d.read(path)
        root, collection_id = path.parent, contract["collection_id"]
        d.verify(contract["scenario"])
        d.verify(contract["routing"])
        env = dict(self.env,
                   DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_SHA256=c.CHECKPOINT_SHA256,
                   DIFFUSIONDRIVE_BEHAVIOR_CHECKPOINT_MANIFEST_SHA256=c.sha256_file(self.checkpoint_manifest),
                   DIFFUSIONDRIVE_ROLLOUT_NOISE_NAMESPACE=c.NOISE_NAMESPACE,
                   DIFFUSIONDRIVE_ROLLOUT_IMPLEMENTATION_SHA256=self.code_sha,
                   DIFFUSIONDRIVE_ROLLOUT_CODE_SHA=self.code_sha,
                   DIFFUSIONDRIVE_CFPI_DEPLOYMENT_COLLECTION_ID=collection_id,
                   DIFFUSIONDRIVE_CFPI_DEPLOYMENT_ROUTING=contract["routing"]["path"],
                   DIFFUSIONDRIVE_CFPI_DEPLOYMENT_ROUTING_SHA256=contract["routing"]["sha256"])
        if not (root / "collection_audit.json").exists():
            archived = archive_incomplete_collection(root)
            if archived:
                self.record(collection_id, "ARCHIVED_PARTIAL", archived + "; recoverable condition-level retry")
            self.python(SCRIPT / "selector_cfpi_deployment_preflight.py", ["--config", self.config],
                        stage=collection_id + "_config", env=env, timeout=300)
            command = [env["SIMENGINE_PYTHON"], self.sim / "worldengine/runner/run_simulation.py",
                "debug_mode=True", "debug_scene_name=null", f"data_file_path={contract['scenario']['path']}",
                f"asset_folder_path={self.args.source_worldengine_root}/data/sim_engine/assets/navtrain/assets",
                f"output_dir={root}/__WORKER_ID__/WE_output", f"job_name=cfpi_deploy_{collection_id}",
                "use_planner_actions=true", "ego_policy=env_input_policy", "ego_client=navformer_client",
                "ego_controller=log_play_controller", "ego_navigation=trajectory_navigation",
                "agent_policy=idm_policy", "agent_navigation=idm_navigation",
                f"planner_data_path={root}/__WORKER_ID__/plan_traj", f"planner_client_folder={root}/__WORKER_ID__/frames",
                "with_metric_manager=true", "with_dense_reward_manager=true", "diffusiondrive_cfpi_deployment=true",
                "diffusiondrive_cfpi_causal_cache=false", "diffusiondrive_v4_causal_cache=false",
                "diffusiondrive_preaction_oracle=false", "diffusiondrive_candidate_sweep=false",
                "diffusiondrive_dynamic_candidate_reward=false",
                f"diffusiondrive_cfpi_deployment_manifest={path}",
                f"diffusiondrive_cfpi_deployment_manifest_sha256={entry['sha256']}",
                f"diffusiondrive_candidate_sidecar_path={root}/__WORKER_ID__/diffusiondrive_candidate_sidecars",
                "diffusiondrive_sidecar_timeout_s=300", "distributed_mode=SCENARIO_BASED",
                "worker=ray_distributed", "worker_id_prefix=split_", "enable_resume=false",
                f"completed_scenarios_dir={root}/__WORKER_ID__/completed_scenarios"]
            jobs = [(command, self.sim, env, root / "logs/worldengine.log")]
            devices = env.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(self.args.gpus)))).split(",")
            if len(devices) != self.args.gpus or len(set(devices)) != self.args.gpus:
                raise RuntimeError("CUDA device map does not match 8 distinct workers")
            checkpoint = d.read(self.checkpoint_manifest)["checkpoint"]
            for i, device in enumerate(devices):
                worker = root / f"split_{i}"
                for directory in ("frames", "plan_traj", "merged_ann_files", "diffusiondrive_candidate_sidecars"):
                    (worker / directory).mkdir(parents=True, exist_ok=True)
                cmd = [env["ALGENGINE_PYTHON"], self.alg / "closed_loop/sim_test.py", self.config, checkpoint,
                       "--seed", "0", "--log-dir", worker, "--cfg-options",
                       f"sim.monitored_folder={worker}/frames", f"sim.plan_save_path={worker}/plan_traj",
                       f"sim.merged_ann_save_dir={worker}/merged_ann_files",
                       f"sim.diffusiondrive_rollout_sidecar_path={worker}/diffusiondrive_candidate_sidecars",
                       "sim.clean_temp_files=False", "sim.clean_record_data=False",
                       f"data_root={worker}/WE_output/openscene_format/"]
                jobs.append((cmd, self.alg, dict(env, CUDA_VISIBLE_DEVICES=device), root / f"logs/planner_split{i}.log"))
            self.execute(jobs, collection_id, gpus=self.args.gpus)
        self.python(SCRIPT / "audit_selector_cfpi_deployment.py", ["--collection", path],
                    stage=collection_id + "_audit", env=env)

    def report(self, phase):
        self.python(SCRIPT / "report_selector_cfpi_deployment.py", ["--run-root", self.run, "--phase", phase],
                    stage=phase + "_report")
        return d.read(self.run / f"{phase}_report.json")

    def dispatch(self):
        if self.args.stage == "report":
            # Report generation is CPU-only and remains available after any budget stop.
            from report_selector_cfpi_deployment import report
            results = [report(self.run, phase) for phase in d.PHASE_LIMITS]
            print(json.dumps([{k: r[k] for k in ("status", "phase", "decision", "missing") if k in r} for r in results]))
            return
        self.preflight()
        self.prepare("freeze")
        self.prepare("verify_baselines")
        if self.args.stage == "preflight":
            self.prepare("phase_a")
            self.python(SCRIPT / "selector_cfpi_deployment_preflight.py", ["--cached-run", self.run],
                        stage="cached_routing_cuda", timeout=900)
            self.record("preflight", "COMPLETE", "Frozen A/B inputs prepared; no rollout launched")
            return
        if self.args.stage == "phase_a":
            self.prepare("phase_a")
            self.python(SCRIPT / "selector_cfpi_deployment_preflight.py", ["--cached-run", self.run],
                        stage="cached_routing_cuda", timeout=900)
            for entry in d.read(self.run / "phase_a_collections.json")["collections"]:
                self.collection(entry)
            result = self.report("phase_a")
            self.record("phase_a", "COMPLETE", result["decision"] + "; phase boundary, not auto-running phase B")
            return
        result = self.report("phase_a")
        if not result["phase_b_authorized"]:
            self.record("phase_b", "NOT_AUTHORIZED", "No fixed policy passed phase A; no refit or rollout launched")
            return
        jobs = []
        devices = self.env.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(self.args.gpus)))).split(",")
        for method in d.POLICIES:
            for seed in d.SEEDS:
                policy = d.learned_id(method, seed)
                env = dict(self.env, CUDA_VISIBLE_DEVICES=devices[len(jobs) % self.args.gpus])
                cmd = [env["ALGENGINE_PYTHON"], SCRIPT / "refit_selector_cfpi_deployment.py", "--run-root", self.run,
                       "--method", method, "--seed", str(seed)]
                jobs.append((cmd, ROOT, env, self.run / "logs" / ("refit_" + policy + ".log")))
        for offset in range(0, len(jobs), self.args.gpus):
            self.execute(jobs[offset:offset + self.args.gpus], f"refit_batch_{offset // self.args.gpus}", timeout=7200)
        self.prepare("phase_b")
        for entry in d.read(self.run / "phase_b_collections.json")["collections"]:
            self.collection(entry)
        self.report("phase_b")
        self.record("phase_b", "COMPLETE", "A/B endpoint reached; no expansion authorized")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("preflight", "phase_a", "phase_b", "report"))
    p.add_argument("run_id")
    p.add_argument("--pilot-run", type=Path, required=True)
    p.add_argument("--cv-run", type=Path, required=True)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--gpu-hours", type=float, default=64)
    p.add_argument("--source-worldengine-root", type=Path, default=CANONICAL)
    args = p.parse_args()
    args.pilot_run = args.pilot_run.expanduser().resolve()
    args.cv_run = args.cv_run.expanduser().resolve()
    args.source_worldengine_root = args.source_worldengine_root.expanduser().resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", args.run_id) or args.gpus != 8:
        p.error("A/B requires a safe new run ID and exactly 8 H100 GPUs")
    if not math.isfinite(args.gpu_hours) or not 0 < args.gpu_hours <= 64:
        p.error("Cumulative GPU budget must be in (0, 64]; A<=24 and B<=40 are also enforced")
    runner = Runner(args)
    try:
        runner.dispatch()
    except BaseException as error:
        runner.stop_children()
        runner.record(args.stage, "STOPPED", str(error))
        raise


if __name__ == "__main__":
    main()
