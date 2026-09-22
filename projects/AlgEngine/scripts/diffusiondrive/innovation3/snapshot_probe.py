"""Real train-source SimEngine branch parity; no renderer, reward or learning."""
import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from .paths import checked_path, sha256_file
from worldengine.online.headless import HeadlessSimulator, diagnostic_candidates
from worldengine.online.state import SnapshotCodec, SnapshotParityError, structural_hash


class IDMFailures(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        if 'IDM bug! fall back' in record.getMessage():
            self.count += 1


_worker_sim = None
_worker_failures = None


def runtime_evidence():
    """Only explicit numeric-runtime settings; never dump the ambient environment."""
    import scipy
    from threadpoolctl import threadpool_info
    code = Path(__file__).resolve().parents[5]
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=code).decode().split('\0')
    sources = {name: sha256_file(code/name) for name in tracked
               if name and (name.startswith('projects/SimEngine/worldengine/') or
                            name.startswith('projects/AlgEngine/scripts/diffusiondrive/innovation3'))}
    cpu = Path('/proc/cpuinfo')
    model = next((line.split(':', 1)[1].strip() for line in cpu.read_text().splitlines()
                  if line.startswith('model name')), None) if cpu.exists() else None
    return dict(python=sys.executable, python_version=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), cpu_model=model, threadpools=threadpool_info(),
                environment={key: os.environ.get(key) for key in
                             ('PYTHONHASHSEED', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                              'OPENBLAS_CORETYPE', 'MKL_CBWR', 'NPY_DISABLE_CPU_FEATURES')},
                code_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip(),
                source_sha256=sources)


def save_parity_failure(error, output):
    target = output.with_suffix('.failure')
    target.mkdir(exist_ok=False)
    artifacts = {'static_bundle.pkl': error.bundle,
                 'replay.pkl': pickle.dumps(dict(before=error.before, expected=error.expected,
                                                actual=error.actual, action=error.action), protocol=5)}
    checksums = {}
    for name, data in artifacts.items():
        with (target/name).open('xb') as stream:
            stream.write(data)
        checksums[name] = hashlib.sha256(data).hexdigest()
    details = dict(error.details, artifacts_sha256=checksums,
                   trusted_local_pickle_only=True)
    (target/'diagnosis.json').write_text(json.dumps(details, indent=2)+'\n')
    return str(target)


def worker_init(scene_id, scene, reaction, steps, seed, bundle):
    global _worker_sim, _worker_failures
    _worker_failures = IDMFailures()
    logging.getLogger().addHandler(_worker_failures)
    _worker_sim = HeadlessSimulator(scene_id, scene, reaction, steps, seed)
    _worker_sim.codec = SnapshotCodec(bundle=bundle)


def worker_step(snapshot, action):
    _worker_sim.restore(snapshot)
    count = _worker_failures.count
    states = _worker_sim.step(action)
    _worker_sim.codec.audit_static()
    result = _worker_sim.snapshot()
    return dict(state_hash=result.state_hash, snapshot=result, states=states,
                idm_fallbacks=_worker_failures.count - count)


def run_scene(scene_id, scene, reaction, steps, seed, progress):
    started = time.monotonic()
    failures = IDMFailures()
    logging.getLogger().addHandler(failures)
    sim = HeadlessSimulator(scene_id, scene, reaction, steps, seed)
    pool = None
    result = dict(scene_id=scene_id, reaction=reaction, steps=[], seed=seed,
                  initial_agents=len(sim.engine.agents), initial_dynamic_agents=len(sim.engine.agent_manager._dynamic_agents),
                  controller='two_stage_controller', motion_model='kinematic_bicycle', dt_s=.5)
    try:
        # Real OS spawn, never fork a renderer/CUDA process. One persistent worker
        # checks transport parity; this is not yet the throughput branch pool.
        pool = mp.get_context('spawn').Pool(1, worker_init, (scene_id, scene, reaction, steps, seed, sim.codec.bundle))
        for decision in range(steps):
            tick = time.monotonic()
            actions = diagnostic_candidates(sim)
            action_hash = structural_hash(actions)
            selected = (12 + 7 * decision) % 20  # predeclared, independent of feedback
            before_agents = set(sim.engine.agents)
            before, main, branches = sim.branch_group(actions, selected)
            group_seconds = time.monotonic() - tick
            if structural_hash(actions) != action_hash:
                raise AssertionError('Candidate bank was mutated during branch rollout')
            record = dict(decision=decision, selected=selected, candidate_count=20,
                          before_hash=before.state_hash, main_hash=main.state_hash,
                          branch_hashes=[x[0].state_hash for x in branches],
                          selected_parity=main.state_hash == branches[selected][0].state_hash,
                          snapshot_bytes=len(before.payload), serial_group_seconds=group_seconds,
                          main_ego=branches[selected][1]['ego'],
                          spawned=sorted(set(sim.engine.agents)-before_agents),
                          despawned=sorted(before_agents-set(sim.engine.agents)))
            ego_states = [b[1]['ego'] for b in branches]
            record['distinct_ego_states'] = len({structural_hash(s) for s in ego_states})
            record['next_speed_range_mps'] = [float(f([np.linalg.norm(s['velocity']) for s in ego_states])) for f in (min, max)]
            record['candidate_position_spread_m'] = float(np.ptp([s['position'] for s in ego_states], axis=0).max())
            if record['distinct_ego_states'] < 2:
                raise AssertionError('Diagnostic candidate interventions had no physical effect')
            if decision in (0, steps-1):
                # Prove candidate-order independence including spawn RNG and
                # hidden navigation/controller state, then restore canonical.
                try:
                    for index in reversed(range(20)):
                        sim.restore(before)
                        sim.step(actions[index])
                        replay = sim.snapshot()
                        if replay.state_hash != branches[index][0].state_hash:
                            raise sim.codec.mismatch('Branch ordering changed state: ' + str(index),
                                                     before, branches[index][0], replay, actions[index])
                finally:
                    sim.restore(main)
                    sim.codec.audit_static()
                record['reverse_order_parity'] = True
                remote = pool.apply_async(worker_step, (before, actions[selected])).get(timeout=180)
                record['spawn_process_parity'] = remote['state_hash'] == main.state_hash
                if not record['spawn_process_parity']:
                    raise sim.codec.mismatch('Spawn-process parity check failed',
                                             before, main, remote['snapshot'], actions[selected])
                if remote['idm_fallbacks']:
                    raise AssertionError('Spawn-process IDM fallback check failed')
            retained = sim.snapshot()
            if retained.state_hash != main.state_hash:
                raise sim.codec.mismatch('Canonical next state was not preserved',
                                         before, main, retained, actions[selected])
            if failures.count:
                raise AssertionError('IDM silently fell back; not valid reactive evidence')
            record['total_seconds'] = time.monotonic() - tick
            result['steps'].append(record)
            progress(dict(event='decision_pass', scene_id=scene_id, reaction=reaction,
                          decision=decision, selected_parity=True,
                          group_seconds=group_seconds, snapshot_bytes=len(before.payload)))
        result.update(status='PASS_HEADLESS_DYNAMICS_ONLY', idm_fallbacks=failures.count,
                      elapsed_seconds=time.monotonic()-started)
        return result
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        sim.close()
        logging.getLogger().removeHandler(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene-count', type=int, default=2)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.scene_count <= 8 or not 1 <= args.steps <= 16 or args.seed < 0:
        parser.error('Bounded probe: scene-count 1..8, steps 1..16, nonnegative seed')
    output = checked_path(args.output, must_exist=False)
    code = Path(__file__).resolve().parents[5]
    if output == code or code in output.parents:
        parser.error('Write private reports outside code')
    if output.exists():
        parser.error('Report exists; choose a fresh attempt')
    output.parent.mkdir(parents=True, exist_ok=True)
    events = output.with_suffix('.events.jsonl')
    report = dict(status='RUNNING', checks=[], errors=[], candidate_source='current_state_analytic_diagnostic',
                  source_split='navtrain_failures_per1', scene_selection='lexicographic_first_without_reward_filter',
                  frozen_generator_verified=False, rendered_main_parity_verified=False,
                  reward_verified=False, closed_loop_verified=False, formal_ready=False,
                  training_split_log_disjoint_verified=False, used_for_training=False)
    # Reserve both outputs before starting; keep partial reports on exceptions.
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    with events.open('x') as event_stream:
        def progress(item):
            event_stream.write(json.dumps(item) + '\n')
            event_stream.flush()
            print(json.dumps(item), flush=True)

        start = time.monotonic()
        try:
            report['runtime'] = runtime_evidence()
            cfg = json.loads(checked_path(args.settings).read_text())
            source = checked_path(Path(cfg['scenario_root']) / 'original/navtrain_failures_per1/all_scenarios.pkl')
            report['source'] = dict(path=str(source), bytes=source.stat().st_size,
                                    sha256=sha256_file(source))
            # This is a trusted, installed simulation dataset. Do not expose this
            # pickle interface to untrusted remote inputs.
            with source.open('rb') as stream:
                all_scenes = pickle.load(stream)
            selected = sorted(all_scenes)[:args.scene_count]
            scenes = {key: all_scenes[key] for key in selected}
            del all_scenes
            report['scene_ids'] = selected
            report['scene_sha256'] = {k: hashlib.sha256(pickle.dumps(v, protocol=5)).hexdigest() for k, v in scenes.items()}
            report['load_seconds'] = time.monotonic() - start
            progress(dict(event='scenes_loaded', selected=selected, seconds=report['load_seconds']))
            if len(scenes) != args.scene_count:
                raise ValueError('Not enough scenarios in registered source')
            for scene_id, scene in scenes.items():
                for reaction in ('NR', 'R'):
                    report['active_scene_mode'] = dict(scene_id=scene_id, reaction=reaction)
                    report['checks'].append(run_scene(scene_id, scene, reaction, args.steps, args.seed, progress))
                    output.write_text(json.dumps(report, indent=2) + '\n')
            report['status'] = 'PASS_HEADLESS_SNAPSHOT_ONLY'
        except Exception as error:
            report['status'] = 'FAIL_HEADLESS_SNAPSHOT'
            report['errors'].append(repr(error))
            if isinstance(error, SnapshotParityError):
                report['parity_diagnosis'] = error.details
                try:
                    report['failure_artifacts'] = save_parity_failure(error, output)
                except OSError as save_error:
                    report['errors'].append('Saving parity evidence failed: ' + repr(save_error))
            report['traceback'] = traceback.format_exc()
            traceback.print_exc()
        finally:
            report['elapsed_seconds'] = time.monotonic() - start
            output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], report=str(output), errors=report['errors']), indent=2))
    return 0 if report['status'] == 'PASS_HEADLESS_SNAPSHOT_ONLY' else 1


if __name__ == '__main__':
    sys.exit(main())
