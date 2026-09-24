"""Spawned, persistent SimEngine workers for CPU branch-parallel experiments.

Enabled only by the explicit branch-workers pilot option. Parent owns canonical execution and
reward history. Every worker restores the same audited dynamic snapshot.
"""
import logging
import multiprocessing as mp
import os
from multiprocessing.connection import wait
import time
import traceback
import numpy as np
from .headless import HeadlessSimulator
from .reward import capture_reward_frame
from .state import SnapshotCodec, structural_hash


class _IDMFailures(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        if 'IDM bug! fall back' in record.getMessage():
            self.count += 1


def _worker_main(conn, scene_id, scene, reaction, max_steps, seed, bundle, offset, actor_types, global_config):
    # Branch workers are CPU-only; do not let an inherited rank GPU become visible.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sim = None
    failures = _IDMFailures()
    logging.getLogger().addHandler(failures)
    try:
        sim = HeadlessSimulator(scene_id, scene, reaction, max_steps, seed, global_config=global_config)
        sim.codec = SnapshotCodec(bundle=bundle)
        conn.send({'kind': 'ready', 'config_hash': structural_hash(sim.engine.global_config),
                   'pid': os.getpid(), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']})
        while True:
            request = conn.recv()
            if request is None:
                break
            started = time.monotonic()
            count = failures.count
            try:
                sim.codec.audit_static()
                sim.restore(request['before'])
                states = sim.step(request['action'])
                snapshot = sim.snapshot()
                frame = capture_reward_frame(sim.engine, offset, actor_types)
                if sim.snapshot().state_hash != snapshot.state_hash:
                    raise RuntimeError('Reward capture mutated branch dynamics')
                sim.codec.audit_static()
                if failures.count != count:
                    raise RuntimeError('Branch encountered IDM fallback')
                conn.send({'kind': 'result', 'index': request['index'],
                           'group': request['group'], 'before_hash': request['before'].state_hash,
                           'snapshot': snapshot, 'states': states, 'frame': frame,
                           'idm_fallbacks': 0, 'elapsed_seconds': time.monotonic()-started})
            except BaseException as error:
                conn.send({'kind': 'error', 'index': request.get('index'),
                           'error': repr(error), 'traceback': traceback.format_exc()})
                break
    except BaseException as error:
        try:
            conn.send({'kind': 'startup_error', 'error': repr(error), 'traceback': traceback.format_exc()})
        except BaseException:
            pass
    finally:
        if sim is not None:
            sim.close()
        logging.getLogger().removeHandler(failures)
        conn.close()


class ParallelBranchPool:
    """Results are returned in candidate order; any failure invalidates the pool."""
    def __init__(self, scene_id, scene, reaction='R', max_steps=8, seed=0,
                 static_bundle=None, offset=None, actor_types=None, workers=2, timeout=180.,
                 global_config=None):
        if type(workers) is not int or not 1 <= workers <= 20:
            raise ValueError('workers must be 1..20')
        if static_bundle is None or offset is None or actor_types is None:
            raise ValueError('static bundle, offset and actor types are required')
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        self.timeout = float(timeout)
        self._connections, self._processes = [], []
        self._group = 0
        self._closed = False
        self.worker_receipts = []
        expected_config_hash = structural_hash(global_config) if global_config is not None else None
        startup_started = time.monotonic()
        ctx = mp.get_context('spawn')
        try:
            for worker in range(workers):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_worker_main,
                    args=(child, scene_id, scene, reaction, max_steps, seed,
                          static_bundle, np.asarray(offset, dtype=float), actor_types, global_config),
                    name='worldengine-branch-%02d' % worker)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self._connections.append(parent)
                self._processes.append(process)
            for conn in self._connections:
                if not conn.poll(self.timeout):
                    raise TimeoutError('Branch worker startup timed out')
                message = conn.recv()
                if message.get('kind') != 'ready':
                    raise RuntimeError('Branch worker startup failed: ' + repr(message))
                if expected_config_hash is not None and message['config_hash'] != expected_config_hash:
                    raise RuntimeError('Branch worker startup config hash differs from parent')
                self.worker_receipts.append(message)
            self.startup_seconds = time.monotonic() - startup_started
        except BaseException:
            self.close(abort=True)
            raise

    def run(self, before, actions):
        if self._closed:
            raise RuntimeError('Branch pool is closed')
        if len(actions) != 20:
            raise ValueError('Exact H1 branch pool requires 20 actions')
        self._group += 1
        pending = iter(enumerate(actions))
        active, results = {}, {}

        def dispatch(worker):
            try:
                index, action = next(pending)
            except StopIteration:
                active.pop(worker, None)
                return
            self._connections[worker].send(dict(index=index, group=self._group, before=before, action=action))
            active[worker] = index

        try:
            for worker in range(len(self._connections)):
                dispatch(worker)
            while active:
                ready = wait([self._connections[w] for w in active], timeout=self.timeout)
                if not ready:
                    raise TimeoutError('Branch group timed out waiting for workers')
                for conn in ready:
                    worker = self._connections.index(conn)
                    message = conn.recv()
                    if message.get('kind') != 'result':
                        raise RuntimeError('Branch worker failed: ' + repr(message))
                    index = message['index']
                    if (index != active[worker] or index in results or
                            message['group'] != self._group or message['before_hash'] != before.state_hash):
                        raise RuntimeError('Stale or mismatched branch identity')
                    results[index] = message
                    dispatch(worker)
            if sorted(results) != list(range(20)):
                raise RuntimeError('Missing branch results')
            return [results[i] for i in range(20)]
        except BaseException:
            self.close(abort=True)
            raise

    def close(self, abort=False):
        if self._closed:
            return
        self._closed = True
        if not abort:
            for conn in self._connections:
                try:
                    conn.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
        # A common deadline avoids workers * timeout shutdown delays.
        deadline = time.monotonic() + (0. if abort else 5.)
        for process in self._processes:
            process.join(timeout=max(0., deadline-time.monotonic()))
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=2.)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.)
        for conn in self._connections:
            conn.close()
        self._connections, self._processes = [], []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def parallel_reward_group(session, pool, actions, selected, on_main=None):
    """Experimental canonical-first H1; identical parent-side scorer and weights.

    The optional callback is called immediately after the canonical main action,
    matching the serial RewardSession.group protocol while workers evaluate the
    detached branch simulations.
    """
    from .reward import RewardHistory, H1Reward, ego_state
    from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
    if len(actions) != 20 or type(selected) is not int or not 0 <= selected < 20:
        raise ValueError('Expected full20 and integral selected index')
    sim = session.sim
    history_before = session.history.snapshot()
    history_hash = session.history.state_hash
    start = history_before[-1]
    radius = 100 + max(np.linalg.norm(f.ego[:2]-start.ego[:2]) for f in history_before)
    drivable = PDMDrivableMap.from_simulation(session.map_api, ego_state(start), radius)
    adapter = H1Reward(session.centerline, session.route_lanes, drivable, session.map_api)
    sim.codec.audit_static()
    before = sim.snapshot()
    main_states = sim.step(actions[selected])
    main = sim.snapshot()
    main_frame = session.capture()
    try:
        main_reward = adapter.score(history_before, main_frame)
        if on_main is not None:
            on_main(main, main_states, main_reward)
        t0 = time.monotonic()
        branches = pool.run(before, actions)
        t1 = time.monotonic()
        rewards, hashes = [], []
        for branch in branches:
            rewards.append(adapter.score(history_before, branch['frame']))
            history = RewardHistory()
            history.restore(history_before)
            history.append(branch['frame'])
            hashes.append(history.state_hash)
        if branches[selected]['snapshot'].state_hash != main.state_hash:
            raise sim.codec.mismatch('Parallel selected branch differs from canonical',
                                     before, main, branches[selected]['snapshot'], actions[selected])
        if rewards[selected] != main_reward:
            raise RuntimeError('Parallel selected reward differs from canonical')
        if session.history.state_hash != history_hash:
            raise RuntimeError('Parallel scoring mutated canonical reward history')
        canonical_history = RewardHistory()
        canonical_history.restore(history_before)
        canonical_history.append(main_frame)
        if hashes[selected] != canonical_history.state_hash:
            raise RuntimeError('Parallel selected reward history differs from canonical')
        t2 = time.monotonic()
    finally:
        sim.restore(main)
        sim.codec.audit_static()
    session.history.restore(canonical_history.snapshot())
    return dict(main=main_reward, branches=rewards,
        branch_hashes=[b['snapshot'].state_hash for b in branches],
        before_hash=before.state_hash, main_hash=main.state_hash,
        selected_state_parity=True, selected_reward_parity=True, selected_history_parity=True,
        history_before_hash=history_hash, history_after_hash=session.history.state_hash,
        reward_std=float(np.std([r['reward'] for r in rewards])),
        component_ranges={k:float(np.ptp([r[k] for r in rewards])) for k in rewards[0]},
        frame_hashes=[b['frame'].state_hash for b in branches],
        profile=dict(pool_seconds=t1-t0,parent_score_seconds=t2-t1,
                     worker_seconds=[b['elapsed_seconds'] for b in branches]))
