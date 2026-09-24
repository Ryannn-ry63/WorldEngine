"""Bounded visual integration worker with an optional causal H1 reward adapter."""
import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import socket
import traceback
import numpy as np
from .paths import checked_path, sha256_file
from .visual_assets import visual_asset
from .transport import Channel, digest
from .live_audit import LiveStepGate
from .image_transport import ImageBuffer, metadata_to_wire
from .snapshot_probe import IDMFailures, runtime_evidence, save_parity_failure
from worldengine.online.headless import HeadlessSimulator, diagnostic_candidates
from worldengine.online.observations import CanonicalObserver
from worldengine.online.actions import to_world_actions
from worldengine.online.state import SnapshotParityError
from worldengine.online.parallel_branches import ParallelBranchPool, parallel_reward_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--images-fd', type=int, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reward-adapter', action='store_true')
    parser.add_argument('--online-updates', action='store_true')
    parser.add_argument('--policy-version', type=int, default=0,
                        help='policy version carried into this episode reset')
    parser.add_argument('--branch-workers', type=int, default=0,
                        help='persistent CPU SimEngine workers for experimental full20 H1 branches')
    args = parser.parse_args()
    if args.online_updates and not args.reward_adapter:
        parser.error('Online updates require causal reward')
    if not 0 <= args.branch_workers <= 20:
        parser.error('--branch-workers must be in [0, 20]')
    if args.branch_workers and not args.reward_adapter:
        parser.error('Parallel branches require causal reward')
    if args.policy_version < 0:
        parser.error('--policy-version must be nonnegative')
    online = LiveStepGate(policy_version=args.policy_version) if args.online_updates else None
    channel = Channel(socket.socket(fileno=args.fd), timeout=900)
    images = ImageBuffer(args.images_fd)
    sim = None
    branch_pool = None
    try:
        cfg = json.loads(checked_path(args.settings).read_text())
        combined_source = checked_path(
            Path(cfg['scenario_root'])/'original/navtrain_failures_per1/all_scenarios.pkl')
        # Distinct-scene DDP assignments already carry an audited single-scene
        # pickle. Loading it avoids reading the ~2.6 GB combined source in every
        # rank while preserving the exact scene/asset contract. The combined
        # path remains the compatibility fallback for older probes.
        source = checked_path(cfg['visual_scene_path']) if cfg.get('visual_scene_path') else combined_source
        with source.open('rb') as stream:
            loaded = pickle.load(stream)
        if cfg.get('visual_scene_path'):
            scene_id = cfg.get('visual_scene_id')
            if not scene_id:
                raise ValueError('visual_scene_path requires visual_scene_id')
            scene = loaded.get(scene_id) if isinstance(loaded, dict) else loaded
            if not isinstance(scene, dict):
                raise ValueError('Single-scene pickle did not contain a scene mapping')
            scenes = {scene_id: scene}
        else:
            scenes = loaded
            scene_id, _, _ = visual_asset(cfg, scenes)
            scene = scenes[scene_id]
        scene_id, asset_root, asset = visual_asset(cfg, scenes)
        del scenes, loaded
        failure = IDMFailures(); logging.getLogger().addHandler(failure)
        warmup_steps = 13 if args.reward_adapter else 3
        sim = HeadlessSimulator(scene_id, scene, 'R', args.steps+warmup_steps+1, args.seed)
        observer = CanonicalObserver(sim, asset_root)
        reward_session = None
        reward_contract = None
        map_evidence = []
        if args.reward_adapter:
            from worldengine.online.reward_runtime import RewardSession
            from worldengine.online.reward import CONTRACT
            root = checked_path(cfg['map_root'])
            metadata = checked_path(root/'nuplan-maps-v1.0.json')
            version = json.loads(metadata.read_text())[scene['map']]['version']
            mapfile = checked_path(root/scene['map']/version/'map.gpkg')
            map_evidence = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha256_file(p))
                            for p in (metadata, mapfile)]
            reward_session = RewardSession(sim, root)
            reward_contract = CONTRACT
        if args.branch_workers:
            if reward_session is None:
                raise RuntimeError('Parallel branch workers require initialized causal reward session')
            branch_pool = ParallelBranchPool(
                scene_id, scene, 'R', args.steps + warmup_steps + 1, args.seed,
                sim.codec.bundle, reward_session.offset, reward_session.actor_types,
                workers=args.branch_workers, timeout=900., global_config=sim.engine.global_config)
        channel.send(dict(kind='ready', scene=scene_id, pid=os.getpid(), runtime=runtime_evidence(),
                          reward_adapter=bool(args.reward_adapter),
                          online_updates=bool(args.online_updates),
                          reward_contract=reward_contract, map=map_evidence, reaction='R',
                          dynamics_contract='online_signed_accel_steering_rate_substeps_v3',
                          branch_execution=('parallel_persistent_full20_workers_%d' % args.branch_workers
                                            if args.branch_workers else 'serial_full20'),
                          branch_workers=args.branch_workers,
                          branch_pool_startup_seconds=(branch_pool.startup_seconds if branch_pool else 0.),
                          branch_worker_receipts=(branch_pool.worker_receipts if branch_pool else []),
                          dynamics_config_hash=sim.snapshot().config_hash,
                          warmup_contract=('map_live_pose_tangent_v2' if args.reward_adapter else 'diagnostic_v1'),
                          warmup_transitions=warmup_steps,
                          source=dict(path=str(source), bytes=source.stat().st_size, sha256=sha256_file(source)),
                          combined_scenario_source=dict(
                              path=str(combined_source),
                              bytes=combined_source.stat().st_size,
                              sha256=(sha256_file(combined_source)
                                      if source == combined_source else None)),
                          asset=dict(path=str(asset), bytes=asset.stat().st_size, sha256=sha256_file(asset))))
        for index in range(args.steps+warmup_steps):
            try:
                frame, payload, state_hash = observer.observe()
            except ValueError as error:
                if 'History requires consecutive frames' in str(error):
                    last = observer.history.raw[-1] if observer.history.raw else None
                    raise ValueError(
                        'History requires consecutive frames in one episode: ' +
                        repr(dict(index=index, episode_step=int(sim.engine.episode_step),
                                  previous_frame_idx=(last.get('frame_idx') if last else None),
                                  previous_timestamp=(last.get('timestamp') if last else None)))) from error
                raise
            identity = dict(scene=scene_id, step=sim.engine.episode_step, state_hash=state_hash,
                            token=frame['token'])
            if online is not None:
                identity['policy_version'] = online.gate.policy_version
                if index >= warmup_steps:
                    online.observe(identity)
            descriptor = images.publish(payload, identity)
            channel.send(dict(kind='observation', identity=identity, frame=metadata_to_wire(frame),
                              images=descriptor, history_size=len(observer.history.frames)))
            request = channel.receive()
            if request.get('identity') != identity:
                raise ValueError('Stale observation acknowledgement')
            images.release(identity)  # reader copied and checked all camera bytes
            if index < warmup_steps:
                if request['kind'] != 'warmup':
                    raise ValueError('Warmup transitions must precede generated actions')
                if reward_session is None:
                    warmup_action = diagnostic_candidates(sim)[12]
                    sim.step(warmup_action)
                else:
                    reward_session.warmup(reward_session.warmup_action())
                continue
            if request['kind'] != 'action':
                raise ValueError('Expected live generated action')
            if digest(request['candidates']) != request['candidate_hash']:
                raise ValueError('Generated candidate bank digest mismatch')
            if online is not None:
                online.choose(request)
            bank = np.asarray(request['candidates'], dtype=np.float32)
            actions = to_world_actions(bank, sim.engine.agent_manager.ego_agent)
            selected = request['selected']
            if type(selected) is not int or not 0 <= selected < 20:
                raise ValueError('Selected index must be integral')
            if sim.snapshot().state_hash != state_hash:
                raise ValueError('Canonical state changed before generated action')
            if reward_session is None:
                def canonical(snapshot, states):
                    channel.send(dict(kind='main', identity=identity, next_hash=snapshot.state_hash,
                                      candidate_hash=request['candidate_hash'], selected=selected,
                                      states=states))
                before, main, branches = sim.branch_group(actions, selected, on_main=canonical)
                reward_result = None
            else:
                def canonical(snapshot, states, reward):
                    receipt = dict(kind='main', identity=identity, next_hash=snapshot.state_hash,
                                      candidate_hash=request['candidate_hash'], selected=selected,
                                      states=states, reward=reward)
                    if online is not None:
                        online.main_executed(receipt)
                    channel.send(receipt)
                if branch_pool is None:
                    reward_result = reward_session.group(actions, selected, on_main=canonical)
                else:
                    reward_result = parallel_reward_group(
                        reward_session, branch_pool, actions, selected, on_main=canonical)
            if failure.count:
                raise RuntimeError('IDM fallback invalidates live branch probe')
            if reward_result is None:
                main_hash = main.state_hash
                branch_hashes = [s.state_hash for s, _ in branches]
                selected_parity = main_hash == branches[selected][0].state_hash
                canonical_restored = sim.snapshot().state_hash == main_hash
                payload = dict(identity=identity, main_hash=main_hash,
                               branch_hashes=branch_hashes, selected_parity=selected_parity,
                               canonical_restored=canonical_restored)
            else:
                main_hash = reward_result['main_hash']
                branch_hashes = reward_result['branch_hashes']
                payload = dict(identity=identity, main_hash=main_hash,
                               branch_hashes=branch_hashes,
                               selected_parity=reward_result['selected_state_parity'],
                               canonical_restored=sim.snapshot().state_hash == main_hash,
                               reward=reward_result['main'], rewards=reward_result['branches'],
                               reward_std=reward_result['reward_std'],
                               component_ranges=reward_result['component_ranges'],
                               selected_reward_parity=reward_result['selected_reward_parity'],
                               selected_history_parity=reward_result['selected_history_parity'],
                               history_before_hash=reward_result['history_before_hash'],
                               history_after_hash=reward_result['history_after_hash'])
                if 'profile' in reward_result:
                    payload['branch_profile'] = reward_result['profile']
            payload.update(kind='branches', selected=selected, candidate_hash=request['candidate_hash'],
                           branch_execution=('parallel_persistent_full20_workers_%d' % args.branch_workers
                                              if args.branch_workers else 'serial_full20'))
            if online is not None:
                online.feedback(payload)
            channel.send(payload)
            if online is not None:
                channel.send(online.updated(channel.receive()))
            elif args.reward_adapter:
                if channel.receive() != dict(kind='feedback_ack', identity=identity,
                                             candidate_hash=request['candidate_hash'], next_hash=main_hash):
                    raise ValueError('Expected audited feedback acknowledgement before next render')
        if channel.receive() != {'kind': 'close'}:
            raise ValueError('Expected explicit close')
        if branch_pool is not None:
            branch_pool.close()
            branch_pool = None
        if online is not None and online.gate.phase != 'ready':
            raise RuntimeError('Cannot close with an unacknowledged update')
        channel.send(dict(kind='closed', idm_fallbacks=failure.count))
    except BaseException as error:
        if isinstance(error, SnapshotParityError):
            save_parity_failure(error, args.output)
        try:
            channel.send(dict(kind='error', error=repr(error), traceback=traceback.format_exc()))
        except BaseException:
            pass
        raise
    finally:
        if branch_pool is not None:
            branch_pool.close(abort=True)
        if 'observer' in locals() and observer is not None:
            observer.close()
        if sim is not None:
            sim.close()
        images.close()
        channel.sock.close()


if __name__ == '__main__':
    main()
