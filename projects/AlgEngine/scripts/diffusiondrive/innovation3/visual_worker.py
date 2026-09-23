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
from .image_transport import ImageBuffer, metadata_to_wire
from .snapshot_probe import IDMFailures, runtime_evidence, save_parity_failure
from worldengine.online.headless import HeadlessSimulator, diagnostic_candidates
from worldengine.online.observations import CanonicalObserver
from worldengine.online.actions import to_world_actions
from worldengine.online.state import SnapshotParityError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--images-fd', type=int, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reward-adapter', action='store_true')
    args = parser.parse_args()
    channel = Channel(socket.socket(fileno=args.fd), timeout=900)
    images = ImageBuffer(args.images_fd)
    sim = None
    try:
        cfg = json.loads(checked_path(args.settings).read_text())
        source = checked_path(Path(cfg['scenario_root'])/'original/navtrain_failures_per1/all_scenarios.pkl')
        with source.open('rb') as stream:
            scenes = pickle.load(stream)
        scene_id, asset_root, asset = visual_asset(cfg, scenes)
        scene = scenes[scene_id]
        del scenes
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
        channel.send(dict(kind='ready', scene=scene_id, pid=os.getpid(), runtime=runtime_evidence(),
                          reward_adapter=bool(args.reward_adapter),
                          reward_contract=reward_contract, map=map_evidence, reaction='R',
                          warmup_transitions=warmup_steps,
                          source=dict(path=str(source), bytes=source.stat().st_size, sha256=sha256_file(source)),
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
                warmup_action = diagnostic_candidates(sim)[12]
                if reward_session is None:
                    sim.step(warmup_action)
                else:
                    reward_session.warmup(warmup_action)
                continue
            if request['kind'] != 'action':
                raise ValueError('Expected live generated action')
            if digest(request['candidates']) != request['candidate_hash']:
                raise ValueError('Generated candidate bank digest mismatch')
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
                    channel.send(dict(kind='main', identity=identity, next_hash=snapshot.state_hash,
                                      candidate_hash=request['candidate_hash'], selected=selected,
                                      states=states, reward=reward))
                reward_result = reward_session.group(actions, selected, on_main=canonical)
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
            payload.update(kind='branches', selected=selected, candidate_hash=request['candidate_hash'])
            channel.send(payload)
            if args.reward_adapter:
                if channel.receive() != dict(kind='feedback_ack', identity=identity,
                                             candidate_hash=request['candidate_hash'], next_hash=main_hash):
                    raise ValueError('Expected audited feedback acknowledgement before next render')
        if channel.receive() != {'kind': 'close'}:
            raise ValueError('Expected explicit close')
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
        if sim is not None:
            sim.close()
        images.close()
        channel.sock.close()


if __name__ == '__main__':
    main()
