"""Bounded visual integration worker; no reward/training or formal evaluation."""
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
        sim = HeadlessSimulator(scene_id, scene, 'R', args.steps+4, args.seed)
        observer = CanonicalObserver(sim, asset_root)
        channel.send(dict(kind='ready', scene=scene_id, pid=os.getpid(), runtime=runtime_evidence(),
                          source=dict(path=str(source), bytes=source.stat().st_size, sha256=sha256_file(source)),
                          asset=dict(path=str(asset), bytes=asset.stat().st_size, sha256=sha256_file(asset))))
        for index in range(args.steps+3):
            frame, payload, state_hash = observer.observe()
            identity = dict(scene=scene_id, step=sim.engine.episode_step, state_hash=state_hash,
                            token=frame['token'])
            descriptor = images.publish(payload, identity)
            channel.send(dict(kind='observation', identity=identity, frame=metadata_to_wire(frame),
                              images=descriptor, history_size=len(observer.history.frames)))
            request = channel.receive()
            if request.get('identity') != identity:
                raise ValueError('Stale observation acknowledgement')
            images.release(identity)  # reader copied and checked all camera bytes
            if index < 3:
                if request['kind'] != 'warmup':
                    raise ValueError('First three transitions are engineering warmup only')
                sim.step(diagnostic_candidates(sim)[12])
                continue
            if request['kind'] != 'action':
                raise ValueError('Expected live generated action')
            if digest(request['candidates']) != request['candidate_hash']:
                raise ValueError('Generated candidate bank digest mismatch')
            bank = np.asarray(request['candidates'], dtype=np.float32)
            actions = to_world_actions(bank, sim.engine.agent_manager.ego_agent)
            selected = request['selected']
            if not isinstance(selected, int):
                raise ValueError('Selected index must be integral')
            if sim.snapshot().state_hash != state_hash:
                raise ValueError('Canonical state changed before generated action')
            def canonical(snapshot, states):
                channel.send(dict(kind='main', identity=identity, next_hash=snapshot.state_hash,
                                  states=states))
            before, main, branches = sim.branch_group(actions, selected, on_main=canonical)
            if failure.count:
                raise RuntimeError('IDM fallback invalidates live branch probe')
            channel.send(dict(kind='branches', identity=identity, main_hash=main.state_hash,
                              branch_hashes=[s.state_hash for s, _ in branches],
                              selected_parity=main.state_hash == branches[selected][0].state_hash,
                              canonical_restored=sim.snapshot().state_hash == main.state_hash))
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
