"""Single GPU live renderer -> frozen DiffusionDrive/V3 -> real branch probe.

This stage certifies the observation/candidate bridge only. It deliberately has
no optimizer update and no PDM reward claim. The worker owns SimEngine and the
parent owns the resident CUDA model; the two exchange control JSON plus one
bounded inherited camera buffer.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .candidate_inputs import from_export
from .image_transport import ImageBuffer, metadata_from_wire
from .live_inputs import LiveInputs
from .paths import checked_path, sha256_file
from .transport import Channel, digest
from .visual_model import configuration, load_frozen


def source_hashes():
    code = Path(__file__).resolve().parents[5]
    names = []
    for base in (code / 'projects/AlgEngine/scripts/diffusiondrive/innovation3',
                 code / 'projects/SimEngine/worldengine/online'):
        names.extend(str(p.relative_to(code)) for p in base.glob('*.py'))
    names.append('projects/SimEngine/worldengine/manager/data_manager.py')
    return {name: sha256_file(code / name) for name in sorted(set(names))}


def _sample_context(result, token, selector):
    context, base, error = from_export(result, token, selector)
    source = result['diffusiondrive_rollout_context']
    candidates = np.asarray(source['candidate_trajectories_8'], dtype=np.float32)
    logits = np.asarray(source['current_logits'], dtype=np.float32)
    selected = int(np.asarray(source['selected_indices']).item())
    if candidates.shape != (20, 8, 3) or not np.isfinite(candidates).all():
        raise ValueError('DiffusionDrive candidate shape or finite check failed')
    if logits.shape != (20,) or not np.isfinite(logits).all():
        raise ValueError('DiffusionDrive selector logits shape or finite check failed')
    if selected != int(logits.argmax()) or not 0 <= selected < 20:
        raise ValueError('DiffusionDrive selected index/logit mismatch')
    return candidates, selected, dict(
        context_hash=digest({key: value.detach().cpu().numpy().tolist() for key, value in context.items()}),
        base_logits_sha256=hashlib.sha256(base.detach().cpu().numpy().tobytes()).hexdigest(),
        current_logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
        candidate_hash=digest(candidates.tolist()),
        frozen_v3_export_max_abs_error=error,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 8:
        raise ValueError('Visual probe steps must be in [1, 8]')
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh visual-probe output: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(checked_path(args.settings).read_text())
    report = dict(status='RUNNING', stage='VISUAL_CANDIDATE_BRIDGE_ONLY', output=str(output),
                  code_head=None, steps=args.steps, seed=args.seed, events=[],
                  source_sha256=source_hashes(), real_generator=False,
                  official_pdm_reward=False, online_update=False,
                  real_closed_loop=False, formal_ready=False)
    output.write_text(json.dumps(report, indent=2) + '\n')
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    channel = Channel(left, timeout=1800)
    images = ImageBuffer()
    child = None
    worker_log = output.with_suffix('.worker.log')
    frames, camera_frames = [], []
    started = time.monotonic()
    try:
        config = configuration(cfg, args.seed)
        model = load_frozen(config, cfg)
        selector = model.module.planning_head.scene_selector
        live = LiveInputs(config.data.test)
        worker_env = dict(os.environ)
        worker_env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                          MKL_NUM_THREADS='1', OPENBLAS_CORETYPE='Prescott', PYTHONDONTWRITEBYTECODE='1')
        command = [cfg['simengine_python'], '-u', '-m', 'innovation3.visual_worker',
                   '--fd', str(right.fileno()), '--images-fd', str(images.fd),
                   '--settings', str(args.settings.resolve()), '--output', str(output),
                   '--steps', str(args.steps), '--seed', str(args.seed)]
        with worker_log.open('x') as log:
            child = subprocess.Popen(command, env=worker_env, pass_fds=(right.fileno(), images.fd),
                                     stdout=log, stderr=subprocess.STDOUT)
            right.close()
            ready = channel.receive()
            if ready.get('kind') != 'ready':
                raise RuntimeError('Expected visual worker ready')
            report['worker'] = ready
            report['events'].append(dict(kind='worker_ready', scene=ready['scene']))
            for index in range(args.steps + 3):
                message = channel.receive()
                if message.get('kind') != 'observation':
                    raise RuntimeError('Expected visual worker observation')
                identity = message['identity']
                frame = metadata_from_wire(message['frame'])
                received_images = images.receive(message['images'], identity)
                if frame['token'] != identity['token']:
                    raise RuntimeError('Frame token and step identity differ')
                frames.append(frame)
                camera_frames.append(received_images)
                if len(frames) < 4:
                    channel.send(dict(kind='warmup', identity=identity))
                    report['events'].append(dict(kind='warmup', step=index, history_size=len(frames)))
                    continue
                if len(frames) > 4:
                    frames, camera_frames = frames[-4:], camera_frames[-4:]
                data = live.prepare(frames, camera_frames)
                with torch.no_grad():
                    results = model(return_loss=False, rescale=True, **data)
                if not isinstance(results, list) or len(results) != 1:
                    raise RuntimeError('Unexpected resident model output')
                candidates, selected, details = _sample_context(results[0], identity['token'], selector)
                request = dict(kind='action', identity=identity, candidates=candidates.tolist(),
                               candidate_hash=digest(candidates.tolist()), selected=selected,
                               logits=details['current_logits_sha256'])
                channel.send(request)
                main = channel.receive()
                branches = channel.receive()
                if main.get('kind') != 'main' or branches.get('kind') != 'branches':
                    raise RuntimeError('Visual worker branch response order changed')
                if main['identity'] != identity or branches['identity'] != identity:
                    raise RuntimeError('Visual branch response identity mismatch')
                if not branches['selected_parity'] or not branches['canonical_restored']:
                    raise RuntimeError('Real selected branch parity failed')
                report['events'].append(dict(kind='generated_branch_group', step=index,
                                             candidate_count=20, selected=selected, **details,
                                             main_hash=main['next_hash'],
                                             selected_parity=branches['selected_parity']))
            channel.send(dict(kind='close'))
            closed = channel.receive()
            if closed.get('kind') != 'closed' or closed.get('idm_fallbacks') != 0:
                raise RuntimeError('Visual worker close/fallback check failed')
            report.update(status='PASS_VISUAL_CANDIDATE_BRIDGE_ONLY', elapsed_seconds=time.monotonic()-started,
                          generated_steps=args.steps, warmup_frames=3, history_frames=4,
                          candidate_groups=args.steps, candidate_branches=args.steps*20,
                          worker_log=str(worker_log), code_head=report['worker']['runtime']['code_head'])
    except BaseException as error:
        report.update(status='FAIL_VISUAL_CANDIDATE_BRIDGE', error=repr(error), traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic()-started, worker_log=str(worker_log))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + '\n')
        if child is not None:
            code = child.poll()
            if code is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
        try:
            right.close()
        except OSError:
            pass
        channel.sock.close()
        images.close()


if __name__ == '__main__':
    main()
