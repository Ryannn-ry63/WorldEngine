"""Single GPU live renderer -> frozen DiffusionDrive/V3 -> real branch probe.

The default stage certifies the observation/candidate bridge only. With
``--reward-adapter`` it additionally scores generated candidates with the
causal H1 adapter. Neither mode performs optimizer updates or formal PDMS.
The worker owns SimEngine and the parent owns the resident CUDA model; the two
exchange control JSON plus one bounded inherited camera buffer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

import numpy as np
import torch

from .candidate_inputs import from_export
from .image_transport import ImageBuffer, metadata_from_wire
from .live_inputs import LiveInputs
from .live_audit import LiveWindow, validate_receipts
from .paths import checked_path, sha256_file
from .transport import Channel, digest
from .visual_model import configuration, load_frozen
from .visual_parity import assert_same, file_oracle, result_parity


def source_hashes():
    code = Path(__file__).resolve().parents[5]
    names = []
    for base in (code / 'projects/AlgEngine/scripts/diffusiondrive/innovation3',
                 code / 'projects/SimEngine/worldengine/online'):
        names.extend(str(p.relative_to(code)) for p in base.glob('*.py'))
    names.extend(['projects/SimEngine/worldengine/manager/data_manager.py',
                  'projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py'])
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
    parser.add_argument('--reward-adapter', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 8:
        raise ValueError('Visual probe steps must be in [1, 8]')
    output = checked_path(args.output.absolute(), must_exist=False)
    code_root = Path(__file__).resolve().parents[5]
    if output == code_root or code_root in output.parents:
        raise ValueError('Probe output must be outside code')
    if any(output.with_suffix(suffix).exists() for suffix in ('.json', '.worker.log', '.failure')):
        raise FileExistsError('Choose a fresh visual-probe output: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(checked_path(args.settings).read_text())
    report = dict(status='RUNNING',
                  stage=('LIVE_GENERATED_REWARD_ADAPTER_ONLY' if args.reward_adapter
                         else 'VISUAL_CANDIDATE_BRIDGE_ONLY'), output=str(output),
                  code_head=None, steps=args.steps, seed=args.seed, events=[],
                  source_sha256=source_hashes(), real_generator=False,
                  real_generated_reward=False, causal_reward_contract=(
                      'causal_h1_pdm_components_v1' if args.reward_adapter else None),
                  official_pdm_reward=False, online_update=False,
                  real_closed_loop=False, formal_ready=False)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    channel = Channel(left, timeout=1800)
    images = ImageBuffer()
    child = None
    live = None
    worker_log = output.with_suffix('.worker.log')
    window = LiveWindow()
    expected_state = expected_history = None
    started = time.monotonic()
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
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
        if args.reward_adapter:
            command.append('--reward-adapter')
        with worker_log.open('x') as log:
            child = subprocess.Popen(command, env=worker_env, pass_fds=(right.fileno(), images.fd),
                                     stdout=log, stderr=subprocess.STDOUT)
            right.close()
            ready = channel.receive()
            if ready.get('kind') != 'ready':
                raise RuntimeError('Expected visual worker ready')
            report['worker'] = ready
            report['events'].append(dict(kind='worker_ready', scene=ready['scene']))
            warmup_steps = 13 if args.reward_adapter else 3
            if (ready.get('warmup_transitions') != warmup_steps or
                    ready.get('reward_adapter') != args.reward_adapter):
                raise RuntimeError('Worker warmup/reward contract mismatch')
            report['gpu'] = dict(name=torch.cuda.get_device_name(0),
                                 total_bytes=torch.cuda.get_device_properties(0).total_memory)
            for index in range(args.steps + warmup_steps):
                message = channel.receive()
                if message.get('kind') != 'observation':
                    raise RuntimeError('Expected visual worker observation')
                identity = message['identity']
                frame = metadata_from_wire(message['frame'])
                received_images = images.receive(message['images'], identity)
                if identity['step'] != index or identity['scene'] != ready['scene']:
                    raise RuntimeError('Unexpected observation step/scene')
                if expected_state is not None and identity['state_hash'] != expected_state:
                    raise RuntimeError('Next observation is not the executed canonical state')
                window.append(frame, received_images, identity)
                if index < warmup_steps:
                    channel.send(dict(kind='warmup', identity=identity))
                    report['events'].append(dict(kind='warmup', step=index,
                                                 history_size=len(window.frames)))
                    continue
                frames, camera_frames = list(window.frames), list(window.cameras)
                if len(frames) != 4:
                    raise RuntimeError('Generated action requires four consecutive frames')
                data = live.prepare(frames, camera_frames)
                # Only the acceptance oracle uses temporary frame files.
                oracle = file_oracle(config.data.test, frames, camera_frames)
                assert_same(data, oracle)
                with torch.no_grad():
                    results = model(return_loss=False, rescale=True, **data)
                if not isinstance(results, list) or len(results) != 1:
                    raise RuntimeError('Unexpected resident model output')
                with torch.no_grad():
                    oracle_results = model(return_loss=False, rescale=True, **oracle)
                forward_errors = result_parity(results[0], oracle_results[0])
                candidates, selected, details = _sample_context(results[0], identity['token'], selector)
                request = dict(kind='action', identity=identity, candidates=candidates.tolist(),
                               candidate_hash=digest(candidates.tolist()), selected=selected,
                               logits=details['current_logits_sha256'])
                channel.send(request)
                main = channel.receive()
                branches = channel.receive()
                validate_receipts(identity, selected, request['candidate_hash'],
                                  main, branches, args.reward_adapter)
                if (args.reward_adapter and expected_history is not None and
                        branches['history_before_hash'] != expected_history):
                    raise RuntimeError('Reward history did not continue from canonical execution')
                expected_state = main['next_hash']
                expected_history = branches.get('history_after_hash')
                report['events'].append(dict(kind='generated_branch_group', step=index,
                                             candidate_count=20, selected=selected, **details,
                                             file_input_parity=True, forward_errors=forward_errors,
                                             main_hash=main['next_hash'],
                                             selected_parity=branches['selected_parity'],
                                             observation_identity=identity,
                                             input_frame_indices=[f['frame_idx'] for f in frames],
                                             main_receipt=main, branch_feedback=branches,
                                             next_observation_parity=(index > warmup_steps)))
                if args.reward_adapter:
                    values = np.array([r['reward'] for r in branches['rewards']])
                    report['events'][-1].update(reward_std=float(values.std()),
                        reward_min=float(values.min()), reward_max=float(values.max()),
                        has_signal=bool(values.std() > 1e-6))
                    print(json.dumps(dict(decision=index-warmup_steps, selected=selected,
                        reward_std=float(values.std()), main=main['reward'])), flush=True)
                # Persist audited evidence, then allow the next render. The
                # reward mode uses this acknowledgement as the placeholder for
                # the future learner update; the legacy visual bridge keeps its
                # original two-message protocol.
                output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                if args.reward_adapter:
                    channel.send(dict(kind='feedback_ack', identity=identity,
                                      candidate_hash=request['candidate_hash'], next_hash=expected_state))
            channel.send(dict(kind='close'))
            closed = channel.receive()
            if closed.get('kind') != 'closed' or closed.get('idm_fallbacks') != 0:
                raise RuntimeError('Visual worker close/fallback check failed')
            child.wait(timeout=30)
            if child.returncode != 0:
                raise RuntimeError('Visual worker exited unsuccessfully')
            report.update(real_generator=True, live_render_verified=True,
                          file_input_forward_parity=True,
                          real_generated_reward=bool(args.reward_adapter),
                          status=('PASS_LIVE_GENERATED_REWARD_ADAPTER_ONLY' if args.reward_adapter
                                  else 'PASS_VISUAL_CANDIDATE_BRIDGE_ONLY'),
                          elapsed_seconds=time.monotonic()-started,
                          generated_steps=args.steps, warmup_frames=warmup_steps, history_frames=4,
                          candidate_groups=args.steps, candidate_branches=args.steps*20,
                          reward_groups=(args.steps if args.reward_adapter else 0),
                          groups_with_signal=sum(e.get('has_signal', False) for e in report['events']),
                          idm_fallbacks=closed['idm_fallbacks'], reaction='R',
                          next_observation_checks=args.steps-1,
                          worker_log=str(worker_log), code_head=report['worker']['runtime']['code_head'])
    except BaseException as error:
        report.update(status=('FAIL_LIVE_GENERATED_REWARD_ADAPTER' if args.reward_adapter
                              else 'FAIL_VISUAL_CANDIDATE_BRIDGE'), error=repr(error), traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic()-started, worker_log=str(worker_log))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(status=report['status'], report=str(output)), indent=2), flush=True)
        if live is not None:
            live.close()
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
