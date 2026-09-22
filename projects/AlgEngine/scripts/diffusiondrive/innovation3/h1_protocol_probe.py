"""Two persistent interpreters: real dynamics + diagnostic feedback + V3 updates.

This is NOT rendered DiffusionDrive or PDM training. All learned probe weights
are disposable and cannot be resumed as formal online training checkpoints.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import io
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
from grpo_selector_v3_cached_common import SceneConditionedTrajectorySetSelector
from selector_runtime import environment
from .diagnostic_signal import CONTRACT
from .learner import OnlineV3Learner
from .paths import checked_path, sha256_file
from .protocol import Action, Feedback, StepGate, StepIdentity
from .transport import Channel, digest


def tensor_digest(values):
    h = hashlib.sha256()
    for key, tensor in sorted(values.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def diagnostic_context(observation, device):
    # Analytic bank includes t=0..4 s; V3 requires the eight FUTURE poses.
    world = np.asarray([np.column_stack([a['waypoints'], a['headings']])
                        for a in observation['candidates']], dtype=np.float64)
    if world.shape != (20, 9, 3) or not np.isfinite(world).all():
        raise ValueError('Invalid analytic diagnostic bank')
    ego = observation['states']['ego']
    theta = float(ego['heading'])
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = world[:, 1:].copy()
    local[..., :2] = (local[..., :2] - np.asarray(ego['position'])[:2]) @ rotation
    local[..., 2] = np.arctan2(np.sin(local[..., 2]-theta), np.cos(local[..., 2]-theta))
    local = local.astype(np.float32)
    features = np.zeros((1, 20, 256), dtype=np.float32)
    features[0, :, :24] = local.reshape(20, 24)
    status = np.zeros((1, 1, 256), dtype=np.float32)
    status[0, 0, :2] = ego['velocity']
    status[0, 0, 2] = theta
    agents = np.zeros((1, 30, 256), dtype=np.float32)
    others = [(key, state) for key, state in sorted(observation['states'].items()) if key != 'ego']
    for index, (_, state) in enumerate(others[:30]):
        agents[0, index, :2] = np.asarray(state['position'])[:2] - np.asarray(ego['position'])[:2]
        agents[0, index, 2:4] = state['velocity']
    # These are explicitly stub tokens, never described as real BEV features.
    return {key: torch.from_numpy(value).to(device=device, dtype=torch.float32) for key, value in dict(
        candidate_features=features, candidate_trajectories=local[None],
        route_bev_features=np.zeros((1, 20, 8, 256), dtype=np.float32),
        status_token=status, ego_query=status.copy(), agents_query=agents).items()}


def expect(channel, kind):
    value = channel.receive()
    if value.get('kind') != kind:
        raise RuntimeError('Expected ' + kind + ', got ' + str(value.get('kind')))
    return value


def run(args, report, progress):
    cfg, worker_env = environment(args.settings, '0')
    for key in ('algengine_python', 'simengine_python', 'selector_state', 'scenario_root'):
        checked_path(cfg[key])
    if Path(sys.executable).resolve() != checked_path(cfg['algengine_python']):
        raise RuntimeError('Run the learner in the configured AlgEngine interpreter')
    worker_env.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_CORETYPE='Prescott', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    selector = checked_path(cfg['selector_state'])
    checkpoint_sha = sha256_file(selector)
    if checkpoint_sha != cfg['expected_sha256']['selector_state']:
        raise ValueError('Registered V3 SHA256 mismatch')
    payload = torch.load(selector, map_location='cpu')
    if payload.get('schema_version') != 3 or payload.get('method') != 'scene_conditioned_exact_group_grpo':
        raise ValueError('Expected standard trained V3 checkpoint')
    model = SceneConditionedTrajectorySetSelector(**payload['scene_selector_config']).float().eval()
    model.load_state_dict(payload['scene_selector_state'], strict=True)
    learner = OnlineV3Learner(model, seed=args.seed)
    initial_frozen = tensor_digest(model.state_dict())
    initial_residual = tensor_digest(learner.selector.state_dict())
    report.update(selector_sha256=checkpoint_sha, learner_python=sys.executable,
                  learner_pid=os.getpid(), torch_version=torch.__version__,
                  parameterization='frozen_v3_plus_zero_residual')
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    channel = Channel(left)
    command = [cfg['simengine_python'], '-u', '-m', 'innovation3.h1_worker',
               '--fd', str(right.fileno()), '--settings', str(args.settings),
               '--scene-count', str(args.scene_count), '--steps', str(args.steps),
               '--seed', str(args.seed), '--failure-report', str(args.output)]
    child = None
    with args.output.with_suffix('.worker.log').open('x') as log:
        try:
            child = subprocess.Popen(command, env=worker_env, pass_fds=(right.fileno(),),
                                     stdout=log, stderr=subprocess.STDOUT)
            right.close()
            ready = expect(channel, 'ready')
            report['simengine'] = ready
            progress(dict(event='worker_ready', pid=ready['pid'], scene_count=len(ready['scenes'])))
            if ready['pid'] == os.getpid() or Path(ready['runtime']['python']).resolve() != checked_path(cfg['simengine_python']):
                raise RuntimeError('Interpreter/process isolation failed')
            for scene_id in ready['scenes']:
                for reaction in args.reactions:
                    episode = scene_id + ':' + reaction
                    channel.send(dict(kind='reset', episode=episode, scene_id=scene_id, reaction=reaction))
                    reset = expect(channel, 'reset')
                    if reset['policy_version'] != learner.version:
                        raise RuntimeError('Policy version lost across episode reset')
                    gate = StepGate(policy_version=learner.version, reward_atol=0.)
                    for decision in range(args.steps):
                        started = time.monotonic()
                        channel.send(dict(kind='observe'))
                        observation = expect(channel, 'observation')
                        identity = StepIdentity(**observation['identity'])
                        if identity.episode != episode or identity.decision != decision:
                            raise RuntimeError('Observation episode/step mismatch')
                        gate.observe(identity)
                        if digest(observation['candidates']) != observation['candidate_hash']:
                            raise RuntimeError('Candidate transport hash mismatch')
                        progress(dict(event='observe', episode=episode, decision=decision, version=learner.version))
                        context = diagnostic_context(observation, torch.device('cpu'))
                        context_hash = tensor_digest(context)
                        base = torch.zeros(1, 20, dtype=torch.float32)
                        with torch.no_grad():
                            frozen_logits = base + learner.reference(**context)
                            prior_logits = frozen_logits + learner.selector(**context)
                        selected, probabilities, version = learner.choose(context, base, identity)
                        if version == 0 and not torch.equal(probabilities, frozen_logits.softmax(-1)):
                            raise RuntimeError('Step-zero frozen V3 parity failed')
                        action = Action(identity, observation['candidate_hash'], selected,
                                        tuple(float(x) for x in probabilities[0]))
                        gate.choose(action)
                        # No rewards or branch outcomes have been received at this point.
                        channel.send(dict(kind='action', **asdict(action)))
                        progress(dict(event='action', episode=episode, decision=decision, version=version))
                        main = expect(channel, 'main')
                        if main['identity'] != asdict(identity) or main['candidate_hash'] != action.candidate_hash:
                            raise RuntimeError('Canonical receipt identity mismatch')
                        gate.main_executed(main['next_hash'])
                        progress(dict(event='main', episode=episode, decision=decision, next_hash=main['next_hash']))
                        raw = expect(channel, 'feedback')
                        feedback = Feedback(StepIdentity(**raw['identity']), raw['candidate_hash'],
                                            tuple(raw['rewards']), tuple(raw['valid']), raw['main_reward'],
                                            raw['main_next_hash'], raw['selected_branch_next_hash'])
                        if feedback.main_reward != main['signal']:
                            raise RuntimeError('Independent canonical signal changed')
                        gate.feedback(feedback)
                        progress(dict(event='feedback', episode=episode, decision=decision))
                        # On the second decision verify actual serialized optimizer/RNG restoration.
                        restored = None
                        if len(report['checks']) == 1:
                            # pending action cannot be checkpointed; the boundary was saved below.
                            restored = OnlineV3Learner(model, seed=args.seed)
                            restored.load_state_dict(resume_boundary)
                            replay = restored.choose(context, base, identity)
                            if replay[0] != selected or not torch.equal(replay[1], probabilities):
                                raise RuntimeError('Serialized action/RNG resume mismatch')
                        signals = torch.tensor([feedback.rewards], dtype=torch.float32)
                        update = learner.update(signals, identity)
                        if restored is not None:
                            restored.update(signals, identity)
                            if tensor_digest(restored.selector.state_dict()) != tensor_digest(learner.selector.state_dict()):
                                raise RuntimeError('Serialized optimizer resume mismatch')
                            report['serialized_learner_resume_verified'] = True
                        gate.updated(update['policy_version'], update['optimized'])
                        if len(report['checks']) == 0:
                            buffer = io.BytesIO()
                            torch.save(learner.state_dict(), buffer)
                            buffer.seek(0)
                            resume_boundary = torch.load(buffer, map_location='cpu')
                        with torch.no_grad():
                            next_logits = frozen_logits + learner.selector(**context)
                        correction_change = float((next_logits-prior_logits).abs().max())
                        if tensor_digest(context) != context_hash or any(p.grad is not None for p in learner.reference.parameters()):
                            raise RuntimeError('Frozen inputs/reference received updates')
                        if tensor_digest(learner.reference.state_dict()) != initial_frozen:
                            raise RuntimeError('Frozen V3 weights changed')
                        progress(dict(event='update', episode=episode, decision=decision,
                                      version=learner.version, optimized=update['optimized']))
                        channel.send(dict(kind='updated', identity=asdict(identity),
                                          policy_version=learner.version, optimized=update['optimized']))
                        ack = expect(channel, 'updated')
                        if ack['policy_version'] != learner.version or ack['next_hash'] != main['next_hash']:
                            raise RuntimeError('Update acknowledgement mismatch')
                        record = dict(episode=episode, decision=decision, selected=selected,
                            version_before=version, version_after=learner.version, update=update,
                            before_hash=identity.state_hash, main_hash=main['next_hash'],
                            branch_hashes=raw['branch_hashes'], signals=raw['rewards'],
                            main_signal=main['signal'], selected_signal_parity=True, selected_state_parity=True,
                            candidate_hash=action.candidate_hash, context_hash=context_hash,
                            same_context_logit_change_after_update=correction_change,
                            seconds=time.monotonic()-started)
                        report['checks'].append(record)
                        progress(dict(event='decision_pass', episode=episode, decision=decision,
                                      version=learner.version, seconds=record['seconds']))
            channel.send(dict(kind='close'))
            report['idm_fallbacks'] = expect(channel, 'closed')['idm_fallbacks']
            if child.wait(timeout=30) != 0:
                raise RuntimeError('SimEngine worker exited with error')
        finally:
            left.close()
            right.close()
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    if tensor_digest(model.state_dict()) != initial_frozen:
        raise RuntimeError('Input V3 changed')
    report.update(actual_updates=learner.version, update_attempts=learner.attempts,
                  frozen_v3_unchanged=True, residual_changed=tensor_digest(learner.selector.state_dict()) != initial_residual,
                  next_step_waited_for_update=True, main_signal_evaluated_independently=True,
                  causal_protocol_verified=True)
    if learner.version == 0 or not any(x['same_context_logit_change_after_update'] > 0 for x in report['checks']):
        raise RuntimeError('No actual learning signal observed in this bounded probe')
    if not report.get('serialized_learner_resume_verified'):
        raise RuntimeError('Probe did not verify a second-step learner resume')
    state_path = args.output.with_suffix('.diagnostic.pt')
    with state_path.open('xb') as stream:
        torch.save(dict(kind='DISPOSABLE_PROTOCOL_PROBE_NOT_TRAINING', signal_contract=CONTRACT,
                        learner=learner.state_dict()), stream)
    report['diagnostic_checkpoint'] = dict(path=str(state_path), sha256=sha256_file(state_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene-count', type=int, default=2)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reactions', nargs='+', choices=('NR', 'R'), default=['NR', 'R'])
    args = parser.parse_args()
    if not 1 <= args.scene_count <= 2 or not 2 <= args.steps <= 4 or args.seed < 0:
        parser.error('Bounded protocol probe: scene-count 1..2, steps 2..4, nonnegative seed')
    if len(set(args.reactions)) != len(args.reactions):
        parser.error('Reaction modes must be unique')
    args.settings = checked_path(args.settings.absolute())
    args.output = checked_path(args.output.absolute(), must_exist=False)
    code = Path(__file__).resolve().parents[5]
    if args.output == code or code in args.output.parents:
        parser.error('Private output must be outside code')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if any(args.output.with_suffix(s).exists() for s in ('.json', '.events.jsonl', '.worker.log', '.diagnostic.pt', '.failure')):
        parser.error('Choose a fresh output basename; prior evidence must be preserved')
    report = dict(status='RUNNING', checks=[], errors=[], signal_contract=CONTRACT,
        candidate_source='current_state_analytic_diagnostic', context_source='diagnostic_stub_tokens',
        base_logits_source='diagnostic_zeros', IPC='AF_UNIX_framed_JSON', image_shared_memory_verified=False,
        real_simulation=True, real_generator=False, live_render_verified=False,
        official_pdm_reward=False, online_reward_verified=False, real_closed_loop_verified=False,
        used_for_training=False, formal_ready=False, full_resume_verified=False)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    started = time.monotonic()
    with args.output.with_suffix('.events.jsonl').open('x') as events:
        def progress(event):
            event['monotonic_ns'] = time.monotonic_ns()
            events.write(json.dumps(event, allow_nan=False)+'\n')
            events.flush()
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(json.dumps(event), flush=True)
        try:
            # Include uncommitted/new research sources in the exact run provenance.
            report['code_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()
            paths = list(Path(__file__).parent.glob('*.py')) + list((code/'projects/SimEngine/worldengine/online').glob('*.py'))
            paths += [Path(__file__).parent.parent/'innovation3_runtime.py']
            report['source_sha256'] = {str(p.relative_to(code)): sha256_file(p) for p in sorted(paths)}
            run(args, report, progress)
            report['status'] = 'PASS_H1_PROTOCOL_PROBE_ONLY'
        except Exception as error:
            report.update(status='FAIL_H1_PROTOCOL_PROBE', traceback=traceback.format_exc())
            report['errors'].append(repr(error))
            traceback.print_exc()
        finally:
            report['elapsed_seconds'] = time.monotonic()-started
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], report=str(args.output), errors=report['errors']), indent=2))
    return 0 if report['status'] == 'PASS_H1_PROTOCOL_PROBE_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
