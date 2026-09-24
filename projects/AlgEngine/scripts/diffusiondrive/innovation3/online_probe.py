"""Bounded single-GPU live full20/H1 learning acceptance, never formal training."""
import argparse
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
from .learner import OnlineV3Learner
from .live_audit import LiveWindow
from .live_learning import LiveLearning, learning_evidence, state_digest
from .paths import checked_path, sha256_file
from .transport import Channel


def source_hashes():
    code = Path(__file__).resolve().parents[5]
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((code/'projects/SimEngine/worldengine/online').glob('*.py'))
    paths += [code/name for name in (
        'projects/SimEngine/worldengine/manager/data_manager.py',
        'projects/SimEngine/worldengine/components/agents/controller/tracker/lqr_tracker.py',
        'projects/SimEngine/worldengine/components/agents/controller/two_stage_controller.py',
        'projects/SimEngine/worldengine/components/agents/controller/motion_model/kinematic_bicycle.py',
        'projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py',
        'projects/AlgEngine/scripts/diffusiondrive/selector_runtime.py',
        'projects/AlgEngine/scripts/diffusiondrive/grpo_selector_v3_cached_common.py',
        'projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py')]
    return {str(p.relative_to(code)): sha256_file(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--throughput', action='store_true',
                        help='measure production-like path without file oracle or duplicate forward')
    parser.add_argument('--branch-workers', type=int, default=0,
                        help='experimental persistent CPU SimEngine branch workers (0 keeps serial H1)')
    args = parser.parse_args()
    if not 0 <= args.branch_workers <= 20:
        parser.error('--branch-workers must be in [0, 20]')
    if not 1 <= args.steps <= 8 or args.seed < 0:
        parser.error('Use steps 1..8 and a nonnegative seed')
    output = checked_path(args.output.absolute(), must_exist=False)
    code = Path(__file__).resolve().parents[5]
    if code == output or code in output.parents:
        parser.error('Probe output must be outside code')
    if any(output.with_suffix(s).exists() for s in ('.json', '.worker.log', '.online.pt', '.failure')):
        raise FileExistsError('Use a fresh output basename: ' + str(output))
    cfg = json.loads(checked_path(args.settings).read_text())
    candidate_seed = cfg.get('online_candidate_seed', args.seed)
    scene_seed = cfg.get('online_scene_seed', args.seed)
    action_seed = cfg.get('online_action_seed', args.seed)
    if any(type(x) is not int or not 0 <= x < 2**31 for x in (candidate_seed, scene_seed, action_seed)):
        raise ValueError('Invalid online seed namespace')
    report = dict(status='RUNNING', stage=('STRICT_ONLINE_THROUGHPUT' if args.throughput
                                           else 'STRICT_ONLINE_SELECTOR_ONLY'),
        steps=args.steps, seed=args.seed, seed_namespaces=dict(training_seed=args.seed,
            candidate_seed=candidate_seed, scene_seed=scene_seed, action_seed=action_seed), events=[], source_sha256=source_hashes(),
        code_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip(),
        real_generator=False, live_render_verified=False, real_generated_reward=False,
        causal_reward_contract='causal_h1_pdm_components_v1',
        official_pdm_reward=False, online_update=False, real_closed_loop=False,
        formal_ready=False, full_resume_verified=False, ddp_verified=False,
        parameterization='frozen_v3_plus_zero_residual', used_for_formal_training=False,
        selection='categorical_T1_private_learner_rng',
        branch_execution=(('parallel_persistent_full20_workers_%d' % args.branch_workers)
                           if args.branch_workers else
                           ('serial_full20_production_like' if args.throughput
                            else 'serial_full20_correctness_reference')),
        branch_workers=args.branch_workers, throughput_mode=bool(args.throughput), profile_events=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    child = live = images = channel = left = right = None
    started = time.monotonic()
    worker_log = output.with_suffix('.worker.log')
    try:
        # Import heavy visual dependencies only inside the recorded failure boundary.
        from .live_inputs import LiveInputs
        from .visual_model import configuration, load_frozen
        from .visual_parity import assert_same, file_oracle, result_parity
        torch.manual_seed(candidate_seed)
        np.random.seed(candidate_seed)
        config = configuration(cfg, candidate_seed)
        model = load_frozen(config, cfg)
        selector = model.module.planning_head.scene_selector
        initial_model_hash = state_digest(model.state_dict())
        learner = OnlineV3Learner(selector, seed=action_seed)
        online = LiveLearning(learner)
        initial_residual_hash = online.last_residual_hash
        live = LiveInputs(config.data.test)
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        channel = Channel(left, timeout=1800)
        images = ImageBuffer()
        worker_env = dict(os.environ)
        worker_env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                          MKL_NUM_THREADS='1', OPENBLAS_CORETYPE='Prescott', PYTHONDONTWRITEBYTECODE='1')
        # A real DDP parent keeps all explicitly requested devices visible so
        # NCCL can address rank-local indices.  The renderer is a separate
        # process, so pin only that child to its rank's physical GPU.
        if worker_gpu := worker_env.get('WORLDENGINE_DDP_LOCAL_GPU'):
            worker_env['CUDA_VISIBLE_DEVICES'] = worker_gpu
        command = [cfg['simengine_python'], '-u', '-m', 'innovation3.visual_worker',
            '--fd', str(right.fileno()), '--images-fd', str(images.fd),
            '--settings', str(args.settings.resolve()), '--output', str(output),
            '--steps', str(args.steps), '--seed', str(scene_seed),
            '--reward-adapter', '--online-updates']
        if args.branch_workers:
            command += ['--branch-workers', str(args.branch_workers)]
        with worker_log.open('x') as log:
            child = subprocess.Popen(command, env=worker_env, pass_fds=(right.fileno(), images.fd),
                                     stdout=log, stderr=subprocess.STDOUT)
            right.close()
            ready = channel.receive()
            if (ready.get('kind') != 'ready' or not ready.get('reward_adapter') or
                    not ready.get('online_updates') or ready.get('warmup_transitions') != 13 or
                    ready.get('reward_contract', {}).get('name') != report['causal_reward_contract']):
                raise RuntimeError('Worker online/reward/warmup contract mismatch')
            if cfg.get('visual_scene_id') and ready.get('scene') != cfg['visual_scene_id']:
                raise RuntimeError('Worker did not use the explicitly assigned scene')
            report['worker'] = ready
            report['gpu'] = dict(name=torch.cuda.get_device_name(0),
                                 total_bytes=torch.cuda.get_device_properties(0).total_memory)
            window = LiveWindow()
            previous_update_time = None
            for index in range(13 + args.steps):
                step_started_ns = time.monotonic_ns()
                message = channel.receive()
                if message.get('kind') != 'observation':
                    raise RuntimeError('Expected live observation')
                identity = message['identity']
                if identity['step'] != index or identity['scene'] != ready['scene']:
                    raise RuntimeError('Unexpected observation scene/step')
                frame = metadata_from_wire(message['frame'])
                window.append(frame, images.receive(message['images'], identity), identity)
                # The reader owns the shared camera buffer only until the
                # current observation has been copied into the resident
                # four-frame window. Release it before the next render.
                images.release(identity)
                observation_copied_ns = time.monotonic_ns()
                if index < 13:
                    if identity.get('policy_version') != 0:
                        raise RuntimeError('Warmup changed policy version')
                    channel.send(dict(kind='warmup', identity=identity))
                    report['events'].append(dict(kind='warmup', step=index, history_size=len(window.frames),
                                                 elapsed_seconds=(time.monotonic_ns()-step_started_ns)/1e9))
                    continue
                online.observe(identity)
                if previous_update_time is not None and online.times['observe'] <= previous_update_time:
                    raise RuntimeError('Observation did not follow completed update')
                data = live.prepare(list(window.frames), list(window.cameras))
                prepared_ns = time.monotonic_ns()
                if args.throughput:
                    # Correctness mode keeps a file oracle and duplicate
                    # forward. Throughput mode measures the resident path.
                    with torch.no_grad():
                        results = model(return_loss=False, rescale=True, **data)
                    oracle_results = None
                else:
                    oracle = file_oracle(config.data.test, list(window.frames), list(window.cameras))
                    assert_same(data, oracle)
                    with torch.no_grad():
                        results = model(return_loss=False, rescale=True, **data)
                        oracle_results = model(return_loss=False, rescale=True, **oracle)
                forwarded_ns = time.monotonic_ns()
                if not isinstance(results, list) or len(results) != 1:
                    raise RuntimeError('Unexpected resident model output')
                forward_errors = ({'mode': 'throughput_no_duplicate_oracle'} if args.throughput
                                  else result_parity(results[0], oracle_results[0]))
                context, base, export_error = from_export(results[0], identity['token'], selector)
                candidates = context['candidate_trajectories'][0].cpu().tolist()
                request = online.choose(context, base, candidates)
                channel.send(request)
                action_sent_ns = time.monotonic_ns()
                main_receipt = channel.receive()
                main_received_ns = time.monotonic_ns()
                online.main_executed(main_receipt)
                branches = channel.receive()
                feedback_received_ns = time.monotonic_ns()
                ack, audit = online.update(branches)
                update_complete_ns = time.monotonic_ns()
                if any(p.requires_grad or p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Frozen generator or input V3 received gradients')
                record = dict(kind='online_update', step=index-13, **audit,
                    selected=request['selected'], candidate_hash=request['candidate_hash'],
                    observation_identity=identity, input_frame_indices=[f['frame_idx'] for f in window.frames],
                    file_input_parity=True, forward_errors=forward_errors,
                    frozen_v3_export_max_abs_error=export_error,
                    main_receipt=main_receipt, branch_feedback=branches,
                    next_observation_parity=(index > 13))
                report['events'].append(record)
                if not args.throughput:
                    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                record['times_ns']['ack_sent'] = time.monotonic_ns()
                channel.send(ack)
                response = channel.receive()
                online.acknowledged(response)
                record['worker_update_receipt'] = response
                worker_ack_received_ns = time.monotonic_ns()
                record['times_ns']['worker_ack_received'] = worker_ack_received_ns
                previous_update_time = record['times_ns']['update_complete']
                record['profile'] = dict(
                    decision_wall_seconds=(worker_ack_received_ns-step_started_ns)/1e9,
                    observation_copy_seconds=(observation_copied_ns-step_started_ns)/1e9,
                    input_prepare_seconds=(prepared_ns-observation_copied_ns)/1e9,
                    forward_seconds=(forwarded_ns-prepared_ns)/1e9,
                    action_to_main_seconds=(main_received_ns-action_sent_ns)/1e9,
                    branch_feedback_seconds=(feedback_received_ns-main_received_ns)/1e9,
                    learner_update_seconds=(update_complete_ns-feedback_received_ns)/1e9,
                    update_ack_seconds=(worker_ack_received_ns-update_complete_ns)/1e9)
                report['profile_events'].append(record['profile'])
                print(json.dumps(dict(decision=index-13, selected=request['selected'],
                    policy_version=learner.version, optimized=audit['optimized'],
                    reward_std=audit['update']['reward_std'],
                    logit_change=audit['same_context_logit_change_after_update'])), flush=True)
            channel.send(dict(kind='close'))
            closed = channel.receive()
            if closed.get('kind') != 'closed' or closed.get('idm_fallbacks') != 0:
                raise RuntimeError('Worker close/fallback check failed')
            if child.wait(timeout=30) != 0:
                raise RuntimeError('Worker exited unsuccessfully')
        if state_digest(model.state_dict()) != initial_model_hash:
            raise RuntimeError('Frozen generator/V3 state changed')
        if source_hashes() != report['source_sha256']:
            raise RuntimeError('Source changed during acceptance run')
        evidence = learning_evidence(report['events'])
        if evidence['actual_optimizer_steps'] != learner.version or learner.attempts != args.steps:
            raise RuntimeError('Optimizer/attempt accounting mismatch')
        checkpoint = output.with_suffix('.online.pt')
        with checkpoint.open('xb') as stream:
            torch.save(dict(kind='DISPOSABLE_STRICT_ONLINE_PROBE_NOT_FORMAL_TRAINING',
                reward_contract=report['worker']['reward_contract'],
                source_sha256=report['source_sha256'],
                learner=learner.state_dict()), stream)
        verified = evidence['closed_loop_learning_verified']
        if args.throughput:
            decision_times = [x['decision_wall_seconds'] for x in report['profile_events']]
            report['throughput'] = dict(
                total_decision_seconds=sum(decision_times),
                mean_decision_seconds=sum(decision_times)/len(decision_times),
                decisions_per_second=len(decision_times)/sum(decision_times),
                min_decision_seconds=min(decision_times),
                max_decision_seconds=max(decision_times),
                warmup_seconds=sum(e['elapsed_seconds'] for e in report['events']
                                   if e['kind'] == 'warmup'))
        report.update(evidence,
            status=(('PASS_STRICT_ONLINE_THROUGHPUT_PROBE' if verified else
                     'INCOMPLETE_ONLINE_THROUGHPUT_EVIDENCE') if args.throughput else
                    ('PASS_STRICT_ONLINE_SELECTOR_ONLY_PROBE' if verified else
                     'INCOMPLETE_ONLINE_LEARNING_EVIDENCE')),
            real_generator=True, live_render_verified=True, real_generated_reward=True,
            online_update=learner.version > 0, real_closed_loop=verified,
            actual_optimizer_steps=learner.version, update_attempts=learner.attempts,
            frozen_generator_and_v3_unchanged=True, frozen_generator_sha256=initial_model_hash,
            residual_changed=initial_residual_hash != online.last_residual_hash,
            policy_version=learner.version, step0_v3_parity=True, next_step_waited_for_update=True,
            generated_steps=args.steps, candidate_groups=args.steps, candidate_branches=args.steps*20,
            warmup_frames=13, history_frames=4, next_observation_checks=args.steps-1,
            learner_checkpoint=dict(path=str(checkpoint), sha256=sha256_file(checkpoint), disposable=True),
            idm_fallbacks=closed['idm_fallbacks'], reaction='R')
    except BaseException as error:
        report.update(status='FAIL_STRICT_ONLINE_SELECTOR_ONLY', error=repr(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, worker_log=str(worker_log))
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(status=report['status'], report=str(output))), flush=True)
        if live is not None: live.close()
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        for resource in (left, right):
            if resource is not None: resource.close()
        if images is not None: images.close()
    return 0 if report['status'] in ('PASS_STRICT_ONLINE_SELECTOR_ONLY_PROBE',
                                     'PASS_STRICT_ONLINE_THROUGHPUT_PROBE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
