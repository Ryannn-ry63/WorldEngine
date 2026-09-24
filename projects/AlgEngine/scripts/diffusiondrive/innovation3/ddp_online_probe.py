"""Real-feedback multi-rank online throughput pilot.

Each rank runs the already accepted resident generator/SimEngine/reward loop
on one GPU. Only the trainable selector residual gradients are averaged. This
is an engineering pilot, never formal training or a scaling claim.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import timedelta
from .paths import checked_path, sha256_file
from .scene_manifest import load_assignments, validate_scene_reports

torch = None
dist = None
online_probe = None
state_digest = None


def validate_bindings(bindings, world):
    """Validate rank/device ownership even on torch builds without UUIDs."""
    if len(bindings) != world:
        raise RuntimeError('Expected one binding record per rank: ' + repr(bindings))
    if any(x.get('current_device') != x.get('local_rank') for x in bindings):
        raise RuntimeError('DDP local device mismatch: ' + repr(bindings))
    local_ranks = [x.get('local_rank') for x in bindings]
    if sorted(local_ranks) != list(range(world)):
        raise RuntimeError('DDP local ranks are not a permutation: ' + repr(bindings))
    visible = [tuple(x.get('visible_devices', ())) for x in bindings]
    if len(set(visible)) != 1 or len(visible[0]) < world:
        raise RuntimeError('DDP visible-device lists disagree: ' + repr(bindings))
    physical = [x['visible_devices'][x['local_rank']] for x in bindings]
    if len(set(physical)) != world:
        raise RuntimeError('DDP physical devices are duplicated: ' + repr(bindings))
    uuids = [x.get('cuda_uuid') for x in bindings]
    usable_uuids = [u for u in uuids if u and u != 'unknown']
    if usable_uuids and (len(usable_uuids) != world or len(set(usable_uuids)) != world):
        raise RuntimeError('DDP CUDA UUIDs are duplicated/incomplete: ' + repr(bindings))
    return 'cuda_uuid' if len(usable_uuids) == world else 'visible_device_local_rank'


def coordinate_update(parameters, local_optimized):
    """Synchronize signal presence and average zero-filled gradients."""
    world = dist.get_world_size()
    parameters = list(parameters)
    active = torch.tensor(int(local_optimized), device=parameters[0].device,
                          dtype=torch.int32)
    dist.all_reduce(active, op=dist.ReduceOp.MAX)
    globally_optimized = bool(active.item())
    if not globally_optimized:
        return False
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)
    return True


def distributed_learner_class(base):
    class DistributedOnlineV3Learner(base):
        def __init__(self, selector, seed=0):
            super().__init__(selector, seed=seed, update_coordinator=coordinate_update)
            initial = state_digest(dict(reference=self.reference.state_dict(), selector=self.selector.state_dict()))
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, initial)
            if len(set(gathered)) != 1:
                raise RuntimeError('DDP ranks initialized from different V3/residual states')

        def update(self, rewards, decision_id):
            result = super().update(rewards, decision_id)
            local = state_digest(dict(selector=self.selector.state_dict(),
                optimizer=self.optimizer.state_dict(), version=self.version, attempts=self.attempts))
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            if len(set(gathered)) != 1:
                raise RuntimeError('Real DDP residual state mismatch: ' + repr(gathered))
            result['ddp_state_parity'] = True
            return result
    return DistributedOnlineV3Learner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene-manifest', type=Path)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--branch-workers', type=int, default=0,
                        help='experimental persistent CPU SimEngine branch workers per rank')
    args = parser.parse_args()
    if not 0 <= args.branch_workers <= 20:
        parser.error('--branch-workers must be in [0, 20]')
    rank = int(os.environ.get('RANK', '-1'))
    world = int(os.environ.get('WORLD_SIZE', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
    if rank < 0 or world < 2 or local_rank < 0:
        parser.error('Run through torch.distributed.run with at least two ranks')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = [x.strip() for x in visible.split(',') if x.strip()]
    if len(devices) < world or local_rank >= len(devices):
        parser.error('CUDA_VISIBLE_DEVICES must expose one device per rank')
    manifest = rows = None
    settings = checked_path(args.settings)
    output = checked_path(args.output.absolute(), must_exist=False)
    if output.exists():
        raise FileExistsError('Use a fresh output basename: ' + str(output))
    if args.scene_manifest:
        manifest, rows = load_assignments(args.scene_manifest, settings, world, verify_files=False)
        if args.steps != manifest['steps']:
            parser.error('Steps differ from the explicit scene manifest')
        row = rows[rank]
        cfg = json.loads(settings.read_text())
        cfg.update(visual_scene_id=row['scene_id'], online_candidate_seed=row['candidate_seed'],
                   online_scene_seed=row['scene_seed'], online_action_seed=args.seed + rank)
        settings = output.with_name(output.stem + '.rank' + str(rank) + '.settings.json')
        settings.parent.mkdir(parents=True, exist_ok=True)
        with settings.open('x') as stream:
            json.dump(cfg, stream, indent=2)
    global torch, dist, online_probe, state_digest
    import torch as _torch
    import torch.distributed as _dist
    # Standard torchrun binding: all ranks retain the explicit visible-device
    # list, then select their unique local device before NCCL/model creation.
    _torch.cuda.set_device(local_rank)
    os.environ['WORLDENGINE_DDP_LOCAL_GPU'] = devices[local_rank]
    from . import online_probe as _online_probe
    from .learner import OnlineV3Learner as _OnlineV3Learner
    from .live_learning import state_digest as _state_digest
    torch, dist, online_probe, state_digest = _torch, _dist, _online_probe, _state_digest
    DistributedOnlineV3Learner = distributed_learner_class(_OnlineV3Learner)
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=300))
    rank_output = args.output.with_name(args.output.stem + '.rank' + str(rank) + args.output.suffix)
    original_class = online_probe.OnlineV3Learner
    online_probe.OnlineV3Learner = DistributedOnlineV3Learner
    try:
        argv = ['online-probe', '--settings', str(settings.absolute()),
                '--output', str(rank_output.absolute()), '--steps', str(args.steps),
                '--seed', str(args.seed), '--throughput']
        if args.branch_workers:
            argv += ['--branch-workers', str(args.branch_workers)]
        with _argv(argv):
            code = online_probe.main()
        rank_report = rank_output
        if rank_report.exists():
            payload = json.loads(rank_report.read_text())
            properties = torch.cuda.get_device_properties(local_rank)
            payload['ddp_binding'] = dict(
                rank=rank, local_rank=local_rank,
                visible_devices=devices, torch_device_count=torch.cuda.device_count(),
                current_device=torch.cuda.current_device(),
                cuda_uuid=getattr(properties, 'uuid', None),
                cuda_name=properties.name)
            rank_report.write_text(json.dumps(payload, indent=2))
        codes = [None] * world
        dist.all_gather_object(codes, code)
        if any(value not in (0, 2) for value in codes):
            raise RuntimeError('Rank online probe failed: ' + repr(codes))
        result_code = [2]
        if rank == 0:
            rank_reports = [json.loads(args.output.with_name(
                args.output.stem + '.rank' + str(i) + args.output.suffix).read_text())
                            for i in range(world)]
            if not all(x.get('status') in ('PASS_STRICT_ONLINE_THROUGHPUT_PROBE',
                                          'INCOMPLETE_ONLINE_THROUGHPUT_EVIDENCE')
                       for x in rank_reports):
                raise RuntimeError('Not all real DDP ranks passed')
            if not all(all(e.get('update', {}).get('ddp_state_parity') is True
                           for e in x.get('events', []) if e.get('kind') == 'online_update')
                       for x in rank_reports):
                raise RuntimeError('Missing per-update real DDP state parity')
            if manifest is not None:
                validate_scene_reports(rank_reports, rows, manifest)
            if not all(x.get('update_attempts') == args.steps and
                       x.get('candidate_branches') == args.steps * 20 and
                       x.get('frozen_generator_and_v3_unchanged') is True and
                       x.get('idm_fallbacks') == 0 for x in rank_reports):
                raise RuntimeError('Incomplete real execution/frozen evidence')
            traces = [[(e['version_before'], e['version_after'], e['optimized'])
                       for e in x['events'] if e.get('kind') == 'online_update'] for x in rank_reports]
            if any(len(trace) != args.steps for trace in traces) or any(trace != traces[0] for trace in traces):
                raise RuntimeError('DDP ranks have different optimizer/version schedules')
            complete = all(x['status'] == 'PASS_STRICT_ONLINE_THROUGHPUT_PROBE' for x in rank_reports)
            result_code[0] = 0 if complete else 2
            bindings = [x.get('ddp_binding', {}) for x in rank_reports]
            binding_verification = validate_bindings(bindings, world)
            report = dict(status=('PASS_DDP_REAL_ONLINE_THROUGHPUT_PILOT' if complete else
                                      'INCOMPLETE_DDP_ONLINE_LEARNING_EVIDENCE'),
                stage='DDP_REAL_ONLINE_THROUGHPUT', world_size=world, steps=args.steps,
                rank_reports=[dict(rank=i, status=x['status'], gpu=x.get('gpu'),
                    scene=x.get('worker', {}).get('scene'), seed_namespaces=x.get('seed_namespaces'),
                    local_groups_with_signal=x.get('local_groups_with_signal'),
                    branch_execution=x.get('branch_execution'), branch_workers=x.get('branch_workers', 0),
                    ddp_binding=x.get('ddp_binding'),
                    throughput=x.get('throughput'), actual_optimizer_steps=x.get('actual_optimizer_steps'),
                    update_attempts=x.get('update_attempts'), ddp_state_parity=True)
                    for i, x in enumerate(rank_reports)],
                real_generator=True, live_render_verified=True, real_generated_reward=True,
                real_closed_loop=complete, ddp_verified=True, frozen_generator_and_v3_unchanged=True,
                ddp_binding_verification=binding_verification,
                formal_ready=False, used_for_formal_training=False,
                source_sha256=rank_reports[0].get('source_sha256'),
                branch_execution=rank_reports[0].get('branch_execution'), branch_workers=args.branch_workers,
                rank_reports_paths=[str(args.output.with_name(
                    args.output.stem + '.rank' + str(i) + args.output.suffix)) for i in range(world)])
            report['distinct_scene_verified'] = manifest is not None
            report['unique_scene_count'] = len({x['worker']['scene'] for x in rank_reports})
            if manifest is not None:
                report['scene_manifest'] = dict(path=str(args.scene_manifest.resolve()),
                    sha256=sha256_file(args.scene_manifest), assigned_scenes=rows)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                json.dump(report, stream, indent=2)
        dist.broadcast_object_list(result_code, src=0)
        return result_code[0]
    finally:
        online_probe.OnlineV3Learner = original_class
        dist.destroy_process_group()


class _argv:
    def __init__(self, values):
        self.values = values
        self.previous = None
    def __enter__(self):
        self.previous = sys.argv
        sys.argv = self.values
    def __exit__(self, *_):
        sys.argv = self.previous


if __name__ == '__main__':
    raise SystemExit(main())
