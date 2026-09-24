"""Bounded multi-rank learner/resume pilot for Innovation 3.

This is an engineering acceptance probe, not formal training.  Each rank owns
an independent synthetic H1 context and reward group, averages selector
gradients through the configured process group, and verifies exact state
parity at a scene-boundary checkpoint.  It deliberately does not claim live
rendering, real reward, or paper metrics.
"""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

from grpo_selector_v3_cached_common import SceneConditionedTrajectorySetSelector
from .learner import OnlineV3Learner
from .paths import checked_path, sha256_file


def digest_state(state):
    h = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            h.update(b'tensor')
            h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.numpy().tobytes())
        elif isinstance(value, dict):
            h.update(b'dict')
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            h.update(type(value).__name__.encode())
            for item in value:
                visit(item)
        else:
            h.update(type(value).__name__.encode())
            h.update(repr(value).encode())
    visit(state)
    return h.hexdigest()


def reduce_gradients(parameters):
    """Average every gradient and fail if ranks disagree on grad presence."""
    world = dist.get_world_size()
    for parameter in parameters:
        present = torch.tensor(1 if parameter.grad is not None else 0,
                               device=parameter.device, dtype=torch.int32)
        dist.all_reduce(present, op=dist.ReduceOp.MIN)
        present_max = present.clone()
        dist.all_reduce(present_max, op=dist.ReduceOp.MAX)
        if int(present.item()) != int(present_max.item()):
            raise RuntimeError('DDP gradient presence mismatch across ranks')
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world)


def context_for(device, seed, rank, step):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed * 100003 + rank * 1009 + step)
    def randn(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)
    return dict(candidate_features=randn((1, 20, 256)),
                candidate_trajectories=randn((1, 20, 8, 3)),
                route_bev_features=randn((1, 20, 8, 256)),
                status_token=randn((1, 1, 256)),
                ego_query=randn((1, 1, 256)),
                agents_query=randn((1, 30, 256)))


def common_context(device, seed):
    return context_for(device, seed, 0, 100000)


def assert_state_parity(learner, rank):
    value = digest_state(learner.state_dict())
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    if len(set(gathered)) != 1:
        raise RuntimeError('DDP state parity mismatch: ' + repr(gathered))
    return value


def build_learner(cfg, device, seed):
    selector_path = checked_path(cfg['selector_state'])
    if sha256_file(selector_path) != cfg['expected_sha256']['selector_state']:
        raise ValueError('Registered V3 selector SHA256 mismatch')
    payload = torch.load(selector_path, map_location='cpu')
    if payload.get('schema_version') != 3 or payload.get('method') != 'scene_conditioned_exact_group_grpo':
        raise ValueError('Expected standard trained V3 selector schema')
    model = SceneConditionedTrajectorySetSelector(**payload['scene_selector_config']).to(device).float().eval()
    model.load_state_dict(payload['scene_selector_state'], strict=True)
    return OnlineV3Learner(model, seed=seed, gradient_reducer=reduce_gradients), model


def run(args):
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    if world < 2:
        raise RuntimeError('ddp-h1-probe requires at least two ranks')
    if not torch.cuda.is_available():
        raise RuntimeError('ddp-h1-probe requires CUDA; use unit tests for CPU protocol checks')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    started = time.monotonic()
    report = None
    try:
        cfg = json.loads(checked_path(args.settings).read_text())
        learner, model = build_learner(cfg, device, args.seed)
        initial_reference = digest_state(learner.reference.state_dict())
        initial_model = digest_state(model.state_dict())
        updates = []
        checkpoint_digest = None
        for step in range(args.steps):
            context = context_for(device, args.seed, rank, step)
            base = torch.randn((1, 20), generator=torch.Generator(device='cpu').manual_seed(
                args.seed * 1009 + rank * 17 + step)).to(device)
            selected, probabilities, version = learner.choose(context, base, ('scene', step))
            # Every rank receives a non-tied, full20 signal; rewards differ by
            # rank so the all-reduced gradient is genuinely exercised.
            rewards = torch.arange(20, device=device, dtype=torch.float32)[None, :] / 19
            rewards = torch.roll(rewards, shifts=(rank + step) % 20, dims=1)
            update = learner.update(rewards, ('scene', step))
            assert_state_parity(learner, rank)
            updates.append(dict(step=step, selected=selected, version_before=version,
                                version_after=learner.version, optimized=update['optimized'],
                                reward_std=update['reward_std']))
            if step + 1 == args.resume_after:
                state = learner.state_dict()
                buffer = io.BytesIO()
                torch.save(state, buffer)
                buffer.seek(0)
                restored, _ = build_learner(cfg, device, args.seed)
                restored.load_state_dict(torch.load(buffer, map_location=device))
                if digest_state(restored.state_dict()) != digest_state(state):
                    raise RuntimeError('Scene-boundary checkpoint digest mismatch')
                resume_context = common_context(device, args.seed)
                resume_base = torch.zeros((1, 20), device=device)
                left = learner.choose(resume_context, resume_base, 'resume')
                right = restored.choose(resume_context, resume_base, 'resume')
                if left[0] != right[0] or not torch.equal(left[1], right[1]):
                    raise RuntimeError('Checkpoint action/RNG resume mismatch')
                resume_rewards = torch.arange(20, device=device, dtype=torch.float32)[None, :]
                learner.update(resume_rewards, 'resume')
                restored.update(resume_rewards, 'resume')
                if digest_state(restored.state_dict()) != digest_state(learner.state_dict()):
                    raise RuntimeError('Checkpoint optimizer resume mismatch')
                checkpoint_digest = digest_state(state)
        assert_state_parity(learner, rank)
        if digest_state(learner.reference.state_dict()) != initial_reference:
            raise RuntimeError('Frozen V3 reference changed')
        if digest_state(model.state_dict()) != initial_model:
            raise RuntimeError('Input V3 selector changed; residual must be isolated')
        gathered_updates = [None] * world
        dist.all_gather_object(gathered_updates, updates)
        dist.barrier()
        if rank == 0:
            output = checked_path(args.output.absolute(), must_exist=False)
            if output.exists():
                raise FileExistsError('Use a fresh output basename: ' + str(output))
            report = dict(status='PASS_DDP_H1_PILOT_ONLY', stage='DDP_H1_PILOT',
                          world_size=world, steps=args.steps, resume_after=args.resume_after,
                          rank_updates=gathered_updates, checkpoint_digest=checkpoint_digest,
                          selector_sha256=sha256_file(cfg['selector_state']),
                          frozen_v3_unchanged=True, state_parity=True,
                          serialized_scene_boundary_resume=True, optimizer_resume_parity=True,
                          real_generator=False, real_render=False, real_reward=False,
                          real_closed_loop=False, used_for_training=False,
                          formal_ready=False, elapsed_seconds=time.monotonic()-started)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + '\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps(dict(status=report['status'], report=str(args.output)), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--resume-after', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.steps < 2 or not 1 <= args.resume_after < args.steps or args.seed < 0:
        parser.error('Use steps >=2, 1 <= resume-after < steps, and a nonnegative seed')
    run(args)


if __name__ == '__main__':
    raise SystemExit(main())
