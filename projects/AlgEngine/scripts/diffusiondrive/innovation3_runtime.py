#!/usr/bin/env python3
"""Run bounded Innovation 3 checks in the explicitly configured environment."""
import argparse
from pathlib import Path
import subprocess
import tempfile
from selector_runtime import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--scene-manifest', type=Path, help='explicit distinct-scene DDP engineering manifest')
    parser.add_argument('--scene-count', type=int, default=2, help='snapshot/protocol probes')
    parser.add_argument('--steps', type=int, default=None, help='default 8 for snapshot-probe, 2 for h1-protocol-probe')
    parser.add_argument('--seed', type=int, default=0, help='snapshot/protocol probes')
    parser.add_argument('--branch-workers', type=int, default=0,
                        help='persistent CPU SimEngine workers per online rank (efficiency pilot)')
    parser.add_argument('stage', choices=['preflight', 'learner-smoke', 'snapshot-probe', 'h1-protocol-probe', 'visual-probe', 'live-reward-probe', 'online-probe', 'online-throughput-probe', 'ddp-h1-probe', 'ddp-online-throughput-probe', 'reward-probe'])
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 2 if args.stage in ('h1-protocol-probe', 'ddp-h1-probe') else (1 if args.stage in ('visual-probe', 'live-reward-probe') else 8)
    if not 0 <= args.branch_workers <= 20:
        parser.error('--branch-workers must be in [0, 20]')
    if args.branch_workers and args.stage not in ('online-probe', 'online-throughput-probe', 'ddp-online-throughput-probe'):
        parser.error('--branch-workers is only supported by online throughput stages')
    if args.scene_manifest and args.stage != 'ddp-online-throughput-probe':
        parser.error('--scene-manifest is only supported for real DDP throughput')
    cfg, env = environment(args.settings, args.devices)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    modules = {'preflight': 'innovation3.preflight', 'learner-smoke': 'innovation3.smoke',
               'snapshot-probe': 'innovation3.snapshot_probe'}
    modules['h1-protocol-probe'] = 'innovation3.h1_protocol_probe'
    modules['visual-probe'] = 'innovation3.visual_online_probe'
    modules['live-reward-probe'] = 'innovation3.visual_online_probe'
    modules['online-probe'] = 'innovation3.online_probe'
    modules['online-throughput-probe'] = 'innovation3.online_probe'
    modules['ddp-online-throughput-probe'] = 'innovation3.ddp_online_probe'
    modules['ddp-h1-probe'] = 'innovation3.ddp_h1_probe'
    modules['reward-probe'] = 'innovation3.reward_probe'
    python = cfg['simengine_python' if args.stage in ('snapshot-probe', 'reward-probe') else 'algengine_python']
    command = [python, '-m', modules[args.stage],
               '--settings', str(args.settings.absolute()), '--output', str(args.output.absolute())]
    if args.stage == 'preflight':
        command += ['--probe', '--required-gpus', str(len(args.devices.split(',')))]
    elif args.stage == 'learner-smoke':
        if len(args.devices.split(',')) != 1:
            parser.error('Learner smoke is single-rank; it is not a DDP test')
        command += ['--device', 'cuda:0']
    elif args.stage in ('visual-probe', 'live-reward-probe', 'online-probe', 'online-throughput-probe'):
        if len(args.devices.split(',')) != 1:
            parser.error('Visual/online probe is single-GPU; use one H100 first')
        env.update(OPENBLAS_CORETYPE='Prescott', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command += ['--steps', str(args.steps), '--seed', str(args.seed)]
        if args.stage == 'live-reward-probe':
            command.append('--reward-adapter')
        if args.stage == 'online-throughput-probe':
            command.append('--throughput')
        if args.branch_workers:
            command += ['--branch-workers', str(args.branch_workers)]
    elif args.stage == 'ddp-h1-probe':
        count = len(args.devices.split(','))
        if count < 2:
            parser.error('DDP H1 pilot requires at least two explicit GPU IDs')
        env.update(OPENBLAS_CORETYPE='Prescott', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [python, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node', str(count), '-m', modules[args.stage],
                   '--settings', str(args.settings.absolute()),
                   '--output', str(args.output.absolute()), '--steps', str(args.steps),
                   '--resume-after', str(max(1, args.steps // 2)), '--seed', str(args.seed)]
    elif args.stage == 'ddp-online-throughput-probe':
        count = len(args.devices.split(','))
        if count < 2:
            parser.error('Real DDP online pilot requires at least two explicit GPU IDs')
        env.update(OPENBLAS_CORETYPE='Prescott', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [python, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node', str(count), '-m', modules[args.stage],
                   '--settings', str(args.settings.absolute()),
                   '--output', str(args.output.absolute()), '--steps', str(args.steps),
                   '--seed', str(args.seed)]
        if args.branch_workers:
            command += ['--branch-workers', str(args.branch_workers)]
    elif args.stage == 'reward-probe':
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_CORETYPE='Prescott')
        command += ['--steps', str(args.steps), '--seed', str(args.seed)]
    elif args.stage == 'snapshot-probe':
        # Dynamics workers use CPU only; never disturb an occupied GPU.
        # Pin before the child imports NumPy. The installed OpenBLAS 0.3.20
        # Cooperlake path fails SVD/pinv health checks on our target runtime.
        # Use this same numeric policy for future rendered-main/branch workers.
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_CORETYPE='Prescott')
        command += ['--scene-count', str(args.scene_count), '--steps', str(args.steps), '--seed', str(args.seed)]
    else:
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_CORETYPE='Prescott')
        command += ['--scene-count', str(args.scene_count), '--steps', str(args.steps), '--seed', str(args.seed)]
    if args.scene_manifest:
        from innovation3.scene_manifest import load_assignments
        manifest, _ = load_assignments(args.scene_manifest, args.settings, len(args.devices.split(',')))
        if args.steps != manifest['steps']:
            parser.error('Steps differ from the explicit scene manifest')
        command += ['--scene-manifest', str(args.scene_manifest.absolute())]
    with tempfile.TemporaryDirectory(prefix='innovation3-mpl-') as cache:
        env['MPLCONFIGDIR'] = cache
        return subprocess.run(command, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
