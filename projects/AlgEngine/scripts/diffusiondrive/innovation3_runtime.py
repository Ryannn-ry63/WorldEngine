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
    parser.add_argument('--scene-count', type=int, default=2, help='snapshot/protocol probes')
    parser.add_argument('--steps', type=int, default=None, help='default 8 for snapshot-probe, 2 for h1-protocol-probe')
    parser.add_argument('--seed', type=int, default=0, help='snapshot/protocol probes')
    parser.add_argument('stage', choices=['preflight', 'learner-smoke', 'snapshot-probe', 'h1-protocol-probe', 'visual-probe'])
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 2 if args.stage == 'h1-protocol-probe' else (1 if args.stage == 'visual-probe' else 8)
    cfg, env = environment(args.settings, args.devices)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    modules = {'preflight': 'innovation3.preflight', 'learner-smoke': 'innovation3.smoke',
               'snapshot-probe': 'innovation3.snapshot_probe'}
    modules['h1-protocol-probe'] = 'innovation3.h1_protocol_probe'
    modules['visual-probe'] = 'innovation3.visual_online_probe'
    python = cfg['simengine_python' if args.stage == 'snapshot-probe' else 'algengine_python']
    command = [python, '-m', modules[args.stage],
               '--settings', str(args.settings.absolute()), '--output', str(args.output.absolute())]
    if args.stage == 'preflight':
        command += ['--probe', '--required-gpus', str(len(args.devices.split(',')))]
    elif args.stage == 'learner-smoke':
        if len(args.devices.split(',')) != 1:
            parser.error('Learner smoke is single-rank; it is not a DDP test')
        command += ['--device', 'cuda:0']
    elif args.stage == 'visual-probe':
        if len(args.devices.split(',')) != 1:
            parser.error('Visual probe is single-GPU; use one H100 first')
        env.update(OPENBLAS_CORETYPE='Prescott', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
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
    with tempfile.TemporaryDirectory(prefix='innovation3-mpl-') as cache:
        env['MPLCONFIGDIR'] = cache
        return subprocess.run(command, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
