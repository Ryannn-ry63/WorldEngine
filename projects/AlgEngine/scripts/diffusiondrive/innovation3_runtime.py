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
    parser.add_argument('--scene-count', type=int, default=2, help='snapshot-probe only')
    parser.add_argument('--steps', type=int, default=8, help='snapshot-probe only')
    parser.add_argument('--seed', type=int, default=0, help='snapshot-probe only')
    parser.add_argument('stage', choices=['preflight', 'learner-smoke', 'snapshot-probe'])
    args = parser.parse_args()
    cfg, env = environment(args.settings, args.devices)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    modules = {'preflight': 'innovation3.preflight', 'learner-smoke': 'innovation3.smoke',
               'snapshot-probe': 'innovation3.snapshot_probe'}
    python = cfg['simengine_python' if args.stage == 'snapshot-probe' else 'algengine_python']
    command = [python, '-m', modules[args.stage],
               '--settings', str(args.settings.absolute()), '--output', str(args.output.absolute())]
    if args.stage == 'preflight':
        command += ['--probe', '--required-gpus', str(len(args.devices.split(',')))]
    elif args.stage == 'learner-smoke':
        if len(args.devices.split(',')) != 1:
            parser.error('Learner smoke is single-rank; it is not a DDP test')
        command += ['--device', 'cuda:0']
    else:
        # Dynamics workers use CPU only; never disturb an occupied GPU.
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command += ['--scene-count', str(args.scene_count), '--steps', str(args.steps), '--seed', str(args.seed)]
    with tempfile.TemporaryDirectory(prefix='innovation3-mpl-') as cache:
        env['MPLCONFIGDIR'] = cache
        return subprocess.run(command, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
