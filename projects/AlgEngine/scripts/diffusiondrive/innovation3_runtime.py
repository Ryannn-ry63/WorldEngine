#!/usr/bin/env python3
"""Run bounded Innovation 3 checks in the explicitly configured environment."""
import argparse
from pathlib import Path
import subprocess
from selector_runtime import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', default='0')
    parser.add_argument('stage', choices=['preflight', 'learner-smoke'])
    args = parser.parse_args()
    cfg, env = environment(args.settings, args.devices)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [cfg['algengine_python'], '-m',
               'innovation3.preflight' if args.stage == 'preflight' else 'innovation3.smoke',
               '--settings', str(args.settings.absolute()), '--output', str(args.output.absolute())]
    if args.stage == 'preflight':
        command += ['--probe', '--required-gpus', str(len(args.devices.split(',')))]
    else:
        if len(args.devices.split(',')) != 1:
            parser.error('Learner smoke is single-rank; it is not a DDP test')
        command += ['--device', 'cuda:0']
    return subprocess.run(command, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
