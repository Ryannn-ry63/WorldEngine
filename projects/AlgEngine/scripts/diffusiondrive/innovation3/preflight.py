"""Read-only runtime probe. Success certifies only the checks actually performed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .paths import checked_path, sha256_file

PROBE = '''
import importlib, json, pathlib, sys
result = {'python': sys.executable, 'modules': {}, 'errors': []}
for name in sys.argv[1:]:
    try:
        m = importlib.import_module(name)
        result['modules'][name] = str(pathlib.Path(m.__file__).resolve()) if getattr(m, '__file__', None) else None
    except Exception as e:
        result['errors'].append(name + ': ' + repr(e))
try:
    import torch
    result.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(), gpu_count=torch.cuda.device_count())
    if torch.cuda.is_available():
        result['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        x = torch.ones(4, device='cuda'); result['cuda_sum'] = x.sum().item(); torch.cuda.synchronize()
except Exception as e:
    result['errors'].append('cuda: ' + repr(e))
result['sys_path'] = sys.path
print('INNOVATION3_PROBE=' + json.dumps(result))
'''


def audit(settings, probe=False, required_gpus=1):
    import selector_runtime
    cfg, env = selector_runtime.environment(settings, ','.join(map(str, range(required_gpus))))
    report = dict(status='FAIL_PREFLIGHT', paths=[], environments={}, errors=[],
                  coverage_verified=False, model_forward_verified=False,
                  closed_loop_verified=False, formal_ready=False)
    entries = [(k, v) for k, v in cfg.items() if isinstance(v, str) and v.startswith('/')]
    entries += [('extra_python_paths', v) for v in cfg.get('extra_python_paths', [])]
    for key in ('scenario_root', 'asset_root', 'map_root', 'selector_state'):
        if not cfg.get(key):
            report['errors'].append('Missing online path: ' + key)
    for key, value in entries:
        item = dict(key=key, path=value)
        try:
            item['realpath'] = str(checked_path(value))
            expected = cfg.get('expected_sha256', {}).get(key)
            if expected:
                item['sha256'] = sha256_file(value)
                if item['sha256'] != expected:
                    raise ValueError('SHA256 mismatch')
            item['ok'] = True
        except (OSError, ValueError, RuntimeError) as error:
            item.update(ok=False, error=str(error))
            report['errors'].append(key + ': ' + str(error))
        report['paths'].append(item)
    for key in ('baseline', 'selector_state', 'anchors'):
        if not cfg.get('expected_sha256', {}).get(key):
            report['errors'].append('Missing expected SHA256: ' + key)
    if probe:
        for kind, modules in [('algengine', ['torch', 'mmcv', 'mmcv._ext', 'nuplan', 'navsim']),
                              ('simengine', ['torch', 'hydra', 'gsplat', 'worldengine'])]:
            try:
                python = str(checked_path(cfg[kind + '_python']))
                proc = subprocess.run([python, '-u', '-c', PROBE, *modules], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, timeout=180)
                record = dict(returncode=proc.returncode, stderr=proc.stderr[-6000:])
                lines = [line for line in proc.stdout.splitlines() if line.startswith('INNOVATION3_PROBE=')]
                if proc.returncode != 0 or len(lines) != 1:
                    raise RuntimeError(str(record) + '\n' + proc.stdout[-2000:])
                record.update(json.loads(lines[0].split('=', 1)[1]))
                for value in [record['python'], *record['modules'].values(), *record['sys_path']]:
                    if value and Path(value).is_absolute():
                        checked_path(value, must_exist=False)
                report['environments'][kind] = record
                report['errors'].extend(kind + ': ' + error for error in record['errors'])
                if record.get('gpu_count', 0) < required_gpus or not record.get('cuda_available'):
                    report['errors'].append(kind + ': insufficient visible CUDA devices')
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                report['errors'].append(kind + ': ' + str(error))
    if not report['errors']:
        report['status'] = 'PASS_RUNTIME_PROBE_ONLY' if probe else 'PASS_PATHS_ONLY'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--required-gpus', type=int, default=1)
    args = parser.parse_args()
    if args.required_gpus < 1:
        parser.error('required-gpus must be positive')
    output = checked_path(args.output, must_exist=False)
    code = Path(__file__).resolve().parents[5]
    if output == code or code in output.parents:
        parser.error('Write private runtime reports outside the code repository')
    if output.exists():
        parser.error('Report already exists; use a fresh attempt')
    report = audit(args.settings, args.probe, args.required_gpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], errors=report['errors'], report=str(output)), indent=2))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
