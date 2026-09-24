"""Explicit, checksummed engineering scene assignments; never a training split."""
import json
import re
from pathlib import Path
from .paths import checked_path, sha256_file


def origin_log(name):
    """Conservatively group timestamp/vehicle logs across numbered clip windows."""
    if not isinstance(name, str) or not re.fullmatch(
            r'\d{4}(?:\.\d{2}){5}_veh-\d+(?:_\d+_\d+)?', name):
        raise ValueError('Unsupported origin log: ' + repr(name))
    return re.sub(r'_\d+_\d+$', '', name)


def load_assignments(manifest_path, settings_path, world, verify_files=True):
    manifest_path = checked_path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    accepted_purposes = {
        'distinct_scene_engineering_pilot',
        'distinct_scene_engineering_pilot_v3_initialized_selector',
    }
    if (manifest.get('schema') != 1 or manifest.get('purpose') not in accepted_purposes
            or manifest.get('formal_ready') is not False):
        raise ValueError('Expected a non-formal engineering scene manifest')
    if manifest.get('reaction') != 'R' or manifest.get('steps') != 8:
        raise ValueError('Only the registered R / 8-decision engineering protocol is supported')
    if not re.fullmatch('[0-9a-f]{64}', manifest.get('scenario_source_sha256', '')):
        raise ValueError('Missing scenario source SHA256')
    if sha256_file(settings_path) != manifest.get('settings_sha256'):
        raise ValueError('Scene manifest settings SHA256 mismatch')
    cfg = json.loads(checked_path(settings_path).read_text())
    if manifest.get('purpose') == 'distinct_scene_engineering_pilot_v3_initialized_selector':
        if cfg.get('online_parameterization') != 'v3_initialized_selector_finetune':
            raise ValueError('Initialized-selector manifest requires initialized-selector settings')
        if manifest.get('parameterization') != cfg.get('online_parameterization'):
            raise ValueError('Scene manifest parameterization mismatch')
    elif cfg.get('online_parameterization') not in (None, 'frozen_v3_plus_zero_residual'):
        raise ValueError('Legacy scene manifest cannot use initialized-selector settings')
    rows = manifest.get('scenes', [])
    if not isinstance(rows, list) or not 2 <= world <= len(rows):
        raise ValueError('Not enough explicitly assigned scenes for world size')
    # Validate the whole manifest, then use a fixed prefix for 2/8-rank staging.
    ids, logs = set(), set()
    for row in rows:
        sid = row['scene_id']
        if not isinstance(sid, str) or Path(sid).name != sid or sid in ('.', '..'):
            raise ValueError('Invalid scene ID')
        log = origin_log(row['origin_log'])
        if sid in ids or log in logs:
            raise ValueError('Duplicate scene or origin log in DDP manifest')
        if origin_log(sid.rsplit('-', 1)[0]) != log:
            raise ValueError('Scene ID and origin log disagree')
        ids.add(sid); logs.add(log)
        for key in ('scene_seed', 'candidate_seed'):
            if type(row.get(key)) is not int or not 0 <= row[key] < 2**31:
                raise ValueError('Invalid seed namespace: ' + key)
        for key in ('asset_sha256', 'scene_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', row.get(key, '')):
                raise ValueError('Missing checksum: ' + key)
    source = checked_path(Path(cfg['scenario_root'])/'original/navtrain_failures_per1/all_scenarios.pkl')
    if verify_files:
        if sha256_file(source) != manifest.get('scenario_source_sha256'):
            raise ValueError('Registered scenario source changed')
        for row in rows[:world]:
            asset = checked_path(Path(cfg['asset_root'])/row['scene_id']/'background'/(row['scene_id']+'.ckpt'))
            if sha256_file(asset) != row['asset_sha256']:
                raise ValueError('Assigned exact asset checksum mismatch: ' + row['scene_id'])
            scene = checked_path(row['scene_path'])
            if sha256_file(scene) != row['scene_sha256']:
                raise ValueError('Assigned scene checksum mismatch: ' + row['scene_id'])
    return manifest, rows[:world]


def validate_scene_reports(reports, rows, manifest):
    """Fail closed on wrong worker, seed, source, asset or incomplete execution."""
    if len(reports) != len(rows):
        raise RuntimeError('Missing rank reports')
    for report, row in zip(reports, rows):
        worker = report.get('worker', {})
        if worker.get('scene') != row['scene_id']:
            raise RuntimeError('Worker used a different scene than its assignment')
        source_ok = (
            worker.get('source', {}).get('path') == row['scene_path'] and
            worker.get('source', {}).get('sha256') == row['scene_sha256'])
        # Older reports loaded the combined source. Keep them readable while
        # requiring the new direct-scene path to prove its own checksum.
        legacy_source_ok = worker.get('source', {}).get('sha256') == manifest['scenario_source_sha256']
        if (worker.get('asset', {}).get('sha256') != row['asset_sha256'] or
                not (source_ok or legacy_source_ok)):
            raise RuntimeError('Worker source/asset provenance mismatch')
        seeds = report.get('seed_namespaces', {})
        if any(seeds.get(k) != row[k] for k in ('candidate_seed', 'scene_seed')):
            raise RuntimeError('Worker seed namespace mismatch')
    return True
