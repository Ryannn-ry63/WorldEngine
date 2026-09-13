"""Immutable, observed-best P1 release; this is NOT a promotion of P3."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d

METHOD = 'selector_p1_seed2_snapshot_v1'
POLICIES = ('scalar_v3', 'P1_seed2')
FORMAL_NOISE = 'formal_navtest_seed0'
SELECTOR_SHA = '331ac4459cef19febff865080e3015f955cae0d2ebbd04bed58e207b2b8ebc6b'
PREFIX = 'planning_head.scene_selector.'
AGGREGATES = {'average', 'overall_average'}
EXPOSURE = dict(development_selected_single_checkpoint=True, optimization_seed=2,
                evaluation_seed=0, independent_blind_test=False, three_seed_main_result=False,
                original_P3_gate_unchanged=True, training_authorized=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def preserved_copy(source, destination):
    source, destination = Path(source), Path(destination)
    expected = c.sha256_file(source)
    if destination.exists():
        if c.sha256_file(destination) != expected:
            raise RuntimeError('Archive collision: '+str(destination))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix+'.tmp')
        shutil.copy2(source, temp)
        if c.sha256_file(temp) != expected:
            raise RuntimeError('Copy verification failed')
        temp.replace(destination)
    return d.artifact(destination)


def real_rows(path, expected=None, valid=False):
    with Path(path).open() as stream:
        raw = list(csv.DictReader(stream))
    rows, seen, aggregates = [], set(), set()
    for row in raw:
        token = row.get('token', '')
        if token in AGGREGATES:
            if token in aggregates:
                raise RuntimeError('Duplicate aggregate row')
            aggregates.add(token)
            continue
        if not token or token in seen:
            raise RuntimeError('Missing/duplicate scene or token')
        if valid and row.get('valid', '').lower() != 'true':
            raise RuntimeError('Invalid official PDM token: '+token)
        seen.add(token)
        rows.append(row)
    if not rows or expected is not None and seen != set(expected):
        raise RuntimeError('Exact token membership mismatch')
    return rows


def metrics(path, expected=None, valid=False):
    rows = real_rows(path, expected, valid)
    result = {}
    for row in rows:
        v = {k: float(row[k]) for k in d.METRICS if k != 'success'}
        if not all(np.isfinite(z) for z in v.values()):
            raise RuntimeError('Nonfinite metrics')
        v['success'] = float(v['no_at_fault_collisions'] == 1 and v['drivable_area_compliance'] == 1)
        result[row['token']] = v
    return result


def mean(values):
    return {k: float(np.mean([v[k] for v in values.values()])) for k in d.METRICS}


def conditions():
    # Each model: complete historical four blocks plus NR/R common retention.
    return [(p, block) for p in POLICIES for block in
            ('open_navtest', 'open_rare', 'full_NR', 'full_R', 'common_NR', 'common_R')]


def checked_inputs(run):
    manifest = d.read(Path(run)/'inputs.json')
    for entry in manifest['artifacts'].values():
        d.verify(entry)
    for entry in manifest['models'].values():
        d.verify(entry)
    return manifest


def validate_native(sidecar, collection, code_sha, terminal=False):
    if (sidecar.get('checkpoint_sha256') != collection['checkpoint']['sha256']
            or sidecar.get('candidate_noise_namespace') != collection['noise']
            or sidecar.get('code_sha') != code_sha
            or sidecar.get('cfpi_deployment') is not None):
        raise RuntimeError('Native inference provenance/router mismatch')
    contract = sidecar.get('selector_rollout_contract', {})
    if (contract.get('experiment') != METHOD or contract.get('condition') != collection['id']
            or contract.get('react_type') != collection['mode']
            or contract.get('inference_uses_reward_or_q') is not False):
        raise RuntimeError('Native observer contract mismatch')
    prefixes = collection['prefixes']
    prefix = sidecar['scene_prefix']
    if prefix not in prefixes:
        raise RuntimeError('Unknown native scene prefix')
    step = int(sidecar['planner_step'])
    if step not in ((12,) if terminal else c.DECISION_STEPS):
        raise RuntimeError('Unexpected consumed/unconsumed step')
    from audit_selector_cfpi_deployment import SHAPES
    for key, shape in SHAPES.items():
        c.checked_array(sidecar[key], shape, key)
    c.checked_array(sidecar['deployed_trajectory'], (40,3), 'deployed trajectory')
    selected = int(np.argmax(sidecar['current_logits']))
    if selected != int(sidecar['selected_index']) or selected != int(sidecar['selected_indices']):
        raise RuntimeError('Native selector is not deployed argmax')
    return dict(scene_id=prefixes[prefix], decision_step=step, selected_index=selected)
