#!/usr/bin/env python3
"""Strict full-benchmark table, from explicitly identified per-scene evaluator CSVs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

PDM_KEYS = ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
            'time_to_collision_within_bound', 'comfort', 'score')
AGGREGATES = {'average', 'overall_average'}
BLOCKS = ('openloop_navtest_ade', 'openloop_navtest_pdm',
          'openloop_failures_pdm', 'closedloop_nr', 'closedloop_r')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get('schema_version') != 1:
        raise ValueError('Unknown protocol schema')
    for key, count in [('navtest_tokens', 12146), ('rare_tokens', 288)]:
        ids = protocol[key]
        if (not isinstance(ids, list) or not all(isinstance(x, str) and x and x not in AGGREGATES for x in ids)
                or len(ids) != count or len(set(ids)) != count):
            raise ValueError(f'{key}: require the frozen {count} unique scene IDs')
    return protocol


def read_checked(path, expected, columns, require_valid=False):
    with Path(path).open(newline='') as f: raw = list(csv.DictReader(f))
    rows = []
    seen = set()
    aggregates = []
    for row in raw:
        token = row.get('token')
        if token in AGGREGATES:
            if token in aggregates: raise ValueError(f'{path}: duplicate aggregate {token}')
            aggregates.append(token)
            continue
        if not token or token in seen:
            raise ValueError(f'{path}: missing or duplicate token {token!r}')
        seen.add(token)
        if require_valid and row.get('valid', '').lower() != 'true':
            raise ValueError(f'{path}: invalid evaluator result for {token}')
        if 'valid' in row and row['valid'].lower() not in ('true', '1'):
            raise ValueError(f'{path}: invalid scene {token}')
        for col in columns:
            value = float(row[col])
            if not math.isfinite(value): raise ValueError(f'{path}: non-finite {col}: {token}')
            if value < 0 or (col in PDM_KEYS and value > 1):
                raise ValueError(f'{path}: out-of-range {col}: {value}')
        rows.append(row)
    expected = set(expected)
    if seen != expected:
        raise ValueError(f'{path}: missing={len(expected-seen)}, unexpected={len(seen-expected)}; '
                         f'missing_examples={sorted(expected-seen)[:5]}')
    return rows, aggregates


def mean(rows, key): return sum(float(r[key]) for r in rows)/len(rows)


def build(protocol_path, evaluation_path, model_name, note):
    protocol = load_protocol(protocol_path)
    ev = json.loads(Path(evaluation_path).read_text())
    if ev.get('status') != 'EVALUATOR_OUTPUTS_COMPLETE':
        raise ValueError('Need a completed evaluator output manifest, not a diagnostic status')
    if ev.get('protocol_sha256') != sha(protocol_path): raise ValueError('Protocol identity mismatch')
    if not isinstance(ev.get('eval_seed'), int): raise ValueError('Missing evaluation seed')
    ckpt = Path(ev['checkpoint']['path'])
    if sha(ckpt) != ev['checkpoint']['sha256']: raise ValueError('Checkpoint hash mismatch')
    metrics = {}; ignored = {}; counts = {}; all_rows = {}
    for block in BLOCKS:
        artifact = ev['artifacts'][block]
        path = Path(artifact['path'])
        if sha(path) != artifact['sha256']: raise ValueError('Metric hash mismatch: '+block)
        cols = ('ade_4s','fde_4s') if block.endswith('_ade') else PDM_KEYS
        tokens = protocol['navtest_tokens' if block.startswith('openloop_navtest') else 'rare_tokens']
        rows, agg = read_checked(path, tokens, cols, block.endswith('_pdm'))
        all_rows[block] = rows; ignored[block] = agg; counts[block] = len(rows)
        metrics[block] = {key: mean(rows,key) for key in cols}
    success = sum(float(r['no_at_fault_collisions']) == 1 and float(r['drivable_area_compliance']) == 1
                  for r in all_rows['closedloop_r']) / len(all_rows['closedloop_r'])
    values = [metrics['openloop_navtest_ade'][x] for x in ('ade_4s','fde_4s')]
    for block in BLOCKS[1:]: values.extend(metrics[block][k] for k in PDM_KEYS)
    values.append(success)
    headers = ['Model Name','Note','ADE','FDE']
    for block in BLOCKS[1:]:
        headers += [block + '/' + name for name in ('NC','DAC','EP','TTC','Comfort','PDM')]
    headers += ['Success Rate']
    row = [model_name, note] + [f'{v:.8f}' for v in values]
    assert len(row) == len(headers) == 29
    return dict(status='PASS_FULL_FORMAL_TABLE', formal_table_ready=True,
                protocol_sha256=sha(protocol_path), evaluation_manifest_sha256=sha(evaluation_path),
                checkpoint=ev['checkpoint'], eval_seed=ev['eval_seed'],
                counts=counts, ignored_aggregate_rows=ignored, metrics=metrics,
                success_rate=success, success_definition='per R scene: NC == 1 AND DAC == 1',
                headers=headers, row=row, units='ADE/FDE metres; other numeric columns fractions [0,1]')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--evaluation-manifest',type=Path,required=True)
    p.add_argument('--model-name',required=True);p.add_argument('--note',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=build(a.protocol,a.evaluation_manifest,a.model_name,a.note)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents replacing an earlier accepted table by accident.
    with a.output.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print('\t'.join(result['headers']));print('\t'.join(result['row']))

if __name__=='__main__': main()
