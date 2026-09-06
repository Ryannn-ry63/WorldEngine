"""Read-only independent review from audited JSON/CSV; never load pickle/torch."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-cfpi-v1')
SOURCE = ROOT / 'experiments/diffusiondrive/selector_cfpi_v1/runs/cfpi_pilot_20260905_r2'
CV = SOURCE.parent / 'cfpi_pilot_20260905_r2_cv1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


audit = read(CV / 'cache/pilot64_cache_audit.json')
for entry in audit['arm_collection_audits']:
    assert sha(entry['path']) == entry['sha256']
    assert sha(entry['metrics_csv']) == entry['metrics_csv_sha256']
targets = read(SOURCE / 'targets/target_manifest.json')
rows = targets['targets']
n = len(rows)
ids = [row['scene_id'] for row in rows]
incumbent = np.array([row['policy_index'] for row in rows])
solved = np.array([row['baseline_success'] for row in rows])
assert len(set(ids)) == n == 64
fields = ['score', 'no_at_fault_collisions', 'drivable_area_compliance',
          'ego_progress', 'time_to_collision_within_bound', 'comfort',
          'driving_direction_compliance']
values = np.zeros((n, 20, len(fields)), np.float32)
for entry in audit['arm_collection_audits']:
    with open(entry['metrics_csv']) as stream:
        data = {row['token']: row for row in csv.DictReader(stream) if row['token'] != 'overall_average'}
    assert set(data) == set(ids)
    for index, scene in enumerate(ids):
        values[index, entry['candidate_index']] = [float(data[scene][key]) for key in fields]
q = values[:, :, 0]
baseline = q[np.arange(n), incumbent]
success = (values[:, :, 1] == 1) & (values[:, :, 2] == 1)
assert np.array_equal(success[np.arange(n), incumbent], solved)
assert np.max(np.abs(baseline - np.array([row['baseline_score'] for row in rows]))) < 1e-3
logs = sorted({row['origin_log'] for row in rows})
log_indices = [np.array([i for i, row in enumerate(rows) if row['origin_log'] == log]) for log in logs]


def interval(gain):
    grouped = np.array([gain[index].mean() for index in log_indices])
    rng = np.random.default_rng(20260905)
    means = grouped[rng.integers(len(grouped), size=(10000, len(grouped)))].mean(1)
    return dict(scene_mean=float(gain.mean()), log_mean=float(grouped.mean()),
                log_bootstrap_95=np.quantile(means, [.025, .975]).tolist())


reports = {}
hash_count = 0
for path in sorted((CV / 'cv').glob('*/report.json')):
    report = read(path)
    assert report['status'] == 'PASS'
    assert report['train_scene_count'] == 48 and report['heldout_scene_count'] == 16
    assert report['incumbent_recompute_parity']['argmax_mismatch_count'] == 0
    assert report['provenance']['cache_sha256'] == audit['cache_file_sha256']
    for checkpoint in report['checkpoints']:
        assert sha(checkpoint['checkpoint']) == checkpoint['checkpoint_sha256']
        hash_count += 1
    reports[(report['method'], report['learning_rate'], report['fold'], report['seed'])] = report
assert len(reports) == 144 and hash_count == 432
summary = read(CV / 'pilot_report.json')
gains, actions, details = {}, {}, {}
for config in summary['configurations']:
    method, lr, step = config['method'], config['learning_rate'], config['step']
    key = (method, lr, step)
    predictions = []
    for seed in range(3):
        prediction = {}
        for fold in range(4):
            report = reports[(method, lr, fold, seed)]
            checkpoint = next(c for c in report['checkpoints'] if c['step'] == step)
            assert set(checkpoint['predictions']) == {row['scene_id'] for row in rows if row['fold'] == fold}
            prediction.update(checkpoint['predictions'])
        predictions.append([prediction[scene] for scene in ids])
    action = np.array(predictions)
    gain = q[np.arange(n), action] - baseline
    average = gain.mean(0).astype(float)
    assert abs(average.mean() - config['mean_gain']) < 1e-9
    gains[key], actions[key] = average, action
    chosen_success = success[np.arange(n), action]
    strata = {}
    for dimension in ['scenario_family', 'time_stratum', 'outcome_stratum']:
        strata[dimension] = {
            value: float(average[[i for i, row in enumerate(rows) if row[dimension] == value]].mean())
            for value in sorted({row[dimension] for row in rows})
        }
    seed_safety = []
    for seed in range(3):
        seed_safety.append(dict(
            changes=int((action[seed] != incumbent).sum()),
            beneficial=int((gain[seed] > .001).sum()), harmful=int((gain[seed] < -.001).sum()),
            rescued_failed=int(chosen_success[seed, ~solved].sum()),
            broken_solved=int((~chosen_success[seed, solved]).sum()),
            success_rate=float(chosen_success[seed].mean())))
    leave_one = [float(average[[i for i, row in enumerate(rows) if row['origin_log'] != log]].mean()) for log in logs]
    details[key] = dict(
        method=method, lr=lr, step=step, mean_gain=float(average.mean()),
        solved_gain=float(average[solved].mean()), failed_gain=float(average[~solved].mean()),
        seed_safety=seed_safety, strata=strata,
        components={field: float((values[np.arange(n), action, j] - values[np.arange(n), incumbent, j]).mean())
                    for j, field in enumerate(fields)},
        leave_one_log_out_gain_range=[min(leave_one), max(leave_one)])
gap = q.max(1) - baseline
local = np.array([row['candidate_rewards'] for row in rows], np.float32)
local_gain = q[np.arange(n), local.argmax(1)] - baseline
out = dict(
    integrity=dict(reports=144, checkpoint_hashes=hash_count, arm_metric_hashes=20,
                   all_36_report_gains_reproduced=True),
    pilot=dict(scenes=n, logs=len(logs), solved=int(solved.sum()),
               mean_incumbent=float(baseline.mean()), mean_oracle=float(q.max(1).mean()),
               mean_oracle_gain=float(gap.mean()), improvable_above_002=int((gap > .02).sum()),
               failed_any_safe_branch=int(success[~solved].any(1).sum()),
               strata_counts=dict(collections.Counter('/'.join(str(row[k]) for k in
                   ['scenario_family', 'outcome_stratum', 'time_stratum']) for row in rows))),
    local_argmax=dict(mean_gain=float(local_gain.mean()), harmful=int((local_gain < -.001).sum()),
                      beneficial=int((local_gain > .001).sum())), configs=[])
chosen_keys = [('q_grpo_t1', 1e-4, 500), ('q_grpo_t5', 3e-5, 500), ('q_mse', 1e-4, 50),
               ('q_ce_d0', 3e-5, 500), ('local_grpo_t1', 1e-4, 500), ('local_grpo_t1', 1e-4, 50)]
out['configs'] = [details[key] for key in chosen_keys]
out['paired_exploratory'] = {}
for label, a, b in [('T1_vs_matched_local', chosen_keys[0], chosen_keys[4]),
                    ('T5_vs_T1', chosen_keys[1], chosen_keys[0]), ('MSE_vs_T1', chosen_keys[2], chosen_keys[0])]:
    out['paired_exploratory'][label] = interval(gains[a] - gains[b])
out['matched_QT1_local'] = [dict(lr=lr, step=step, **interval(
    gains[('q_grpo_t1', lr, step)] - gains[('local_grpo_t1', lr, step)]))
    for lr in [3e-5, 1e-4] for step in [50, 150, 500]]
ledger = read(CV / 'decision_ledger.json')
out['cv_seconds'] = (datetime.datetime.fromisoformat(ledger['events'][-1]['time']) -
                     datetime.datetime.fromisoformat(ledger['events'][0]['time'])).total_seconds()
out['gpu_hours_total'] = sum(read(SOURCE.parent / name / 'decision_ledger.json')['gpu_hours_used']
    for name in ['cfpi_pilot_20260905', 'cfpi_pilot_20260905_r2', 'cfpi_pilot_20260905_r2_cv1'])
print(json.dumps(out, ensure_ascii=False, indent=2))
