"""Pre-registered decision-feedback rules. No simulator/model dependencies."""
import hashlib
import itertools
import json
import math
import numpy as np
import selector_feedback_common as f

METHOD = 'selector_decision_feedback_v1'
ARMS = ('P0', 'P1', 'P2', 'P3')
SEEDS = (0, 1, 2)
QUERY_STEPS = (100, 300)
END_STEP = 500
MODES = f.MODES
TRAIN = dict(f.TRAIN, winner_nll=1., query_steps=list(QUERY_STEPS),
             max_additional_candidates_per_state=2)
EXPOSURE = ('Legacy-exposed development/common; conditional on seed0-collected '
            'states and fixed noise0; optimization and adaptive-query seeds, NOT independent acquisition.')
DIAGNOSIS = dict(min_dominated_logs=3, min_R_dominated_decisions=2,
                 continuation_per_mode=8, max_reversals=4)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def row_id(row):
    return digest({k: row[k] for k in ('scene_id', 'mode', 'decision_step',
                                      'source_fingerprint', 'continuation_policy')})


def candidate_bank_hash(row):
    return hashlib.sha256(np.asarray(row['candidate_trajectories_8'], dtype='<f4').tobytes()).hexdigest()


def check_row(row):
    ids, outcomes = row['candidate_indices'], row['branch_outcomes']
    if (len(ids) != len(outcomes) or not 2 <= len(ids) <= 4 or len(set(ids)) != len(ids)
            or any(type(i) is not int or not 0 <= i < 20 for i in ids)):
        raise RuntimeError('Invalid evaluated subset; no missing-label imputation')
    if row['mode'] not in MODES or row['decision_step'] not in range(4, 12):
        raise RuntimeError('Invalid feedback condition')
    for outcome in outcomes:
        f.pareto_pair(outcome, outcome)  # validates bounds/finiteness
        if outcome['success'] != int(f.safe(outcome)):
            raise RuntimeError('Strict NC/DAC success label differs')
    if row.get('row_id', row_id(row)) != row_id(row):
        raise RuntimeError('Feedback state/continuation identity drift')
    if row.get('candidate_bank_sha', candidate_bank_hash(row)) != candidate_bank_hash(row):
        raise RuntimeError('Candidate bank changed')


def preferences(row):
    check_row(row)
    found = []
    for a, b in itertools.combinations(range(len(row['candidate_indices'])), 2):
        preference = f.pareto_pair(row['branch_outcomes'][a], row['branch_outcomes'][b])
        if preference is not None:
            winner, loser, gap = preference
            slots = (a, b)
            found.append((row['candidate_indices'][slots[winner]],
                          row['candidate_indices'][slots[loser]], gap))
    return found


def feedback_loss(logits, rows, winner_nll=True):
    import torch
    import torch.nn.functional as F
    pair_terms, nll_terms = [], []
    logp = F.log_softmax(logits, dim=-1) if winner_nll else None
    for i, row in enumerate(rows):
        pairs = preferences(row)
        denominator = math.comb(len(row['candidate_indices']), 2)
        if pairs:
            pair_terms.append(sum(float(gap) * F.softplus(logits[i, loser] - logits[i, winner])
                                  for winner, loser, gap in pairs) / denominator)
            if winner_nll:
                nll_terms.append(sum(-float(gap) * logp[i, winner]
                                     for winner, _, gap in pairs) / denominator)
    zero = logits.sum() * 0.
    pair = torch.stack(pair_terms).sum() / len(rows) if pair_terms else zero
    nll = torch.stack(nll_terms).sum() / len(rows) if nll_terms else zero
    return pair, nll


def query_plan(rows, logits, arm, seed, step, active_requests=None):
    if arm not in ('P2', 'P3') or seed not in SEEDS or step not in QUERY_STEPS:
        raise RuntimeError('Unregistered query')
    fresh = {r['row_id']: r for r in rows if r['fresh']}
    if len(fresh) != 64:
        raise RuntimeError('Queries require the fixed 64 refreshed states')
    requests = []
    if arm == 'P3':
        if len(logits) != len(rows) or not np.isfinite(logits).all():
            raise RuntimeError('Invalid current policy scores')
        for row, z in zip(rows, logits):
            if row['fresh'] and int(np.argmax(z)) not in row['candidate_indices']:
                requests.append(dict(row_id=row['row_id'], candidate_index=int(np.argmax(z))))
    else:
        if active_requests is None:
            raise RuntimeError('Random control requires frozen active query schedule')
        if len({r['row_id'] for r in active_requests}) != len(active_requests):
            raise RuntimeError('Duplicate active query state')
        for request in active_requests:
            row = fresh[request['row_id']]
            unseen = [i for i in range(20) if i not in row['candidate_indices']]
            namespace = f'decision-feedback-random-{seed}-{step}-{row["row_id"]}'
            candidate = int(np.random.default_rng(int(digest(namespace)[:16], 16)).choice(unseen))
            requests.append(dict(row_id=row['row_id'], candidate_index=candidate))
    for request in requests:
        row = fresh[request['row_id']]
        if len(row['candidate_indices']) >= 4:
            raise RuntimeError('Candidate query ceiling reached')
        request.update(mode=row['mode'], scene_id=row['scene_id'],
                       candidate_bank_sha=row['candidate_bank_sha'],
                       source_fingerprint=row['source_fingerprint'],
                       continuation_policy=row['continuation_policy'])
    return requests


def diagnosis_gate(dominated, reversals, continuation_count):
    if continuation_count != 16:
        raise RuntimeError('Incomplete continuation diagnostic')
    checks = dict(dominated_logs=len({r['origin_log'] for r in dominated}) >= 3,
                  R_decisions=sum(r['mode'] == 'R' for r in dominated) >= 2,
                  continuation_stable=reversals <= 4)
    return dict(passed=all(checks.values()), checks=checks,
                decision='PROCEED_FIXED_PILOT' if all(checks.values()) else 'STOP_DIAGNOSIS_REVIEW')


def budget_limits(total):
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Positive finite cumulative budget required')
    return {k: total*v/128 for k, v in dict(diagnose=12., query=28., train=8., eval=68., buffer=12.).items()}


def evaluation_conditions():
    return [(cohort, mode, policy) for cohort in ('development', 'common') for mode in MODES
            for policy in ('scalar_v3', 'gate_v3', *[f'{arm}_seed{s}' for arm in ARMS for s in SEEDS])]


def point_gate(effects):
    checks = {}
    for mode in MODES:
        e = effects[mode]
        values = [v for key in ('scalar','gate','P0','P1','P2','common') for v in e[key].values()]
        if len(e['seed_score_gains']) != 3 or not all(np.isfinite(v) for v in values+e['seed_score_gains']):
            raise RuntimeError('All three finite paired seed effects are required')
        for key, threshold in dict(scalar=.01, gate=-.02, P0=.005, P1=.005, P2=.005).items():
            checks[f'{mode}_score_{key}'] = e[key]['score'] >= threshold-1e-12
        for key in ('success', 'no_at_fault_collisions', 'drivable_area_compliance'):
            checks[f'{mode}_rare_{key}'] = e['scalar'][key] >= -1e-12
        for key in ('score', 'success', 'no_at_fault_collisions', 'drivable_area_compliance'):
            checks[f'{mode}_common_{key}'] = e['common'][key] >= -.01-1e-12
        checks[f'{mode}_positive_seeds'] = sum(v > 0 for v in e['seed_score_gains']) >= 2
    return dict(passed=all(checks.values()), checks=checks)
