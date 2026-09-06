"""Fixed rare-first protocol; engineering PASS is not effect success."""
from collections import Counter
import hashlib
import numpy as np
import selector_cfpi_deployment_common as d

METHOD = 'selector_rare_retention_v1'
POLICIES = ('local_anchor', 'q_anchor', 'q_full')
SEEDS = (0, 1, 2)
LIMITS = dict(bridge=6., screen=32., confirm=22., buffer=4.)
TRAIN = dict(steps=500, lr=1e-4, weight_decay=1e-4, correction_batch=16,
             replay_batch=16, temperature=1., teacher_kl=.01, grad_clip=10.)
BRIDGE = ('q_grpo_t1_seed0', 'q_grpo_t1_seed2', 'local_grpo_t1_seed2')
REFERENCES = ('scalar_v3', 'gate_v3', *[f'{m}_seed{s}' for m in ('q_grpo_t1','q_mse') for s in SEEDS])
EXPOSURE = 'Legacy-exposed benchmark; frozen-candidate confirmation, NOT unseen/blind test.'
RARE_SPLITS = {
    'development': (58,22,'6bdb15f638ea9843f0e69aa30a9ea732a1f9f3dd1351bdbe259ef180cb63a577'),
    'confirmation': (230,67,'60d37f4d6fef1dec93330b2c99060324513aea6b942f228edd6e1dedcf168db0'),
}


def expected_conditions(phase,winner=None):
    if phase=='bridge':
        return [('bridge',p) for p in BRIDGE]
    if phase=='screen':
        return ([('development',p) for p in REFERENCES]+
                [(cohort,f'{m}_seed{s}') for m in POLICIES for s in SEEDS for cohort in ('development','common')])
    if phase=='confirm' and winner in POLICIES:
        return [('confirmation',p) for p in ('scalar_v3','gate_v3',*[f'{winner}_seed{s}' for s in SEEDS])]
    raise RuntimeError('Invalid stage or unfrozen winner')


def choose_replay(metrics, excluded, origin, count=64):
    eligible = [s for s,v in metrics.items() if origin(s) not in excluded and v['success'] == 1 and v['ego_progress'] >= .2]
    ordered = sorted(eligible,key=lambda s:(hashlib.sha256(('rare-replay-v1:'+s).encode()).hexdigest(),s))
    selected, used = [], Counter()
    for s in ordered:
        if used[origin(s)] < 2:
            selected.append(s)
            used[origin(s)] += 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError('Insufficient replay capacity; no quota relaxation')
    return selected, dict(eligible_scenes=len(eligible), eligible_logs=len({origin(s) for s in eligible}),
                          selected_scenes=len(selected), selected_logs=len(used), maximum_per_log=2)


def cluster_interval(deltas, rows, samples=10000):
    """Resample logs; ratio of sums estimates scene-weighted mean."""
    if set(deltas) != {r['scene_id'] for r in rows} or len(deltas) != len(rows):
        raise RuntimeError('Incomplete/duplicate paired coverage')
    logs = sorted({r['origin_log'] for r in rows})
    groups = [np.asarray([deltas[r['scene_id']] for r in rows if r['origin_log']==log]) for log in logs]
    if not groups or not all(np.isfinite(g).all() for g in groups):
        raise RuntimeError('Empty/nonfinite statistic')
    sums, sizes = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    draws = np.random.default_rng(20260906).integers(len(logs),size=(samples,len(logs)))
    boot = sums[draws].sum(1)/sizes[draws].sum(1)
    equal = (sums/sizes)[draws].mean(1)
    return dict(scene_mean=float(sums.sum()/sizes.sum()), scene_weighted_cluster_95=np.quantile(boot,[.025,.975]).tolist(),
                equal_log_mean=float((sums/sizes).mean()), equal_log_cluster_95=np.quantile(equal,[.025,.975]).tolist(),
                logs=len(logs), scenes=len(rows), seed_average_first=True, exposure=EXPOSURE)


def effect_gate(rare_scalar, rare_gate, common_scalar):
    if any(len(x)!=3 for x in (rare_scalar,rare_gate,common_scalar)):
        raise RuntimeError('All three seeds required')
    if any(not np.isfinite(v) for group in (rare_scalar,rare_gate,common_scalar) for row in group for v in row.values()):
        raise RuntimeError('Nonfinite effects')
    mean = lambda rows,k: float(np.mean([r[k] for r in rows]))
    checks = dict(rare_vs_scalar=mean(rare_scalar,'score')>=.03, rare_vs_gate=mean(rare_gate,'score')>=.03,
                  positive_vs_scalar=sum(r['score']>0 for r in rare_scalar)>=2,
                  positive_vs_gate=sum(r['score']>0 for r in rare_gate)>=2,
                  rare_success=mean(rare_scalar,'success')>=0, rare_progress=mean(rare_scalar,'ego_progress')>=0)
    for k in ('score','success','no_at_fault_collisions','drivable_area_compliance'):
        checks['common_'+k] = mean(common_scalar,k)>=-.01
    return dict(passed=all(checks.values()),checks=checks)


def delta(a,b):
    if set(a)!=set(b) or not a:
        raise RuntimeError('Paired coverage mismatch')
    return {k:float(np.mean([a[s][k]-b[s][k] for s in a])) for k in d.METRICS}
