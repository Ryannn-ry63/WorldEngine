"""Frozen v2 feedback experiment. These constants are not tuning knobs."""
from collections import Counter
import hashlib
import math
import numpy as np
import cfpi_common as c

METHOD = 'selector_feedback_repair_v2'
MODES = ('NR', 'R')
ARMS = ('S', 'U', 'T')
LIMITS = dict(feedback=24., train=4., eval=30., buffer=6.)
TRAIN = dict(steps_per_round=500, lr=1e-4, weight_decay=1e-4,
             dense_batch=16, feedback_batch=16, teacher_kl=.01, grad_clip=10., temperature=1.)
STRATA = (('rare_failed',16), ('rare_slow',32), ('rare_other',48), ('common_retention',32))
EXPOSURE = 'Legacy-exposed development; NOT unseen or blind. No confirmation authorized.'
FINGERPRINT_FIELDS = ('candidate_features','candidate_trajectories_8','route_bev_features',
                      'status_tokens','ego_queries','agents_queries','reference_logits')


def budget_limits(total):
    """Explicit resource allowance only; no change to the frozen experiment."""
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Cumulative GPU budget must be finite and positive')
    return {phase: total * (amount / sum(LIMITS.values())) for phase,amount in LIMITS.items()}


def gpu_devices(env, count):
    """Keep scheduler-provided IDs; never silently take GPUs outside the map."""
    if count not in (4,8):
        raise ValueError('Feedback runner supports four or eight H100 GPUs')
    raw = env.get('CUDA_VISIBLE_DEVICES',','.join(map(str,range(count))))
    devices = [x.strip() for x in raw.split(',')]
    if (len(devices)!=count or len(set(devices))!=count
            or any(not x or x=='-1' for x in devices)):
        raise ValueError(f'CUDA_VISIBLE_DEVICES must contain exactly {count} distinct GPU IDs; got {raw!r}')
    return devices


def normalized_runtime_samples(samples):
    """Convert measured allocation time to the legacy eight-GPU forecast basis."""
    result = []
    for sample in samples:
        gpus = sample.get('gpu_count',8)  # Historical deployment runs used eight GPUs.
        seconds = sample['seconds']
        if gpus not in (4,8) or not math.isfinite(seconds) or seconds<0 or sample['scenes']<=0:
            raise ValueError('Invalid allocation-aware runtime sample')
        result.append(dict(sample,seconds=seconds*gpus/8))
    return result


def ordered(items, namespace):
    return sorted(items, key=lambda x:(hashlib.sha256((namespace+':'+x).encode()).hexdigest(),x))


def safe(row):
    return row['no_at_fault_collisions'] == 1 and row['drivable_area_compliance'] == 1


def source_selection(pools, metrics, excluded):
    """Fixed baseline-R strata, global max two scenes/log; never read new outcomes."""
    choices = {}
    for family, scenes in pools.items():
        for scene in scenes:
            if scene not in metrics or c.scene_origin_log(scene) in excluded:
                continue
            v = metrics[scene]
            if family == 'rare_union':
                kind = 'rare_failed' if not safe(v) else ('rare_slow' if v['ego_progress'] < .5 else 'rare_other')
            elif family == 'matched_common' and safe(v) and v['ego_progress'] >= .5:
                kind = 'common_retention'
            else:
                continue
            if scene in choices:
                raise RuntimeError('Source families overlap')
            choices[scene] = kind
    selected, used = [], Counter()
    for kind, quota in STRATA:
        rows = []
        for scene in ordered([s for s,k in choices.items() if k==kind], 'rare-feedback-v2-source'):
            log = c.scene_origin_log(scene)
            if used[log] < 2:
                rows.append(dict(scene_id=scene, origin_log=log, origin_token=c.scene_token(scene),
                                 stratum=kind, family='common' if kind=='common_retention' else 'rare'))
                used[log] += 1
            if len(rows) == quota:
                break
        if len(rows) != quota:
            raise RuntimeError(f'Source capacity {kind}: {len(rows)} < {quota}; no quota relaxation')
        selected.extend(rows)
    return selected


def second_selection(rows, base, current, arm, mode):
    if arm not in ARMS or mode not in MODES:
        raise ValueError('Unregistered feedback condition')
    selected, accounting = [], {}
    for family, quota in (('rare',24),('common',8)):
        pool = {r['scene_id'] for r in rows if r['family']==family}
        if not pool <= base.keys() or not pool <= current.keys():
            raise RuntimeError('Incomplete paired training outcomes')
        if arm != 'T':
            chosen = [(s,'uniform') for s in ordered(pool,f'feedback-v2-uniform-{mode}')[:quota]]
        else:
            broken = {s for s in pool if (safe(base[s]) and not safe(current[s]))
                      or base[s]['score']-current[s]['score'] >= .05-1e-12}
            unresolved = {s for s in pool-broken if not safe(current[s]) or current[s]['ego_progress'] < .5}
            counts = [('unresolved',12,unresolved),('regression',8,broken),('uniform',4,pool)] if family=='rare' else [
                ('regression',4,broken),('uniform',4,pool)]
            chosen, used = [], set()
            for kind,n,eligible in counts:
                take = ordered(eligible-used,f'feedback-v2-target-{mode}-{kind}')[:n]
                chosen.extend((s,kind) for s in take)
                used.update(take)
            fill = ordered(pool-used,f'feedback-v2-target-{mode}-fallback')[:quota-len(chosen)]
            chosen.extend((s,'fallback_uniform') for s in fill)
        if len(chosen)!=quota or len({s for s,_ in chosen})!=quota:
            raise RuntimeError('Feedback selection capacity changed')
        selected.extend(dict(scene_id=s, reason=k, family=family) for s,k in chosen)
        accounting[family] = dict(Counter(k for _,k in chosen))
    return selected, accounting


def target_step(scene, metrics, sidecars, namespace, event=False):
    steps = sorted(sidecars)
    if steps != list(c.DECISION_STEPS):
        raise RuntimeError('Target requires all eight actually executed decisions')
    if not safe(metrics):
        first = metrics.get('first_violation_step')
        if first is None or not np.isfinite(first):
            raise RuntimeError('Failed scene is missing first violation timing')
        legal = [k for k in steps if k < first]
        if not legal:
            raise RuntimeError('No legal pre-violation decision; do not silently drop scene')
        if event:
            return legal[-1]
        steps = legal
    return int(ordered([str(k) for k in steps], namespace+':'+scene)[0])


def candidate_pair(v3_scores, challenger_scores, gate_scores):
    arrays = [c.checked_array(x,(20,),'proposal scores') for x in (v3_scores,challenger_scores,gate_scores)]
    incumbent, other = int(arrays[0].argmax()), int(arrays[1].argmax())
    if other == incumbent:
        other = next(int(i) for i in np.argsort(-arrays[2],kind='stable') if i!=incumbent)
    return [incumbent,other]


def pareto_pair(a,b):
    """Return (winner slot, loser slot, raw PDM gap), or no preference."""
    for v in (a,b):
        if any(not np.isfinite(v[k]) or not 0<=v[k]<=1 for k in
               ('score','no_at_fault_collisions','drivable_area_compliance','success')):
            raise RuntimeError('Invalid pair labels')
    for winner,loser in ((0,1),(1,0)):
        high, low = (a,b) if winner==0 else (b,a)
        if high['score']-low['score'] >= .01-1e-12 and all(high[k]>=low[k] for k in
                ('no_at_fault_collisions','drivable_area_compliance','success')):
            return winner,loser,high['score']-low['score']
    return None


def fingerprint(context):
    digest = hashlib.sha256()
    for key in FINGERPRINT_FIELDS:
        value = np.asarray(context[key],dtype='<f4')
        if not np.isfinite(value).all():
            raise RuntimeError('Nonfinite visible context')
        digest.update(key.encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def shortlist_gate(effects):
    checks = {}
    for mode in MODES:
        v = effects[mode]
        checks[mode] = (v['scalar']['score'] >= .01-1e-12 and v['gate']['score'] >= -.02-1e-12
                        and v['scalar']['success'] >= -1e-12
                        and all(v[k]['score'] >= .005-1e-12 for k in ('S','U')))
    return dict(passed=all(checks.values()),checks=checks)


def final_gate(effects):
    checks = {}
    for mode in MODES:
        v = effects[mode]
        mean = lambda k,m: float(np.mean([x[m] for x in v[k]]))
        if any(len(v[k])!=3 for k in ('scalar','gate','common')):
            raise RuntimeError('Final gate requires all three seeds')
        checks[mode] = (mean('scalar','score') >= .02-1e-12 and mean('gate','score') >= -.01-1e-12
            and mean('gate','success') >= .01-1e-12
            and all(mean('scalar',k)>=-1e-12 for k in ('success','no_at_fault_collisions','drivable_area_compliance','ego_progress'))
            and sum(x['score']>0 for x in v['scalar'])>=2
            and all(mean('common',k)>=-.01-1e-12 for k in ('score','success','no_at_fault_collisions','drivable_area_compliance')))
    return dict(passed=all(checks.values()),checks=checks)
