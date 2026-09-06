"""Whole-cohort paired outcomes, fixed winner and log-cluster uncertainty."""
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_rare_common as r
from report_selector_cfpi_deployment import checked_collection, differences


def average(data):
    if len(data)!=3 or any(set(x)!=set(data[0]) for x in data):
        raise RuntimeError('Three complete seed cohorts required')
    return {s:{k:float(np.mean([x[s][k] for x in data])) for k in d.METRICS} for s in data[0]}


def summarize(data, scalar, gate, rows):
    avg = average(data)
    result = dict(mean={k:float(np.mean([v[k] for v in avg.values()])) for k in d.METRICS},
                  per_seed_vs_scalar=[r.delta(x,scalar) for x in data],per_seed_vs_gate=[r.delta(x,gate) for x in data],
                  intervals={name:r.cluster_interval({s:avg[s]['score']-ref[s]['score'] for s in avg},rows)
                             for name,ref in [('scalar',scalar),('gate',gate)]},strata={})
    for name,ids in [('baseline_failed',[s for s in scalar if scalar[s]['success']==0]),
                     ('baseline_solved',[s for s in scalar if scalar[s]['success']==1]),
                     ('baseline_low_ep',[s for s in scalar if scalar[s]['ego_progress']<.2])]:
        subset = [x for x in rows if x['scene_id'] in ids]
        if ids:
            per_seed = [differences(x,scalar,subset) for x in data]
            result['strata'][name] = dict(scenes=len(ids),delta=differences(avg,scalar,subset)['delta'],
                                         per_seed=per_seed,
                                         mean_rescued_failed=float(np.mean([x['rescued_failed'] for x in per_seed])),
                                         mean_broken_solved=float(np.mean([x['broken_solved'] for x in per_seed])))
        else:
            result['strata'][name] = dict(scenes=0)
    return result


def report(run, phase):
    inventory = run/f'{phase}_collections.json'
    if not inventory.exists():
        return dict(status='INCOMPLETE',phase=phase,missing=[str(inventory)])
    inputs = d.read(run/'rare_inputs.json')
    if phase=='confirm':
        winner = d.read(run/'winner.json')
        d.verify(winner['screen_report'])
        for model in winner['models'].values():
            d.verify(model)
    from selector_rare_common import expected_conditions
    expected = set(expected_conditions(phase, winner['method'] if phase=='confirm' else None))
    inventory_keys = []
    for entry in d.read(inventory)['collections']:
        contract = d.verified_read(entry)
        inventory_keys.append((contract['cohort'],contract['policy']))
    if len(inventory_keys)!=len(expected) or set(inventory_keys)!=expected:
        raise RuntimeError('Incomplete/extra frozen condition inventory')
    data, audits, missing = {},[],[]
    for entry in d.read(inventory)['collections']:
        checked = checked_collection(entry)
        if checked is None:
            missing.append(entry['path'])
            continue
        contract = checked['collection']
        key = (contract['cohort'],contract['policy'])
        if key in data:
            raise RuntimeError('Duplicate result')
        data[key] = checked['metrics']
        audits.append(checked['audit'])
    if missing:
        out = dict(status='INCOMPLETE',phase=phase,missing=missing,engineering_pass=False)
        c.atomic_json(run/f'{phase}_progress.json',out)
        return out
    result = dict(status='PASS',engineering_pass=True,phase=phase,exposure=r.EXPOSURE,audits=audits)
    if phase=='bridge':
        result['cached_diagnostic'] = d.artifact(run/'bridge_cached_diagnostic.json')
        old = Path(inputs['artifacts']['old_inputs']['path']).parent
        rows = inputs['cohorts']['bridge']['rows']
        result['comparisons'] = {}
        for policy in r.BRIDGE:
            old_result = checked_collection(d.artifact(old/'collections'/f'phase_a_{policy}'/'deployment_collection.json'))
            if old_result is None:
                raise RuntimeError('Missing original target-start comparison')
            a,b = data['bridge',policy],old_result['metrics']
            result['comparisons'][policy] = dict(delta=r.delta(a,b),interval=r.cluster_interval(
                {s:a[s]['score']-b[s]['score'] for s in a},rows),reference_audit=old_result['audit'])
        result.update(decision='DIAGNOSTIC_ONLY',posthoc_seed_selection=True,efficacy_claim=False)
    else:
        # Historical common data are independently audit/hash checked every report.
        ids = {x['scene_id'] for x in inputs['cohorts']['common']['rows']}
        for policy,entry in inputs['common_reuse'].items():
            d.verify(entry['audit'])
            checked = checked_collection(entry['collection'])
            data['common',policy] = {s:checked['metrics'][s] for s in ids}
        cohort = 'development' if phase=='screen' else 'confirmation'
        scalar,gate = data[cohort,'scalar_v3'],data[cohort,'gate_v3']
        common_scalar = data['common','scalar_v3']
        rows = inputs['cohorts'][cohort]['rows']
        result['baselines'] = {p:{k:float(np.mean([v[k] for v in data[cohort,p].values()])) for k in d.METRICS}
                               for p in ('scalar_v3','gate_v3')}
        methods = r.POLICIES if phase=='screen' else (d.read(run/'winner.json')['method'],)
        result['methods'] = {}
        for method in methods:
            metrics = [data[cohort,f'{method}_seed{s}'] for s in r.SEEDS]
            summary = summarize(metrics,scalar,gate,rows)
            if phase=='screen':
                common = [r.delta(data['common',f'{method}_seed{s}'],common_scalar) for s in r.SEEDS]
            else:
                common = d.read(run/'screen_report.json')['methods'][method]['common_per_seed_vs_scalar']
            summary['common_per_seed_vs_scalar'] = common
            summary['effect_gate'] = r.effect_gate(summary['per_seed_vs_scalar'],summary['per_seed_vs_gate'],common)
            train_reports = [d.read(run/'train'/f'{method}_seed{s}'/'report.json') for s in r.SEEDS]
            summary['trainable_parameters'] = train_reports[0]['trainable_parameters']
            result['methods'][method] = summary
        if phase=='screen':
            result['historical_references'] = {m:summarize([data[cohort,f'{m}_seed{s}'] for s in r.SEEDS],scalar,gate,rows)
                                                for m in ('q_grpo_t1','q_mse')}
            passing = [m for m in methods if result['methods'][m]['effect_gate']['passed']]
            passing.sort(key=lambda m:(-result['methods'][m]['mean']['score'],-result['methods'][m]['mean']['success'],
                                        -result['methods'][m]['mean']['ego_progress'],result['methods'][m]['trainable_parameters'],m))
            result['winner'] = passing[0] if passing else None
            result['decision'] = 'PROCEED_FIXED_WINNER' if passing else 'STOP_NO_WINNER'
        else:
            summary = result['methods'][methods[0]]
            positive_ci = all(v['scene_weighted_cluster_95'][0]>0 for v in summary['intervals'].values())
            result['decision'] = ('FAIL' if not summary['effect_gate']['passed'] else
                                  'CONFIRMED_LEGACY_EXPOSED' if positive_ci else 'INSUFFICIENT_EVIDENCE')
            result['winner_replacement_authorized'] = False
    c.locked_json(run/f'{phase}_report.json',result)
    if phase=='screen' and result['winner']:
        winner = result['winner']
        models = {f'{winner}_seed{s}':d.read(run/'train'/f'{winner}_seed{s}'/'report.json')['selector'] for s in r.SEEDS}
        c.locked_json(run/'winner.json',dict(method=winner,models=models,screen_report=d.artifact(run/'screen_report.json'),
                                           refit_authorized=False,exposure=r.EXPOSURE))
    return result


if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--phase',choices=('bridge','screen','confirm'),required=True)
    a = p.parse_args()
    result = report(a.run_root,a.phase)
    print({k:result[k] for k in ('status','phase','decision') if k in result})
