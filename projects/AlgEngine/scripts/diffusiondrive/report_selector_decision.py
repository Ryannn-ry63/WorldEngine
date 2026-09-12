"""All-arm/all-seed development reporting; no winner replacement or confirmation."""
import csv
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_decision_common as x
import selector_decision_data as data
from report_selector_cfpi_deployment import checked_collection, differences
from selector_rare_common import delta, cluster_interval


def name(cohort, mode, policy):
    return f'eval_{cohort}_{mode}_{policy}'


def average(values):
    if not values or any(set(v) != set(values[0]) for v in values):
        raise RuntimeError('Seed coverage differs')
    return {s: {k: float(np.mean([v[s][k] for v in values])) for k in d.METRICS} for s in values[0]}


def seed_summary(values, reference, rows):
    """Rescue/broken are per-seed binary events, not rounded seed-mean success."""
    per_seed = [differences(v,reference,rows) for v in values]
    merged = differences(average(values),reference,rows)
    return dict(mean=merged['mean'], delta=merged['delta'],
                mean_rescued_failed=float(np.mean([v['rescued_failed'] for v in per_seed])),
                mean_broken_solved=float(np.mean([v['broken_solved'] for v in per_seed])))


def summarize_feedback(run, arm, seed):
    generation = 2 if arm in ('P2', 'P3') else 0
    rows, entry = data.load_rows(run, arm, seed, generation)
    model, _ = m.load_selector(d.verify(data.model_entry(run, f'{arm}_seed{seed}')))
    z = data.scores(model, rows)
    result = {}
    for mode in x.MODES:
        ids = [i for i,r in enumerate(rows) if r['fresh'] and r['mode'] == mode]
        comparisons, correct, winners, unsupported, clear_states = 0, 0, 0, 0, 0
        for i in ids:
            row = rows[i]
            action = int(z[i].argmax())
            prefs = x.preferences(row)
            comparisons += len(prefs)
            correct += sum(z[i,a] > z[i,b] for a,b,_ in prefs)
            unsupported += action not in row['candidate_indices']
            # One unambiguous winner only if it dominates EVERY other evaluated action.
            global_winners = [a for a in row['candidate_indices']
                              if all(a == b or any(w == a and l == b for w,l,_ in prefs)
                                     for b in row['candidate_indices'])]
            if global_winners:
                clear_states += 1
                winners += action in global_winners
        result[mode] = dict(states=len(ids), informative_comparisons=comparisons, correct_comparisons=int(correct),
                            decisive_states=clear_states, full_winner_selected=int(winners),
                            unsupported_argmax=int(unsupported))
    result['logical_queries'] = (sum(d.read(run/'queries'/f'{arm}_seed{seed}'/f'g{g}'/'requests.json')['logical_queries']
                                      for g in (1,2)) if generation else 0)
    return dict(data=entry, metrics=result)


def report(run):
    import torch
    torch.set_num_threads(4)
    inputs = data.checked_inputs(run)
    table, missing, artifacts = {}, [], [d.artifact(run/'decision_inputs.json')]
    for cohort, mode, policy in x.evaluation_conditions():
        path = run/'collections'/name(cohort, mode, policy)/'deployment_collection.json'
        if not path.exists():
            missing.append(path.parent.name)
            continue
        entry = d.artifact(path)
        checked = checked_collection(entry)
        if checked is None:
            missing.append(path.parent.name)
            continue
        condition = checked['collection']
        expected = dict(cohort=cohort, react_type=mode, policy=policy, research_method=x.METHOD,
                        role='evaluation', continuation_policy=None)
        if (any(condition.get(k) != v for k,v in expected.items())
                or condition['run_contract'] != d.artifact(run/'run_contract.json')
                or condition['inputs'] != d.artifact(run/'decision_inputs.json')
                or set(checked['metrics']) != {r['scene_id'] for r in inputs['cohorts'][cohort]['rows']}):
            raise RuntimeError('Evaluation identity/coverage changed')
        routing = d.verified_read(condition['routing'])
        if (set(routing['models']) != {policy} or routing['models'][policy] != data.model_entry(run, policy)
                or any('feedback_target' in r for r in routing['routes'])):
            raise RuntimeError('Evaluation used wrong model or training intervention')
        table[(cohort,mode,policy)] = checked['metrics']
        artifacts.extend((entry, checked['audit']))
    if missing:
        result = dict(status='INCOMPLETE', decision='NOT_AUTHORIZED', missing=missing,
                      complete_collections=len(table), expected_collections=len(x.evaluation_conditions()),
                      confirmation_authorized=False)
        c.atomic_json(run/'pilot_progress.json', result)
        return result
    effects, intervals, summaries, feedback, paired = {}, {}, {}, {}, []
    for mode in x.MODES:
        summaries[mode], intervals[mode] = {}, {}
        means = {arm: average([table[('development',mode,f'{arm}_seed{s}')] for s in x.SEEDS])
                 for arm in x.ARMS}
        scalar, gate = (table[('development',mode,p)] for p in ('scalar_v3','gate_v3'))
        common_scalar = table[('common',mode,'scalar_v3')]
        rare_rows, common_rows = (inputs['cohorts'][k]['rows'] for k in ('development','common'))
        reference_sets = dict(scalar=scalar, gate=gate, **{a:means[a] for a in ('P0','P1','P2')})
        effects[mode] = {k:delta(means['P3'],v) for k,v in reference_sets.items()}
        effects[mode]['common'] = delta(average([table[('common',mode,f'P3_seed{s}')] for s in x.SEEDS]), common_scalar)
        effects[mode]['seed_score_gains'] = [delta(table[('development',mode,f'P3_seed{s}')],scalar)['score'] for s in x.SEEDS]
        intervals[mode] = {key: {metric:cluster_interval(
            {s: means['P3'][s][metric]-reference[s][metric] for s in reference}, rare_rows)
            for metric in ('score','success','no_at_fault_collisions','drivable_area_compliance')}
            for key,reference in reference_sets.items()}
        common_mean = average([table[('common',mode,f'P3_seed{s}')] for s in x.SEEDS])
        intervals[mode]['common'] = {metric:cluster_interval(
            {s:common_mean[s][metric]-common_scalar[s][metric] for s in common_scalar}, common_rows)
            for metric in ('score','success','no_at_fault_collisions','drivable_area_compliance')}
        strata = dict(failed=[r for r in rare_rows if scalar[r['scene_id']]['success'] == 0],
                      safe_slow=[r for r in rare_rows if scalar[r['scene_id']]['success'] == 1
                                 and scalar[r['scene_id']]['ego_progress'] < .5],
                      safe_other=[r for r in rare_rows if scalar[r['scene_id']]['success'] == 1
                                  and scalar[r['scene_id']]['ego_progress'] >= .5])
        for arm in x.ARMS:
            arm_seeds = [table[('development',mode,f'{arm}_seed{s}')] for s in x.SEEDS]
            summaries[mode][arm] = dict(mean=seed_summary(arm_seeds, scalar, rare_rows),
                seeds=[dict(seed=seed,
                    rare=differences(table[('development',mode,f'{arm}_seed{seed}')],scalar,rare_rows),
                    common=differences(table[('common',mode,f'{arm}_seed{seed}')],common_scalar,common_rows))
                    for seed in x.SEEDS],
                strata={k:dict(scenes=len(rows), mean=seed_summary(arm_seeds,scalar,rows) if rows else None,
                    seeds=[differences(table[('development',mode,f'{arm}_seed{s}')],scalar,rows) for s in x.SEEDS]
                    if rows else []) for k,rows in strata.items()})
            for seed in x.SEEDS:
                for cohort in ('development','common'):
                    reference = table[(cohort,mode,'scalar_v3')]
                    for row in inputs['cohorts'][cohort]['rows']:
                        scene = row['scene_id']
                        actual = table[(cohort,mode,f'{arm}_seed{seed}')][scene]
                        paired.append(dict(cohort=cohort, mode=mode, arm=arm, seed=seed,
                            scene_id=scene, origin_log=row['origin_log'],
                            **{k:actual[k] for k in d.METRICS},
                            **{'scalar_'+k:reference[scene][k] for k in d.METRICS}))
    for arm in x.ARMS:
        feedback[arm] = [summarize_feedback(run,arm,seed) for seed in x.SEEDS]
    gate = x.point_gate(effects)
    support = {}
    for mode in x.MODES:
        for key in ('scalar','P0','P1','P2'):
            support[f'{mode}_superiority_{key}'] = intervals[mode][key]['score']['scene_weighted_cluster_95'][0] > 0
        support[f'{mode}_gate_noninferiority'] = intervals[mode]['gate']['score']['scene_weighted_cluster_95'][0] >= -.02
        for metric in ('success','no_at_fault_collisions','drivable_area_compliance'):
            support[f'{mode}_rare_{metric}'] = intervals[mode]['scalar'][metric]['scene_weighted_cluster_95'][0] >= 0
        for metric in ('score','success','no_at_fault_collisions','drivable_area_compliance'):
            support[f'{mode}_common_{metric}'] = intervals[mode]['common'][metric]['scene_weighted_cluster_95'][0] >= -.01
    decision = ('STOP_EFFECT_FAIL' if not gate['passed'] else
                'REQUIRES_SEPARATE_CONFIRMATION_PLAN' if all(support.values()) else 'INSUFFICIENT_EVIDENCE')
    target = run/'pilot_paired_scenes.csv'
    with target.with_suffix('.tmp').open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    target.with_suffix('.tmp').replace(target)
    result = dict(status='PASS', method=x.METHOD, decision=decision, gate=gate, interval_support=support,
                  effects=effects, summaries=summaries, intervals=intervals, feedback=feedback,
                  paired_scenes=d.artifact(target), artifacts=artifacts, exposure=x.EXPOSURE,
                  reference_means={f'{cohort}_{mode}_{policy}':
                      {k:float(np.mean([v[k] for v in table[(cohort,mode,policy)].values()])) for k in d.METRICS}
                      for cohort in ('development','common') for mode in x.MODES for policy in ('scalar_v3','gate_v3')},
                  confirmation_authorized=False, optimization_seeds=list(x.SEEDS),
                  independent_end_to_end_acquisition_replicates=False, safety_guarantee=False)
    c.locked_json(run/'pilot_report.json', result)
    return result
