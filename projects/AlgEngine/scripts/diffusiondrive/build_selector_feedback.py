"""Freeze all targets before observing branch returns; no reward-based candidate proposal."""
import argparse
import csv
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import prepare_selector_feedback as prep
from report_selector_cfpi_deployment import checked_collection


def collection_data(run, name):
    entry = d.artifact(run/'collections'/name/'deployment_collection.json')
    checked = checked_collection(entry)
    if checked is None:
        raise RuntimeError('Collection incomplete: '+name)
    audit = d.verified_read(checked['audit'])
    records = {}
    for item in audit['artifacts']:
        if '/cfpi_deployment_records/' not in item['path']:
            continue
        row = d.verified_read(item)
        scene, step = row['scene_id'],row['state_step']+1
        if step in records.setdefault(scene,{}):
            raise RuntimeError('Duplicate source decision')
        records[scene][step] = row['sidecar']
    metrics = {s:dict(v) for s,v in checked['metrics'].items()}
    mode = checked['collection']['react_type']
    timing = {}
    for item in audit['artifacts']:
        if Path(item['path']).name != f'all_scenes_pdm_averages_{mode}.csv':
            continue
        with d.verify(item).open() as stream:
            for row in csv.DictReader(stream):
                if row['token']=='overall_average':
                    continue
                raw = row.get('first_violation_step','').strip()
                timing[row['token']] = None if raw.lower() in ('','nan','none') else int(float(raw))
    if set(timing)!=set(metrics) or set(records)!=set(metrics):
        raise RuntimeError('Source actions/timing/metrics coverage mismatch')
    for scene in metrics:
        metrics[scene]['first_violation_step'] = timing[scene]
        if set(records[scene])!=set(c.DECISION_STEPS):
            raise RuntimeError('Source lacks eight executed decisions')
    return dict(entry=entry,audit=checked['audit'],metrics=metrics,records=records)


def targets(run, round_name, arm='shared', device='cpu'):
    import torch
    from selector_cfpi_deployment_router import load_bank,selector_scores
    torch.set_num_threads(4)
    output = run/'feedback'/round_name/arm/'targets.json'
    if output.exists():
        result = d.read(output)
        for item in result['artifacts']:
            d.verify(item)
        return result
    if (round_name,arm) != ('round1','shared') and (round_name!='round2' or arm not in f.ARMS):
        raise RuntimeError('Unknown feedback round/arm')
    inputs = d.read(run/'feedback_inputs.json')
    rows = inputs['cohorts']['train']['rows']
    keys = ['scalar_v3','gate_v3']+(['shared_seed0'] if round_name=='round2' else [])
    entries = {k:prep.model_entry(run,k,inputs) for k in keys}
    bank = {k:load_bank(v).to(device) for k,v in entries.items()}
    all_targets, counts, artifacts = [], {}, [d.artifact(run/'feedback_inputs.json'),*entries.values()]
    for mode in f.MODES:
        base = collection_data(run,'baseline_'+mode)
        visit_name = 'baseline_'+mode if round_name=='round1' or arm=='S' else 'visit_'+mode
        visit = base if visit_name=='baseline_'+mode else collection_data(run,visit_name)
        artifacts.extend((base['entry'],base['audit'],visit['entry'],visit['audit']))
        if round_name=='round1':
            other = f.ordered([r['scene_id'] for r in rows if r['stratum']=='rare_other'],'feedback-v2-round1-other')[:16]
            chosen = [dict(scene_id=r['scene_id'],reason=r['stratum'],family=r['family']) for r in rows
                      if r['stratum']!='rare_other' or r['scene_id'] in other]
            counts[mode] = dict(total=len(chosen),fixed_round1=True)
        else:
            current = collection_data(run,'visit_'+mode)
            artifacts.extend((current['entry'],current['audit']))
            chosen, counts[mode] = f.second_selection(rows,base['metrics'],current['metrics'],arm,mode)
        for item in chosen:
            scene = item['scene_id']
            steps = visit['records'][scene]
            # S and U have identical uniform scene/timing namespaces. T event targets
            # use the last pre-violation decision only when an actual failure exists.
            event = round_name=='round1' or (
                arm=='T' and item['reason'] in ('unresolved','regression'))
            step = f.target_step(scene,visit['metrics'][scene],steps,
                                 f'feedback-v2-uniform-step-{mode}',event=event)
            context = c.load_pickle(d.verify(steps[step]))
            scores = {k:selector_scores(v,context,device,'residual') for k,v in bank.items()}
            challenger = 'gate_v3' if round_name=='round1' else 'shared_seed0'
            pair = f.candidate_pair(scores['scalar_v3'],scores[challenger],scores['gate_v3'])
            visiting = 'scalar_v3' if visit_name.startswith('baseline') else 'shared_seed0'
            natural = int(context['selected_index'])
            if natural != int(scores[visiting].argmax()):
                raise RuntimeError('Visiting bank/source action mismatch')
            prefix = {str(k):v for k,v in steps.items() if k<=step}
            artifacts.extend(prefix.values())
            all_targets.append(dict(item,mode=mode,decision_step=step,candidate_indices=pair,
                visiting_policy=visiting,continuation_policy=visiting,visiting_index=natural,
                source_collection=visit['entry'],source_audit=visit['audit'],
                context=steps[step],prefix=prefix,source_fingerprint=f.fingerprint(context)))
    if len(all_targets)!=(192 if round_name=='round1' else 64):
        raise RuntimeError('Fixed feedback query count drift')
    result = dict(status='PASS',method=f.METHOD,round=round_name,arm=arm,targets=all_targets,
                  selection_counts=counts,artifacts=artifacts,selection_reads_branch_returns=False)
    c.locked_json(output,result)
    return result


def collections(run, round_name, arm):
    path = run/'feedback'/round_name/arm/'collections.json'
    if path.exists():
        result = d.read(path)
        for e in result['collections']:
            d.verify(e)
        return result
    data = d.read(run/'feedback'/round_name/arm/'targets.json')
    entries = []
    for mode in f.MODES:
        subset = [x for x in data['targets'] if x['mode']==mode]
        for slot in ('sentinel',0,1):
            chosen = subset
            if slot=='sentinel':
                selected = set(f.ordered([x['scene_id'] for x in subset],f'feedback-v2-sentinel-{mode}')[:8])
                chosen = [x for x in subset if x['scene_id'] in selected]
            routes = {x['scene_id']:dict(x,forced_index=x['visiting_index'] if slot=='sentinel' else x['candidate_indices'][slot],
                                       sentinel=slot=='sentinel') for x in chosen}
            name = f'{round_name}_{arm}_{mode}_{slot}'
            policy = subset[0]['visiting_policy']
            entries.append(prep.collection(run,name,'train',mode,policy,list(routes),routes,policy))
    result = dict(status='PASS',round=round_name,arm=arm,collections=entries)
    c.locked_json(path,result)
    return result


def verify_sentinel(run, name):
    value = collection_data(run,name)
    collection = d.verified_read(value['entry'])
    routing = d.verified_read(collection['routing'])
    sources = {}
    for route in routing['routes']:
        target = route['feedback_target']
        if not target['sentinel'] or target['forced_index']!=target['visiting_index']:
            raise RuntimeError('Not an actual visiting-action sentinel')
        source_name = Path(target['source_collection']['path']).parent.name
        if source_name not in sources:
            sources[source_name] = collection_data(run,source_name)
        source = sources[source_name]
        scene = route['scene_id']
        if any(abs(value['metrics'][scene][k]-source['metrics'][scene][k])>c.OUTCOME_TOLERANCE for k in d.METRICS):
            raise RuntimeError('Visiting sentinel changed outcome')
        if value['metrics'][scene]['first_violation_step']!=source['metrics'][scene]['first_violation_step']:
            raise RuntimeError('Visiting sentinel changed event timing')
        for step in c.DECISION_STEPS:
            a = c.load_pickle(d.verify(value['records'][scene][step]))
            b = c.load_pickle(d.verify(source['records'][scene][step]))
            if int(a['selected_index'])!=int(b['selected_index']) or any(
                np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))>c.ARRAY_TOLERANCE for k in f.FINGERPRINT_FIELDS):
                raise RuntimeError('Sentinel changed full natural trajectory')
    c.atomic_json(Path(value['entry']['path']).parent/'sentinel_audit.json',
                  dict(status='PASS',collection=value['entry'],audit=value['audit'],visiting_action_checked=True))


def build_cache(run, round_name, arm):
    folder = run/'feedback'/round_name/arm
    if (folder/'cache_audit.json').exists():
        previous = d.read(folder/'cache_audit.json')
        d.verify(previous['cache'])
        for item in previous['artifacts']:
            d.verify(item)
        return previous
    target_entry = d.artifact(folder/'targets.json')
    data = d.verified_read(target_entry)
    rows, artifacts, preferences = [], [target_entry], 0
    for mode in f.MODES:
        verify_sentinel(run,f'{round_name}_{arm}_{mode}_sentinel')
        pair = [collection_data(run,f'{round_name}_{arm}_{mode}_{i}') for i in range(2)]
        for entry in pair:
            artifacts.extend((entry['entry'],entry['audit']))
        for target in [x for x in data['targets'] if x['mode']==mode]:
            scene = target['scene_id']
            for slot,entry in enumerate(pair):
                contract = d.verified_read(entry['entry'])
                routing = d.verified_read(contract['routing'])
                matching = [r for r in routing['routes'] if r['scene_id']==scene]
                if len(matching)!=1:
                    raise RuntimeError('Branch target membership drift')
                actual = matching[0]['feedback_target']
                if (actual['forced_index']!=target['candidate_indices'][slot]
                        or actual['source_fingerprint']!=target['source_fingerprint']
                        or actual['source_collection']!=target['source_collection']
                        or actual['mode']!=mode or actual['decision_step']!=target['decision_step']
                        or actual['sentinel']):
                    raise RuntimeError('Pair slot/target/state lineage mismatch')
            labels = [x['metrics'][scene] for x in pair]
            preference = f.pareto_pair(*labels)
            preferences += preference is not None
            context = c.load_pickle(d.verify(target['context']))
            row = {k:np.asarray(context[k],dtype=np.float32) for k in f.FINGERPRINT_FIELDS}
            row.update(scene_id=scene,origin_log=c.scene_origin_log(scene),mode=mode,
                       candidate_indices=target['candidate_indices'],branch_outcomes=labels,preference=preference,
                       source_fingerprint=target['source_fingerprint'],decision_step=target['decision_step'],
                       visiting_policy=target['visiting_policy'],continuation_policy=target['continuation_policy'])
            rows.append(row)
    result = dict(status='PASS',method=f.METHOD,rows=rows,round=round_name,arm=arm)
    path = folder/'cache.pkl'
    c.atomic_pickle(path,result)
    report = dict(status='PASS',cache=d.artifact(path),artifacts=artifacts,rows=len(rows),
                  informative_preferences=preferences,conflict_or_tie_rows=len(rows)-preferences,
                  no_missing_candidate_imputation=True)
    c.locked_json(folder/'cache_audit.json',report)
    return report


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('targets','collections','cache','sentinel'))
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--round',choices=('round1','round2'),default='round1')
    p.add_argument('--arm',choices=('shared',*f.ARMS),default='shared')
    p.add_argument('--collection')
    p.add_argument('--device',default='cpu')
    a = p.parse_args()
    if a.stage=='sentinel':
        verify_sentinel(a.run_root,a.collection)
    elif a.stage=='targets':
        targets(a.run_root,a.round,a.arm,a.device)
    elif a.stage=='collections':
        collections(a.run_root,a.round,a.arm)
    else:
        build_cache(a.run_root,a.round,a.arm)
