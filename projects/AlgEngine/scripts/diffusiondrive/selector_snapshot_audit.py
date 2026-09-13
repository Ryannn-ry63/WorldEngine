"""Verify native simulation consumption, full membership and export bridge."""
import csv
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_snapshot_common as s
from audit_selector_cfpi_deployment import action_error, load_current_reports, validate_coverage
from audit_selector_root_cause_r1_collection import completed_scenes


def make_collection(run, policy, cohort, mode):
    inputs=s.checked_inputs(run)
    name=f'{cohort}_{mode}_{policy}'
    root=Path(run)/'collections'/name
    rows=inputs['cohorts'][cohort]
    scene=inputs['artifacts'][{'full':'full_scenario','common':'common_scenario','bridge':'bridge_scenario'}[cohort]]
    prefixes={}
    for row in rows:
        for prefix in (row['scene_id'],row['scene_id'].rsplit('-',1)[1]):
            if prefix in prefixes and prefixes[prefix]!=row['scene_id']:
                raise RuntimeError('Ambiguous native scene prefix')
            prefixes[prefix]=row['scene_id']
    obj=dict(method=s.METHOD,id=name,policy=policy,cohort=cohort,mode=mode,
             checkpoint=inputs['models'][policy],scenario=scene,
             rows=rows,prefixes=prefixes,asset_folder=inputs['assets'][cohort],
             noise=c.NOISE_NAMESPACE if cohort=='bridge' else s.FORMAL_NOISE,
             contract=d.artifact(Path(run)/'run_contract.json'),inputs=d.artifact(Path(run)/'inputs.json'),
             workers=4)
    c.locked_json(root/'collection.json',obj)
    return root/'collection.json'


def audit(path):
    path=Path(path);root=path.parent;co=d.read(path);entry=d.artifact(path)
    code=d.verified_read(co['contract'])['code_sha']
    for k in ('scenario','checkpoint','inputs'):
        d.verify(co[k])
    scenes={r['scene_id'] for r in co['rows']}
    records=[];artifacts=[];sidecars=set();frames={};workers=set();plans={};scene_workers={}
    for p in sorted(root.glob('split_*/WE_output/openscene_format/snapshot_records/*.json')):
        rec=d.read(p)
        if rec['collection']!=entry:
            raise RuntimeError('Wrong observation collection')
        sc=c.load_pickle(d.verify(rec['sidecar']))
        v=s.validate_native(sc,co,code)
        if rec['validation']!=v or rec['state_step']!=v['decision_step']-1:
            raise RuntimeError('Native consumed frame identity mismatch')
        action_diff=action_error(rec['actual_action'],rec['expected_action'])
        if (not np.isfinite(action_diff) or action_diff>c.ACTION_TOLERANCE
                or not np.isfinite(rec['candidate_trajectory_max_abs'])
                or rec['candidate_trajectory_max_abs']>c.ACTION_TOLERANCE):
            raise RuntimeError('Actual action audit failed')
        plan=np.load(d.verify(rec['plan']),allow_pickle=False)
        if not np.isfinite(plan).all() or np.max(np.abs(plan-sc['deployed_trajectory']))>c.ACTION_TOLERANCE:
            raise RuntimeError('Plan/sidecar mismatch')
        key=(v['scene_id'],v['decision_step'])
        if key in frames:
            raise RuntimeError('Duplicate native action')
        frames[key]=dict(sidecar=rec['sidecar'],record=d.artifact(p))
        sidecars.add(Path(rec['sidecar']['path']).resolve())
        worker=p.relative_to(root).parts[0]
        if scene_workers.setdefault(v['scene_id'],worker)!=worker:
            raise RuntimeError('Scene actions cross workers')
        workers.add(worker)
        plans[(worker,sc['scene_prefix'],v['decision_step'])]=v['selected_index']
        records.append(v);artifacts.extend((d.artifact(p),rec['sidecar'],rec['plan']))
    paths,reports=load_current_reports(root);ledgers,completed=completed_scenes(root)
    validate_coverage(records,scenes,completed,reports)
    if workers!={f'split_{k}' for k in range(co['workers'])}:
        raise RuntimeError('Worker coverage mismatch')
    terminal=set()
    for p in root.glob('split_*/diffusiondrive_candidate_sidecars/*.pkl'):
        if p.resolve() in sidecars:
            continue
        sc=c.load_pickle(p);v=s.validate_native(sc,co,code,terminal=True)
        if v['scene_id'] in terminal:
            raise RuntimeError('Duplicate terminal publication')
        terminal.add(v['scene_id'])
        worker=p.relative_to(root).parts[0]
        if scene_workers.get(v['scene_id'])!=worker:
            raise RuntimeError('Terminal publication on wrong worker')
        plan=p.parent.parent/'plan_traj'/f"{sc['scene_prefix']}_12.npy"
        published=np.load(plan,allow_pickle=False)
        if not np.isfinite(published).all() or np.max(np.abs(published-sc['deployed_trajectory']))>c.ACTION_TOLERANCE:
            raise RuntimeError('Terminal plan mismatch')
        plans[(worker,sc['scene_prefix'],12)]=v['selected_index']
        artifacts.extend((d.artifact(p),d.artifact(plan)))
    seen_plans={}
    for p in root.glob('split_*/plan_traj/plan_idx.csv'):
        worker=p.relative_to(root).parts[0]
        with p.open() as stream:
            for item in csv.DictReader(stream):
                key=(worker,item['prefix'],int(item['step']))
                value=int(item['plan_idx'])
                if key not in plans or value!=plans[key] or (key in seen_plans and seen_plans[key]!=value):
                    raise RuntimeError('Unknown/conflicting planner CSV decision')
                seen_plans[key]=value
        artifacts.append(d.artifact(p))
    if seen_plans!=plans:
        raise RuntimeError('Missing planner CSV decisions')
    values={}
    for w in sorted(workers):
        p=root/w/f"WE_output/openscene_format/all_scenes_pdm_averages_{co['mode']}.csv"
        subset=s.metrics(p)
        if values.keys()&subset.keys():
            raise RuntimeError('Duplicate scene metrics')
        values.update(subset);artifacts.append(d.artifact(p))
    if set(values)!=scenes:
        raise RuntimeError('Missing final metrics')
    output=root/'metrics.csv'
    temp=output.with_suffix('.tmp')
    with temp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['token',*d.METRICS])
        writer.writeheader()
        writer.writerows(dict(token=k,**values[k]) for k in sorted(values))
    temp.replace(output)
    result=dict(status='PASS',collection=entry,metrics=values,scenes=len(scenes),
                records={scene:{str(step):frames[(scene,step)] for step in c.DECISION_STEPS} for scene in sorted(scenes)},
                artifacts=artifacts+[d.artifact(p) for p in paths+ledgers],
                merged_metrics=d.artifact(output),terminal_publications=len(terminal),
                incomplete_or_dropped_scenes=0)
    c.locked_json(root/'audit.json',result)
    return result


def checked(root, full=False):
    obj=d.read(Path(root)/'audit.json')
    d.verify(obj['collection']);d.verify(obj['merged_metrics'])
    if full:
        for e in obj['artifacts']:
            d.verify(e)
    co=d.verified_read(obj['collection'])
    run=Path(root).parents[1]
    inputs=d.verified_read(co['inputs'])
    if (co['contract']!=d.artifact(run/'run_contract.json') or co['inputs']!=d.artifact(run/'inputs.json')
            or co['method']!=s.METHOD or co['checkpoint']!=inputs['models'][co['policy']]
            or co['rows']!=inputs['cohorts'][co['cohort']]
            or co['noise']!=(c.NOISE_NAMESPACE if co['cohort']=='bridge' else s.FORMAL_NOISE)):
        raise RuntimeError('Audited native condition identity drift')
    if obj['status']!='PASS' or obj['metrics']!=s.metrics(d.verify(obj['merged_metrics']),{r['scene_id'] for r in co['rows']}):
        raise RuntimeError('Audited metric drift')
    return obj


def bridge_report(run):
    inputs=s.checked_inputs(run);results=[]
    from selector_feedback_common import FINGERPRINT_FIELDS
    for policy in s.POLICIES:
        for mode in ('NR','R'):
            native=checked(Path(run)/'collections'/f'bridge_{mode}_{policy}',full=True)
            old=d.verified_read(inputs['artifacts'][f'bridge_reference_{policy}_{mode}'])
            old_records={}
            for entry in old['artifacts']:
                if '/cfpi_deployment_records/' in entry['path']:
                    rec=d.verified_read(entry)
                    if rec['scene_id'] in native['metrics']:
                        old_records[(rec['scene_id'],rec['state_step']+1)]=rec
            for scene,values in native['metrics'].items():
                if any(abs(values[k]-old['metrics'][scene][k])>c.OUTCOME_TOLERANCE for k in d.METRICS):
                    raise RuntimeError('Native bridge changed outcome')
                for step in c.DECISION_STEPS:
                    n=d.verified_read(native['records'][scene][str(step)]['record'])
                    o=old_records[(scene,step)]
                    a=c.load_pickle(d.verify(n['sidecar']));b=c.load_pickle(d.verify(o['sidecar']))
                    # P1 bank recomputation and native full head may differ within score tolerance.
                    if int(a['selected_index'])!=int(b['selected_index']):
                        raise RuntimeError('Native bridge changed selection')
                    for key in FINGERPRINT_FIELDS:
                        if np.max(np.abs(np.asarray(a[key])-np.asarray(b[key])))>c.ARRAY_TOLERANCE:
                            raise RuntimeError('Native bridge changed context: '+key)
                    if (np.max(np.abs(a['current_logits']-b['current_logits']))>c.MODEL_RECOMPUTE_TOLERANCE
                            or action_error(n['actual_action'],o['actual_action'])>c.ACTION_TOLERANCE
                            or np.max(np.abs(a['deployed_trajectory']-b['deployed_trajectory']))>c.ACTION_TOLERANCE):
                        raise RuntimeError('Native bridge changed scores/trajectory')
            results.append(d.artifact(Path(run)/'collections'/f'bridge_{mode}_{policy}'/'audit.json'))
    result=dict(status='PASS',conditions=results,inputs=d.artifact(Path(run)/'inputs.json'),
                noise=c.NOISE_NAMESPACE,scenes_per_condition=8)
    c.locked_json(Path(run)/'bridge_report.json',result)
    return result
