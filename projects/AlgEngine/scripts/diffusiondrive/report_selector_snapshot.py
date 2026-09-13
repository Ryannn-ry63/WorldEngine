"""Complete observed-best report; no performance-dependent artifact suppression."""
import csv
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_snapshot_common as s
from selector_snapshot_audit import checked
from selector_snapshot_tasks import checked_open
from selector_rare_common import cluster_interval


def comparison(actual,reference,rows):
    ids=[r['scene_id'] for r in rows]
    if not ids or not set(ids)<=set(actual) or not set(ids)<=set(reference):
        raise RuntimeError('Incomplete paired comparison')
    differences={k:{i:actual[i][k]-reference[i][k] for i in ids} for k in d.METRICS}
    intervals={}
    for k in ('score','success','no_at_fault_collisions','drivable_area_compliance'):
        interval=cluster_interval(differences[k],rows)
        interval.update(seed_average_first=False,exposure=s.EXPOSURE,
                        conditional_on_fixed_checkpoint_and_noise=True)
        intervals[k]=interval
    return dict(scenes=len(ids),delta={k:float(np.mean(list(v.values()))) for k,v in differences.items()},
                rescued=sum(reference[i]['success']==0 and actual[i]['success']==1 for i in ids),
                broken=sum(reference[i]['success']==1 and actual[i]['success']==0 for i in ids),
                intervals=intervals)


def report(run):
    run=Path(run);inputs=s.checked_inputs(run);table={};missing=[];artifacts=[]
    for policy,block in s.conditions():
        if block.startswith('open'):
            value=checked_open(run,policy,block)
            path=run/'openloop'/policy/block/'audit.json'
        else:
            path=run/'collections'/f'{block}_{policy}'/'audit.json'
            value=checked(path.parent) if path.exists() else None
        if value is None:
            missing.append([policy,block])
        else:
            table[(policy,block)]=value
            artifacts.append(d.artifact(path))
    if missing:
        result=dict(status='INCOMPLETE',missing=missing,complete_conditions=len(table),expected_conditions=12)
        c.atomic_json(run/'progress.json',result)
        return result
    models={p:{block:dict(count=len(table[(p,block)]['metrics']),mean=s.mean(table[(p,block)]['metrics']),
                        **({'ade_fde':table[(p,block)]['ade_fde']} if block.startswith('open') else {}))
               for p2,block in s.conditions() if p2==p} for p in s.POLICIES}
    comparisons={};paired=[]
    dev_ids={r['scene_id'] for r in inputs['cohorts']['development']}
    for cohort in ('full','common'):
        for mode in ('NR','R'):
            block=cohort+'_'+mode
            baseline=table[('scalar_v3',block)]['metrics'];actual=table[('P1_seed2',block)]['metrics']
            rows=inputs['cohorts'][cohort]
            subsets={cohort:rows}
            if cohort=='full':
                subsets.update(development=[r for r in rows if r['scene_id'] in dev_ids],
                               remainder230=[r for r in rows if r['scene_id'] not in dev_ids])
            for label,subset in subsets.items():
                comparisons[label+'_'+mode]=comparison(actual,baseline,subset)
            for row in rows:
                i=row['scene_id']
                paired.append(dict(cohort=cohort,mode=mode,scene_id=i,origin_log=row['origin_log'],
                              development_exposed=i in dev_ids,
                              **actual[i],**{'scalar_'+k:v for k,v in baseline[i].items()}))
    historical={};old_summary=d.verified_read(inputs['artifacts']['historical_summary'])
    source_summary=d.verified_read(inputs['artifacts']['source_report'])
    for mode in ('NR','R'):
        previous=s.metrics(d.verify(inputs['artifacts']['historical_closedloop_'+mode.lower()]))
        if set(previous)!={v['scene_id'] for v in inputs['cohorts']['full']}:
            raise RuntimeError('Historical full membership changed')
        historical[mode]=dict(original_reported=old_summary['metrics']['closedloop_nonreactive' if mode=='NR' else 'closedloop_reactive'],
            historical_real_scene_mean=s.mean(previous),
            historical_development_mean=s.mean({k:v for k,v in previous.items() if k in dev_ids}),
            current_noise0_development=source_summary['reference_means'][f'development_{mode}_scalar_v3'],
            formal_noise_development=s.mean({k:v for k,v in table[('scalar_v3','full_'+mode)]['metrics'].items() if k in dev_ids}),
            native_formal_vs_historical=comparison(table[('scalar_v3','full_'+mode)]['metrics'],previous,inputs['cohorts']['full']))
    result=dict(status='PASS',decision='COMPLETE_OBSERVED_BEST_SNAPSHOT',performance_gate_applied=False,
        exposure=s.EXPOSURE,models=models,comparisons=comparisons,historical_protocol_bridge=historical,
        source_P3_decision=inputs['source_decision'],
        source_three_seed_report=inputs['artifacts']['source_report'],artifacts=artifacts,
        inputs=d.artifact(run/'inputs.json'),bridge=d.artifact(run/'bridge_report.json'),
        count_convention='Exclude average/overall_average. navtest12146; rare/full288; common64.',
        historical_success_rate_note='Legacy denominator289 included aggregate; corrected denominator288.',
        independent_confirmation=False)
    target=run/'paired_scenes.csv';temp=target.with_suffix('.tmp')
    with temp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    temp.replace(target);result['paired_scenes']=d.artifact(target)
    lines=['# P1 seed2 observed-best snapshot','',
           'Development-selected single checkpoint; not a three-seed main result or blind confirmation.','',
           '| Model | OL-nav PDM | OL-rare PDM | CL-NR PDM | CL-R PDM | CL-R success |',
           '|---|---:|---:|---:|---:|---:|']
    keys=('no_at_fault_collisions','drivable_area_compliance','ego_progress',
          'time_to_collision_within_bound','comfort','score')
    blocks=('open_navtest','open_rare','full_NR','full_R')
    tsv=['\t'.join(['model','note','ADE','FDE',*[b+'_'+k for b in blocks for k in keys],'CL_R_success'])]
    for p in s.POLICIES:
        v=models[p];scores=[v[b]['mean']['score'] for b in ('open_navtest','open_rare','full_NR','full_R')]
        sr=v['full_R']['mean']['success']
        lines.append('| '+p+' | '+' | '.join(f'{z:.6f}' for z in scores+[sr])+' |')
        tsv.append('\t'.join([p,'observed-best snapshot; eval noise0; real-scene denominator',
             *[f"{v['open_navtest']['ade_fde'][k]:.6f}" for k in ('ade_4s','fde_4s')],
             *[f"{v[b]['mean'][k]:.6f}" for b in blocks for k in keys],f'{sr:.6f}']))
    lines+=['','See summary.json for all components, common retention, paired intervals and historical count correction.',
            'All outcomes are retained irrespective of improvement; no next training/evaluation is authorized.']
    # Atomic, repeatable outputs are derived solely from the frozen artifacts.
    for name,text in (('summary.md','\n'.join(lines)+'\n'),('summary.tsv','\n'.join(tsv)+'\n')):
        p=run/name;tmp=p.with_suffix('.tmp');tmp.write_text(text);tmp.replace(p)
    c.locked_json(run/'summary.json',result)
    c.locked_json(run/'archive/release_manifest.json',dict(method=s.METHOD,status='PASS',exposure=s.EXPOSURE,
        contract=d.artifact(run/'run_contract.json'),audit=d.artifact(run/'audit_report.json'),
        inputs=d.artifact(run/'inputs.json'),summary=d.artifact(run/'summary.json'),
        paired_scenes=result['paired_scenes'],models=inputs['models'],source_artifacts=inputs['artifacts'],
        outputs=[d.artifact(run/'summary.md'),d.artifact(run/'summary.tsv')]))
    return result
