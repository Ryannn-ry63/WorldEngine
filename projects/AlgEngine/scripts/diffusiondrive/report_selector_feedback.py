"""Fixed T-only advancement; seed-averaged scene-weighted log-cluster intervals."""
import argparse
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
from selector_rare_common import cluster_interval, delta
from report_selector_cfpi_deployment import checked_collection,differences


def expected(phase):
    if phase=='screen':
        return [('development',mode,policy) for mode in f.MODES
                for policy in ('scalar_v3','gate_v3','S_seed0','U_seed0','T_seed0')]
    if phase=='final':
        return ([('development',mode,policy) for mode in f.MODES
                 for policy in ('scalar_v3','gate_v3','T_seed0','T_seed1','T_seed2')]
                +[('common',mode,policy) for mode in f.MODES
                  for policy in ('scalar_v3','T_seed0','T_seed1','T_seed2')])
    raise RuntimeError('No confirmation stage authorized')


def collection_name(cohort,mode,policy):
    return f'eval_{cohort}_{mode}_{policy}'


def report(run,phase):
    inputs = d.read(run/'feedback_inputs.json')
    data,missing,artifacts = {},[],[]
    for cohort,mode,policy in expected(phase):
        name = collection_name(cohort,mode,policy)
        path = run/'collections'/name/'deployment_collection.json'
        if not path.exists():
            missing.append(name); continue
        entry = d.artifact(path)
        checked = checked_collection(entry)
        if checked is None:
            missing.append(name); continue
        contract = checked['collection']
        if (contract['cohort']!=cohort or contract['react_type']!=mode or contract['policy']!=policy
                or contract['research_method']!=f.METHOD):
            raise RuntimeError('Evaluation condition identity mismatch')
        expected_scenes = {r['scene_id'] for r in inputs['cohorts'][cohort]['rows']}
        if set(checked['metrics'])!=expected_scenes:
            raise RuntimeError('Evaluation coverage changed')
        data[(cohort,mode,policy)] = checked['metrics']
        artifacts.extend((entry,checked['audit']))
    if missing:
        result = dict(status='INCOMPLETE',phase=phase,missing=missing,decision='NOT_AUTHORIZED')
        c.atomic_json(run/(phase+'_progress.json'),result)
        return result
    effects,details,intervals = {},{},{ }
    rows = inputs['cohorts']['development']['rows']
    for mode in f.MODES:
        v3 = data[('development',mode,'scalar_v3')]
        gate = data[('development',mode,'gate_v3')]
        if phase=='screen':
            current = data[('development',mode,'T_seed0')]
            controls = dict(scalar=v3,gate=gate,S=data[('development',mode,'S_seed0')],U=data[('development',mode,'U_seed0')])
            effects[mode] = {k:delta(current,v) for k,v in controls.items()}
            details[mode] = {k:differences(current,v,rows) for k,v in controls.items()}
            intervals[mode] = {k:cluster_interval({s:current[s]['score']-v[s]['score'] for s in current},rows) for k,v in controls.items()}
        else:
            current = [data[('development',mode,f'T_seed{s}')] for s in (0,1,2)]
            common = [data[('common',mode,f'T_seed{s}')] for s in (0,1,2)]
            common_v3 = data[('common',mode,'scalar_v3')]
            effects[mode] = dict(scalar=[delta(x,v3) for x in current],gate=[delta(x,gate) for x in current],
                                 common=[delta(x,common_v3) for x in common])
            details[mode] = {k:[differences(x,v,rows) for x in current] for k,v in dict(scalar=v3,gate=gate).items()}
            intervals[mode] = {}
            for name,reference in dict(scalar=v3,gate=gate).items():
                intervals[mode][name] = {metric:cluster_interval(
                    {scene:float(np.mean([x[scene][metric] for x in current]))-reference[scene][metric] for scene in reference},rows)
                    for metric in ('score','success')}
    gate = f.shortlist_gate(effects) if phase=='screen' else f.final_gate(effects)
    if phase=='screen':
        decision = 'PROCEED_T_SEEDS_1_2' if gate['passed'] else 'STOP_NO_MECHANISM_SIGNAL'
    elif not gate['passed']:
        decision = 'STOP_EFFECT_FAIL'
    else:
        # Point thresholds alone do not establish superiority/noninferiority.
        supported = all(intervals[mode]['scalar']['score']['scene_weighted_cluster_95'][0]>0
                        and intervals[mode]['gate']['score']['scene_weighted_cluster_95'][0]>=-.01
                        and intervals[mode]['scalar']['success']['scene_weighted_cluster_95'][0]>=0
                        and intervals[mode]['gate']['success']['scene_weighted_cluster_95'][0]>0 for mode in f.MODES)
        decision = 'CANDIDATE_REQUIRES_NEW_CONFIRMATION_PLAN' if supported else 'INSUFFICIENT_EVIDENCE'
    result = dict(status='PASS',phase=phase,method=f.METHOD,decision=decision,gate=gate,effects=effects,
                  details=details,intervals=intervals,artifacts=artifacts,exposure=f.EXPOSURE,
                  feedback_collection_seed=0,formal_confirmation_authorized=False)
    c.locked_json(run/(phase+'_report.json'),result)
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--phase',choices=('screen','final'),required=True)
    a = p.parse_args()
    v = report(a.run_root,a.phase)
    print({k:v[k] for k in ('status','phase','decision')},flush=True)
