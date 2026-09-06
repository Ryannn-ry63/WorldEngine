"""Immutable sources, replay allowlist, legacy exposure and fixed collection manifests."""
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_rare_common as r
from prepare_selector_cfpi_deployment import save_subset
from report_selector_cfpi_deployment import checked_collection


def row(scene):
    return dict(scene_id=scene,origin_log=c.scene_origin_log(scene),origin_token=c.scene_token(scene),fold=None)


def freeze(run, old, canonical):
    path = run/'rare_inputs.json'
    if path.exists():
        result = d.read(path)
        if result.get('status')!='PASS' or result.get('method')!=r.METHOD:
            raise RuntimeError('Invalid frozen rare inputs')
        for item in result['artifacts'].values():
            d.verify(item)
        for item in result['replay_records']:
            d.verify(item)
        for item in result['models'].values():
            d.verify(item)
        for cohort in result['cohorts'].values():
            d.verify(cohort['scenario'])
        c.locked_json(run/'deployment_inputs.json',d.verified_read(result['artifacts']['old_inputs']))
        return result
    source = d.read(old/'deployment_inputs.json')
    if d.read(old/'decision_ledger.json').get('active') or d.read(old/'phase_b_report.json')['status']!='PASS':
        raise RuntimeError('Prior deployment incomplete or active')
    files = dict(source['artifacts'])
    for item in files.values():
        d.verify(item)
    targets = d.verified_read(files['targets'])['targets']
    natural = d.verified_read(files['natural_membership'])['rows']
    split_path = canonical/'experiments/diffusiondrive/grpo_selector_v3_rare_clpdms_tuning_v1/closed_loop_split/split_audit.json'
    split = d.read(split_path)
    cohorts = {}
    for name,(count,log_count,expected_sha) in r.RARE_SPLITS.items():
        v = split['splits'][name]
        scenario = dict(path=v['scenario_file'],sha256=v['scenario_file_sha256'])
        if scenario['sha256']!=expected_sha:
            raise RuntimeError('Rare split differs from approved legacy membership')
        payload = c.load_pickle(d.verify(scenario))
        if (len(payload)!=count or set(payload)!=set(v['tokens'])
                or len({c.scene_origin_log(s) for s in payload})!=log_count):
            raise RuntimeError('Legacy rare split coverage changed')
        cohorts[name] = dict(rows=[row(s) for s in sorted(payload)],scenario=scenario,asset_family='navtest_failures')
    a,b = [{x['origin_log'] for x in cohorts[k]['rows']} for k in ('development','confirmation')]
    if a&b:
        raise RuntimeError('Rare splits overlap by log')
    common = [x for x in natural if x['scenario_family']=='matched_common']
    if len(common)!=64:
        raise RuntimeError('Common64 membership changed')
    source_audit = d.verified_read(files['source_audit'])
    source_scenario = dict(path=source_audit['train_scenario_file'],sha256=source_audit['train_scenario_file_sha256'])
    cohorts['common'] = dict(rows=common,scenario=save_subset(run/'scenarios/common64.pkl',source_scenario,[x['scene_id'] for x in common]),asset_family='navtrain')
    target_manifest = d.verified_read(files['targets'])
    cohorts['bridge'] = dict(rows=targets,scenario=dict(path=target_manifest['pilot64_scenario_file'],
                              sha256=target_manifest['pilot64_scenario_file_sha256']),asset_family='navtrain')
    excluded = a|b|{x['origin_log'] for x in natural+targets}
    history = source_audit['historical_v3_membership_audit']
    import json
    for key in ('development','certification'):
        item = history['splits'][key]
        entry = dict(path=item['source'],sha256=item['sha256'])
        excluded.update(json.loads(line)['log_name'] for line in d.verify(entry).read_text().splitlines() if line.strip())
        files['legacy_'+key] = entry
    entry = dict(path=source_audit['exclusions'],sha256=source_audit['exclusions_sha256'])
    excluded.update(d.verified_read(entry)['excluded_origin_logs'])
    files['ccv_exclusions'] = entry
    metrics = d.metrics_csv(d.verify(files['baseline_csv']))
    baseline_audit = d.verified_read(files['baseline_audit'])
    if len(metrics)!=1024 or {x['origin_log'] for x in targets}&(a|b):
        raise RuntimeError('Correction/evaluation split overlap or source coverage drift')
    selected, capacity = r.choose_replay(metrics,excluded,c.scene_origin_log)
    record_dir = Path(files['baseline_csv']['path']).parent/'diffusiondrive_cfpi_causal_records'
    replay, records = [], []
    from selector_cfpi_model import INPUTS
    for scene in selected:
        for step in c.DECISION_STEPS:
            item = d.artifact(record_dir/f'{c.scene_token(scene)}_{step}_cfpicausal.pkl')
            record = c.load_pickle(d.verify(item))
            if (record['rollout_scene_id']!=scene or record['decision_step']!=step
                    or record['checkpoint_sha256']!=c.CHECKPOINT_SHA256
                    or record['candidate_noise_namespace']!=c.NOISE_NAMESPACE
                    or record['code_sha']!=baseline_audit['code_sha']
                    or record['selector_rollout_contract']['rollout_implementation_sha256']!=baseline_audit['rollout_implementation_sha256']
                    or record['behavior_checkpoint_manifest_sha256']!=baseline_audit['checkpoint_manifest_sha256']
                    or record['state_step']!=step-1 or record['intervention_applied'] is not False):
                raise RuntimeError('Replay source identity/noise/checkpoint drift')
            visible = {key:c.checked_array(record[key],shape,key).astype(np.float32) for key,shape in INPUTS.values()}
            visible['reference_logits'] = c.checked_array(record['reference_logits'],(20,),'base logits').astype(np.float32)
            visible['v3_logits'] = c.checked_array(record['current_logits'],(20,),'V3 logits').astype(np.float32)
            visible.update(scene_id=scene,decision_step=step,origin_log=c.scene_origin_log(scene))
            replay.append(visible)
            records.append(item)
    c.atomic_pickle(run/'cache/replay512.pkl',dict(rows=replay,scene_ids=selected,decisions=list(c.DECISION_STEPS)))
    files.update(replay=d.artifact(run/'cache/replay512.pkl'),old_inputs=d.artifact(old/'deployment_inputs.json'),
                 old_contract=d.artifact(old/'run_contract.json'),old_negative_report=d.artifact(old/'phase_b_report.json'),
                 rare_split=d.artifact(split_path))
    for name in ('0827_meeting_note.md','project_progress_note.md',
                 'DIFFUSIONDRIVE_SELECTOR_V4_ASTRA_SESSION_HANDOFF_20260905.md',
                 'DIFFUSIONDRIVE_SELECTOR_V3_DIAGNOSTIC_SYNTHESIS_AND_V4_FOUNDATION_20260904.md'):
        if (canonical/name).is_file():
            files['context_'+name] = d.artifact(canonical/name)
    models = dict(source['models'])
    for method in ('q_grpo_t1','q_mse','local_grpo_t1'):
        for seed in r.SEEDS:
            key = f'{method}_seed{seed}'
            report_path = old/'refit'/key/'report.json'
            report = d.read(report_path)
            if report['status']!='PASS' or report['train_scene_count']!=64:
                raise RuntimeError('Incomplete historical reference')
            models[key] = dict(report['selector'],kind='cfpi',score_mode=report['score_mode'],provenance=report['provenance'])
            d.verify(models[key])
            files['refit_'+key] = d.artifact(report_path)
    # Retain old successful engineering audits and negative results. Metrics reuse
    # is limited to identical scene/noise/controller + original model checkpoints.
    old_contract = d.verified_read(files['old_contract'])
    current_contract = d.read(run/'run_contract.json')
    protected = ('projects/AlgEngine/mmdet3d_plugin/navformer/', 'projects/SimEngine/worldengine/',
                 'projects/AlgEngine/closed_loop/', 'projects/AlgEngine/configs/diffusiondrive/')
    for name,sha in old_contract['implementation'].items():
        if name.startswith(protected) and current_contract['implementation'].get(name)!=sha:
            raise RuntimeError('Historical common reuse requires identical planner/controller/config: '+name)
    reuse = {}
    for policy in r.REFERENCES:
        print('Verifying historical common reference: '+policy,flush=True)
        entry = d.artifact(old/'collections'/('phase_b_'+policy)/'deployment_collection.json')
        checked = checked_collection(entry)
        if checked is None:
            raise RuntimeError('Missing audited historical common reference')
        contract = checked['collection']
        routing = d.verified_read(contract['routing'])
        if (contract['noise_namespace']!=c.NOISE_NAMESPACE or contract['checkpoint_sha256']!=c.CHECKPOINT_SHA256
                or routing['models'][policy]['sha256']!=models[policy]['sha256']
                or any(x['start_decision']!=4 for x in routing['routes'])):
            raise RuntimeError('Historical common inference contract mismatch')
        reuse[policy] = dict(collection=entry,audit=checked['audit'])
    result = dict(status='PASS',method=r.METHOD,artifacts=files,models=models,cohorts=cohorts,
                  replay_scene_ids=selected,replay_records=records,replay_capacity=capacity,
                  excluded_replay_logs=sorted(excluded),common_reuse=reuse,exposure=r.EXPOSURE,
                  aggregate_row_policy='Exclude overall_average; old 289-row report represented 288 scenes.',
                  new_branch_feedback_collected=False,inference_uses_reward_or_q=False)
    # Generic audit transport uses the historical schema, not a new efficacy claim.
    c.locked_json(run/'deployment_inputs.json',source)
    c.locked_json(path,result)
    return result


def collections(run, inputs, phase):
    output = run/f'{phase}_collections.json'
    if output.exists():
        result = d.read(output)
        for item in result['collections']:
            d.verify(item)
        return result
    winner = None
    if phase=='confirm':
        winner = d.read(run/'winner.json')
        d.verify(winner['screen_report'])
        for entry in winner['models'].values():
            d.verify(entry)
    jobs = r.expected_conditions(phase,winner['method'] if winner else None)
    entries = []
    for cohort,policy in jobs:
        data = inputs['cohorts'][cohort]
        folder = run/'collections'/f'{phase}_{cohort}_{policy}'
        routes, models, initial = [], {}, {}
        for item in data['rows']:
            key = f"{policy}_fold{item['fold']}" if phase=='bridge' else policy
            if key in inputs['models']:
                model = inputs['models'][key]
            else:
                report = d.read(run/'train'/policy/'report.json')
                method,seed = policy.rsplit('_seed',1)
                if (report['status']!='PASS' or report['provenance']['method']!=method
                        or report['provenance']['seed']!=int(seed) or report['provenance']['step']!=500
                        or report['provenance']['inputs_sha256']!=c.sha256_file(run/'rare_inputs.json')):
                    raise RuntimeError('Unverified trained model')
                model = dict(report['selector'],kind='rare',score_mode='residual',provenance=report['provenance'])
                if phase=='confirm' and report['selector']!=winner['models'][key]:
                    raise RuntimeError('Winner model changed after selection')
            d.verify(model)
            models[key] = model
            routes.append(dict(scene_id=item['scene_id'],origin_token=item['origin_token'],fold=item['fold'],
                               start_decision=4,model_key=key))
            if phase=='bridge':
                initial[item['scene_id']] = item['prefix_records']['4']
        routing = dict(status='PASS',method=d.METHOD,phase=phase,policy=policy,models=models,routes=routes,
                       terminal_publication_decision=12,incumbent_checkpoint_sha256=c.CHECKPOINT_SHA256,
                       incumbent_selector_sha256=inputs['models']['scalar_v3']['sha256'],inference_uses_reward_or_q=False)
        c.locked_json(folder/'routing.json',routing)
        canonical = d.read(run/'run_contract.json')['source_worldengine_root']
        contract = dict(status='PASS',method=d.METHOD,research_method=r.METHOD,phase=phase,cohort=cohort,policy=policy,
                        collection_id=folder.name,sentinel=False,audit_targets={},initial_contexts=initial,
                        routing=d.artifact(folder/'routing.json'),scenario=data['scenario'],
                        asset_folder=f"{canonical}/data/sim_engine/assets/{data['asset_family']}/assets",
                        inputs=d.artifact(run/'deployment_inputs.json'),run_contract=d.artifact(run/'run_contract.json'),
                        checkpoint_sha256=c.CHECKPOINT_SHA256,noise_namespace=c.NOISE_NAMESPACE,
                        expected_decisions=list(c.DECISION_STEPS))
        c.locked_json(folder/'deployment_collection.json',contract)
        entries.append(d.artifact(folder/'deployment_collection.json'))
    result = dict(status='PASS',phase=phase,collections=entries)
    c.locked_json(output,result)
    return result


if __name__=='__main__':
    import argparse
    from prepare_selector_cfpi_deployment import verify_baselines
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('freeze','baselines','bridge','screen','confirm'))
    for key in ('run-root','deployment-run','canonical'):
        p.add_argument('--'+key,type=Path,required=True)
    a = p.parse_args()
    inputs = freeze(a.run_root,a.deployment_run,a.canonical)
    if a.stage=='baselines':
        verify_baselines(a.run_root,d.read(a.run_root/'deployment_inputs.json'))
    elif a.stage!='freeze':
        collections(a.run_root,inputs,a.stage)
    print('PASS: rare preparation '+a.stage)
