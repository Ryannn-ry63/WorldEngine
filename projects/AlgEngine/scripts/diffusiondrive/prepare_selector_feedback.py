"""Immutable source membership, filtered dense replay, and collection contracts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f


def freeze(run, prior, canonical):
    destination = run/'feedback_inputs.json'
    if destination.exists():
        result = d.read(destination)
        if result['method'] != f.METHOD or result['status'] != 'PASS':
            raise RuntimeError('Wrong feedback source')
        for item in result['artifacts'].values():
            d.verify(item)
        for model in result['models'].values():
            d.verify(model)
        for cohort in result['cohorts'].values():
            d.verify(cohort['scenario'])
        return result
    old = d.read(prior/'rare_inputs.json')
    if d.read(prior/'decision_ledger.json').get('active'):
        raise RuntimeError('Prior run is active')
    report = d.read(prior/'screen_report.json')
    if report.get('status') != 'PASS':
        raise RuntimeError('Prior negative result must be engineering-complete')
    artifacts = dict(prior_inputs=d.artifact(prior/'rare_inputs.json'),
                     prior_report=d.artifact(prior/'screen_report.json'))
    for key in ('baseline_csv','baseline_audit','legacy_development','legacy_certification',
                'ccv_exclusions','natural_membership','rare_split','scalar_manifest','gate_manifest'):
        artifacts[key] = old['artifacts'][key]
        d.verify(artifacts[key])
    old_inputs = d.verified_read(old['artifacts']['old_inputs'])
    source_entry = old_inputs['artifacts']['source_audit']
    source = d.verified_read(source_entry)
    artifacts['source_audit'] = source_entry
    natural = d.verified_read(artifacts['natural_membership'])['rows']
    excluded = {r['origin_log'] for name in ('development','confirmation') for r in old['cohorts'][name]['rows']}
    excluded.update(r['origin_log'] for r in natural)
    for key in ('legacy_development','legacy_certification'):
        excluded.update(json.loads(line)['log_name'] for line in d.verify(artifacts[key]).read_text().splitlines() if line.strip())
    excluded.update(d.verified_read(artifacts['ccv_exclusions'])['excluded_origin_logs'])
    pools = {}
    for item in source['source_inventories']:
        entry = dict(path=item['index'],sha256=item['index_sha256'])
        index = d.verified_read(entry)
        artifacts['index_'+item['family']] = entry
        pools[item['family']] = {s for shard in index['shards'] for s in shard['scenario_ids']}
    metrics = d.metrics_csv(d.verify(artifacts['baseline_csv']))
    rows = f.source_selection(pools,metrics,excluded)
    source_scenario = dict(path=source['train_scenario_file'],sha256=source['train_scenario_file_sha256'])
    artifacts['source_scenario'] = source_scenario
    from prepare_selector_cfpi_deployment import save_subset
    scenario = save_subset(run/'scenarios/train128.pkl',source_scenario,[r['scene_id'] for r in rows])
    models = {k:old['models'][k] for k in ('scalar_v3','gate_v3')}
    for model in models.values():
        d.verify(model)
    cohorts = {k:old['cohorts'][k] for k in ('development','common')}
    cohorts['train'] = dict(rows=rows,scenario=scenario,asset_family='navtrain')
    result = dict(status='PASS',method=f.METHOD,artifacts=artifacts,models=models,cohorts=cohorts,
                  excluded_logs=sorted(excluded),source_baseline_mode='R',exposure=f.EXPOSURE,
                  historical_training_pilot_logs_may_be_reused=True,inference_uses_reward_or_q=False,
                  source_counts=dict(scenes=len(rows),logs=len({r['origin_log'] for r in rows}),maximum_per_log=2))
    c.locked_json(destination,result)
    return result


def dense_cache(run, canonical):
    """Materialize filtered arrays once; training memory-maps instead of loading 8GB per job."""
    import torch
    output = run/'dense/manifest.json'
    if output.exists():
        result = d.read(output)
        if result['inputs'] != d.artifact(run/'feedback_inputs.json'):
            raise RuntimeError('Dense source contract drift')
        for item in result['sources'].values():
            for entry in item['arrays'].values():
                d.verify(entry)
        for item in result['artifacts'].values():
            d.verify(item)
        return result
    inputs = d.read(run/'feedback_inputs.json')
    excluded = set(inputs['excluded_logs'])
    path = canonical/'experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/data/manifest.json'
    manifest = d.read(path)
    hard_entry = dict(path=manifest['hard_pool'],sha256=manifest['hard_pool_sha256'])
    pair_entry = dict(path=manifest['pair_manifest'],sha256=manifest['pair_manifest_sha256'])
    hard = [json.loads(x) for x in d.verify(hard_entry).read_text().splitlines() if x.strip()]
    # The original pairing is same-log. Independently verify both token origins.
    pairs = [json.loads(x) for x in d.verify(pair_entry).read_text().splitlines() if x.strip()]
    origins = {}
    for row in pairs:
        for key in ('rare_token','paired_common_token','common_token'):
            if key in row:
                token = row[key]
                if token in origins and origins[token] != row['log_name']:
                    raise RuntimeError('Dense token has ambiguous log')
                origins[token] = row['log_name']
    kept = [r for r in hard if r['split']=='train' and r['log_name'] not in excluded]
    if not kept:
        raise RuntimeError('No clean dense training data')
    for row in kept:
        if any(origins.get(row[k]) != row['log_name'] for k in ('rare_token','paired_common_token')):
            raise RuntimeError('Cannot prove dense common/rare same-log membership')
    fields = (*f.FINGERPRINT_FIELDS,'candidate_rewards','candidate_reward_valid_mask')
    sources, artifacts, mappings = {}, dict(manifest=d.artifact(path),hard=hard_entry,pairs=pair_entry), {}
    sources_spec = [(f'real{s}',dict(path=v['cache'],sha256=v['cache_sha256'])) for s,v in sorted(manifest['real_caches'].items())]
    sources_spec.append(('synthetic',dict(path=manifest['synthetic_cache'],sha256=manifest['synthetic_cache_sha256'])))
    for name,entry in sources_spec:
        print('Preparing filtered dense source: '+name,flush=True)
        cache = torch.load(d.verify(entry),map_location='cpu')
        artifacts[name] = entry
        if name=='synthetic':
            indices = sorted({int(r['synthetic_index']) for r in kept if r['hard_kind']=='synthetic_rollout'})
            mapping = {str(i):j for j,i in enumerate(indices)}
            for row in kept:
                if row['hard_kind']=='synthetic_rollout' and cache['origin_rare_tokens'][int(row['synthetic_index'])] != row['rare_token']:
                    raise RuntimeError('Synthetic origin changed')
        else:
            tokens = {r['paired_common_token'] for r in kept}|{r['rare_token'] for r in kept if r['hard_kind']=='real_rare'}
            positions = {t:i for i,t in enumerate(cache['tokens'])}
            if not tokens <= positions.keys():
                raise RuntimeError('Missing dense tokens')
            ordered = sorted(tokens)
            indices = [positions[t] for t in ordered]
            mapping = {t:i for i,t in enumerate(ordered)}
        arrays = {}
        for key in fields:
            target = run/'dense'/name/(key+'.npy')
            target.parent.mkdir(parents=True,exist_ok=True)
            value = cache[key][indices].cpu().numpy()
            if key != 'candidate_reward_valid_mask' and not np.isfinite(value).all():
                raise RuntimeError('Nonfinite dense input/label')
            temporary = target.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream,value,allow_pickle=False)
            temporary.replace(target)
            arrays[key] = d.artifact(target)
        mappings[name] = mapping
        sources[name] = dict(arrays=arrays,rows=len(indices))
        del cache
    records = []
    for row in kept:
        common = [[name,mappings[name][row['paired_common_token']]] for name in mappings if name.startswith('real')]
        hard_rows = ([[name,mappings[name][row['rare_token']]] for name in mappings if name.startswith('real')]
                     if row['hard_kind']=='real_rare' else [['synthetic',mappings['synthetic'][str(row['synthetic_index'])]]])
        records.append(dict(common=common,hard=hard_rows,origin_log=row['log_name'],kind=row['hard_kind']))
    result = dict(status='PASS',inputs=d.artifact(run/'feedback_inputs.json'),sources=sources,records=records,
                  artifacts=artifacts,excluded_overlap=0)
    c.locked_json(output,result)
    return result


def model_entry(run, policy, inputs):
    if policy in inputs['models']:
        return inputs['models'][policy]
    report = d.read(run/'train'/policy/'report.json')
    arm,seed = policy.rsplit('_seed',1)
    provenance = report['provenance']
    if (report['status']!='PASS' or provenance['inputs_sha256']!=c.sha256_file(run/'feedback_inputs.json')
            or provenance['method']!=f.METHOD or provenance['arm']!=arm or provenance['seed']!=int(seed)
            or provenance['steps']!=500 or provenance['code_sha']!=d.read(run/'run_contract.json')['code_sha']):
        raise RuntimeError('Unverified feedback model')
    d.verify(report['selector'])
    return dict(report['selector'],kind='cfpi',score_mode='residual',provenance=report['provenance'])


def collection(run, name, cohort, mode, policy, scene_ids=None, targets=None, continuation=None):
    inputs = d.read(run/'feedback_inputs.json')
    data = inputs['cohorts'][cohort]
    members = {r['scene_id']:r for r in data['rows']}
    scenes = sorted(members if scene_ids is None else scene_ids)
    if mode not in f.MODES or not scenes or len(set(scenes))!=len(scenes) or not set(scenes)<=members.keys():
        raise RuntimeError('Invalid collection membership/mode')
    if targets and (cohort!='train' or set(targets)!=set(scenes)):
        raise RuntimeError('Invalid intervention membership')
    folder = run/'collections'/name
    from prepare_selector_cfpi_deployment import save_subset
    scenario = (data['scenario'] if set(scenes)==set(members) else
                save_subset(folder/'scenarios.pkl',data['scenario'],scenes))
    models = {key:model_entry(run,key,inputs) for key in {policy,continuation} if key is not None}
    routes = []
    for scene in scenes:
        route = dict(scene_id=scene,origin_token=c.scene_token(scene),fold=None,start_decision=4,model_key=policy)
        if targets:
            route.update(feedback_target=targets[scene],continuation_model_key=continuation)
        routes.append(route)
    routing = dict(status='PASS',method=d.METHOD,research_method=f.METHOD,policy=policy if not targets else name,
                   models=models,routes=routes,terminal_publication_decision=12,
                   incumbent_selector_sha256=inputs['models']['scalar_v3']['sha256'],inference_uses_reward_or_q=False)
    c.locked_json(folder/'routing.json',routing)
    canonical = d.read(run/'run_contract.json')['source_worldengine_root']
    result = dict(status='PASS',method=d.METHOD,research_method=f.METHOD,collection_id=name,
                  cohort=cohort,react_type=mode,policy=policy,continuation_policy=continuation,
                  sentinel=False,audit_targets={},initial_contexts={},routing=d.artifact(folder/'routing.json'),
                  scenario=scenario,asset_folder=f"{canonical}/data/sim_engine/assets/{data['asset_family']}/assets",
                  inputs=d.artifact(run/'feedback_inputs.json'),run_contract=d.artifact(run/'run_contract.json'),
                  checkpoint_sha256=c.CHECKPOINT_SHA256,noise_namespace=c.NOISE_NAMESPACE,
                  expected_decisions=list(c.DECISION_STEPS))
    c.locked_json(folder/'deployment_collection.json',result)
    return d.artifact(folder/'deployment_collection.json')


def stage_collections(run, stage):
    if stage in ('baseline','visit'):
        policy = 'scalar_v3' if stage=='baseline' else 'shared_seed0'
        result = [collection(run,stage+'_'+mode,'train',mode,policy) for mode in f.MODES]
    else:
        from report_selector_feedback import expected,collection_name
        result = [collection(run,collection_name(cohort,mode,policy),cohort,mode,policy)
                  for cohort,mode,policy in expected(stage)]
    c.locked_json(run/(stage+'_collections.json'),dict(status='PASS',collections=result))
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('freeze','dense','baseline','visit','screen','final'))
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--prior-run',type=Path,required=True)
    p.add_argument('--canonical',type=Path,required=True)
    a = p.parse_args()
    freeze(a.run_root,a.prior_run,a.canonical)
    if a.stage=='dense':
        dense_cache(a.run_root,a.canonical)
    elif a.stage!='freeze':
        stage_collections(a.run_root,a.stage)
