"""Audited training interventions, hybrid sentinels and unchanged closed-loop evaluation."""
from pathlib import Path
import json
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import selector_decision_common as x
import selector_decision_data as data
from build_selector_feedback import collection_data


def make(run, name, cohort, mode, policy, requests=None, continuation=None,
         role='evaluation'):
    inputs = data.checked_inputs(run)
    if cohort not in ('train', 'development', 'common') or mode not in x.MODES:
        raise RuntimeError('Unregistered collection condition')
    members = {r['scene_id']: r for r in inputs['cohorts'][cohort]['rows']}
    blueprints = data.targets(run) if requests is not None else {}
    if requests is not None and (cohort != 'train' or not requests):
        raise RuntimeError('Interventions are training-only and nonempty')
    if role not in ('evaluation', 'query', 'diagnostic', 'sentinel', 'hybrid_reference'):
        raise RuntimeError('Unknown intervention role')
    routes = []
    if requests is not None:
        for q in requests:
            source = blueprints[q['row_id']]
            if (source['mode'] != mode or source['scene_id'] not in members
                    or source['visiting_policy'] != policy):
                raise RuntimeError('Requested source state does not belong to this condition')
            target = dict(source, row_id=q['row_id'],
                          forced_index=int(q['candidate_index']),
                          sentinel=role == 'sentinel',
                          continuation_policy=continuation,
                          no_intervention=role == 'hybrid_reference')
            routes.append(dict(scene_id=source['scene_id'], origin_token=c.scene_token(source['scene_id']),
                               fold=None, start_decision=4, model_key=policy,
                               feedback_target=target, continuation_model_key=continuation))
    else:
        routes = [dict(scene_id=s, origin_token=c.scene_token(s), fold=None,
                       start_decision=4, model_key=policy) for s in sorted(members)]
    scenes = [r['scene_id'] for r in routes]
    if not scenes or len(set(scenes)) != len(scenes):
        raise RuntimeError('Duplicate scene in an intervention slot')
    from prepare_selector_cfpi_deployment import save_subset
    folder = run/'collections'/name
    scenario = inputs['cohorts'][cohort]['scenario']
    if set(scenes) != set(members):
        scenario = save_subset(folder/'scenarios.pkl', scenario, scenes)
    models = {p: data.model_entry(run, p) for p in {policy, continuation} if p is not None}
    routing = dict(status='PASS', method=d.METHOD, research_method=x.METHOD,
                   policy=policy if requests is None else name, models=models, routes=routes,
                   terminal_publication_decision=12,
                   incumbent_selector_sha256=inputs['models']['scalar_v3']['sha256'],
                   inference_uses_reward_or_q=False)
    c.locked_json(folder/'routing.json', routing)
    contract = d.read(run/'run_contract.json')
    result = dict(status='PASS', method=d.METHOD, research_method=x.METHOD, collection_id=name,
                  cohort=cohort, react_type=mode, policy=policy, continuation_policy=continuation,
                  role=role, sentinel=False, audit_targets={}, initial_contexts={},
                  routing=d.artifact(folder/'routing.json'), scenario=scenario,
                  asset_folder=f"{contract['source_worldengine_root']}/data/sim_engine/assets/{inputs['cohorts'][cohort]['asset_family']}/assets",
                  inputs=d.artifact(run/'decision_inputs.json'), run_contract=d.artifact(run/'run_contract.json'),
                  checkpoint_sha256=c.CHECKPOINT_SHA256, noise_namespace=c.NOISE_NAMESPACE,
                  expected_decisions=list(c.DECISION_STEPS),
                  worker_count=min(contract['gpu_count'], len(scenes)))
    c.locked_json(folder/'deployment_collection.json', result)
    return d.artifact(folder/'deployment_collection.json')


def natural_requests(run, ids):
    blueprints = data.targets(run)
    return [dict(row_id=i, candidate_index=blueprints[i]['visiting_index']) for i in ids]


def diagnostic_bundle(run):
    spec = data.freeze_diagnostic(run)
    blueprints = data.targets(run)
    checks, treatments = [], []
    for mode in x.MODES:
        ids = f.ordered([i for i,t in blueprints.items() if t['mode'] == mode],
                        'decision-feedback-standard-sentinel-'+mode)[:8]
        sentinel = make(run, 'diag_standard_'+mode, 'train', mode, 'shared_seed0',
                        natural_requests(run, ids), 'shared_seed0', 'sentinel')
        references = {Path(blueprints[i]['source_collection']['path']).parent.name for i in ids}
        if len(references) != 1:
            raise RuntimeError('Ambiguous natural source collection')
        reference = blueprints[ids[0]]['source_collection']
        checks.append(dict(sentinel=sentinel, reference=reference))
        unsupported = [q for q in spec['unsupported'] if q['mode'] == mode]
        if unsupported:
            treatments.append(make(run, 'diag_missing_'+mode, 'train', mode, 'shared_seed0',
                                   unsupported, 'shared_seed0', 'diagnostic'))
        ids = [i for i in spec['continuation'] if blueprints[i]['mode'] == mode]
        natural = natural_requests(run, ids)
        reference = make(run, 'diag_hybrid_reference_'+mode, 'train', mode, 'shared_seed0',
                         natural, 'T_source', 'hybrid_reference')
        sentinel = make(run, 'diag_hybrid_sentinel_'+mode, 'train', mode, 'shared_seed0',
                        natural, 'T_source', 'sentinel')
        checks.append(dict(sentinel=sentinel, reference=reference))
        for slot in (0, 1):
            requests = [dict(row_id=i, candidate_index=blueprints[i]['candidate_indices'][slot]) for i in ids]
            treatments.append(make(run, f'diag_continuation_{mode}_{slot}', 'train', mode,
                                   'shared_seed0', requests, 'T_source', 'diagnostic'))
    result = dict(status='PASS', prechecks=checks, treatments=treatments,
                  targets=d.artifact(run/'diagnose/targets.json'))
    c.locked_json(run/'diagnose/collections.json', result)
    return result


def compare_sentinel(pair):
    def read(entry):
        d.verify(entry)
        path = Path(entry['path'])
        return collection_data(path.parents[2], path.parent.name)
    left, right = read(pair['sentinel']), read(pair['reference'])
    routing = d.verified_read(d.verified_read(pair['sentinel'])['routing'])
    reference_contract = d.verified_read(pair['reference'])
    reference_routing = d.verified_read(reference_contract['routing'])
    for key, model in routing['models'].items():
        if (key not in reference_routing['models'] or
                model['sha256'] != reference_routing['models'][key]['sha256']):
            raise RuntimeError('Sentinel/reference do not share the same policy bank')
    sentinel_contract = d.verified_read(pair['sentinel'])
    if sentinel_contract['react_type'] != reference_contract['react_type']:
        raise RuntimeError('Sentinel/reference mode mismatch')
    for route in routing['routes']:
        target = route['feedback_target']
        if not target['sentinel'] or target['forced_index'] != target['visiting_index']:
            raise RuntimeError('Sentinel is not the actual visiting action')
    for scene in left['metrics']:
        if scene not in right['metrics']:
            raise RuntimeError('Sentinel scene absent from matching continuation reference')
        if (any(abs(left['metrics'][scene][k]-right['metrics'][scene][k]) > c.OUTCOME_TOLERANCE
                for k in d.METRICS) or
                left['metrics'][scene]['first_violation_step'] != right['metrics'][scene]['first_violation_step']):
            raise RuntimeError('Sentinel changed outcome/timing under its matching continuation')
        for step in c.DECISION_STEPS:
            a = c.load_pickle(d.verify(left['records'][scene][step]))
            b = c.load_pickle(d.verify(right['records'][scene][step]))
            if (int(a['selected_index']) != int(b['selected_index']) or
                    any(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k]))) > c.ARRAY_TOLERANCE
                        for k in f.FINGERPRINT_FIELDS)):
                raise RuntimeError('Sentinel changed natural trajectory')
    result = dict(status='PASS', **pair, audits=[left['audit'], right['audit']])
    path = Path(pair['sentinel']['path']).parent/'sentinel_audit.json'
    c.locked_json(path, result)
    return d.artifact(path)


def outcomes(entry, continuation):
    path = d.verify(entry)
    checked = collection_data(path.parents[2], path.parent.name)
    contract = d.verified_read(entry)
    if (contract['research_method'] != x.METHOD or contract['cohort'] != 'train'
            or contract['continuation_policy'] != continuation
            or contract['role'] not in ('query', 'diagnostic')):
        raise RuntimeError('Invalid branch-label condition')
    routing = d.verified_read(contract['routing'])
    result = {}
    for route in routing['routes']:
        target = route['feedback_target']
        key = target['row_id']
        if target['sentinel'] or target.get('no_intervention') or key in result:
            raise RuntimeError('Sentinel/reference/duplicate cannot supply treatment label')
        result[key] = dict(candidate_index=target['forced_index'],
                           source_fingerprint=target['source_fingerprint'],
                           outcome={k: checked['metrics'][route['scene_id']][k] for k in d.METRICS})
    return result, checked['audit']


def require_pilot(run):
    report = d.read(run/'diagnose/report.json')
    if (report['status'] != 'PASS' or report['method'] != x.METHOD
            or report['gate']['decision'] != 'PROCEED_FIXED_PILOT'
            or report['code_sha'] != d.read(run/'run_contract.json')['code_sha']):
        raise RuntimeError('Diagnosis did not authorize the fixed pilot')
    for entry in report['artifacts']:
        d.verify(entry)
    if (report['gate'] != x.diagnosis_gate(
            [r for r in report['missing_action_details'] if r['dominators']],
            report['reversals'], len(report['continuation_details']))):
        raise RuntimeError('Diagnostic decision is inconsistent with evidence')
    return report


def diagnostic_report(run):
    spec = data.freeze_diagnostic(run)
    bundle = d.read(run/'diagnose/collections.json')
    artifacts = [d.artifact(run/'diagnose/targets.json'), d.artifact(run/'diagnose/collections.json')]
    for pair in bundle['prechecks']:
        artifacts.append(compare_sentinel(pair))
    rows = {r['row_id']: r for r in data.initial_rows(run) if r['fresh']}
    missing, changed = {}, {}
    for entry in bundle['treatments']:
        contract = d.verified_read(entry)
        result, audit = outcomes(entry, contract['continuation_policy'])
        artifacts.extend((entry, audit))
        for key, value in result.items():
            if contract['continuation_policy'] == 'shared_seed0':
                if key in missing:
                    raise RuntimeError('Duplicate missing-action diagnostic')
                missing[key] = value
            else:
                changed.setdefault(key, {})[value['candidate_index']] = value['outcome']
    if set(missing) != {q['row_id'] for q in spec['unsupported']} or set(changed) != set(spec['continuation']):
        raise RuntimeError('Incomplete diagnostic coverage')
    dominated, details, reversals = [], [], 0
    for q in spec['unsupported']:
        row, actual = rows[q['row_id']], missing[q['row_id']]
        if actual['candidate_index'] != q['candidate_index'] or actual['source_fingerprint'] != row['source_fingerprint']:
            raise RuntimeError('Wrong missing-action treatment')
        dominators = [i for i,known in zip(row['candidate_indices'], row['branch_outcomes'])
                      if (f.pareto_pair(known, actual['outcome']) or (-1,))[0] == 0]
        detail = dict(row_id=q['row_id'], scene_id=row['scene_id'], origin_log=row['origin_log'],
                      mode=row['mode'], selected=q['candidate_index'], outcome=actual['outcome'],
                      dominators=dominators)
        details.append(detail)
        if dominators:
            dominated.append(detail)
    continuation_details = []
    for key in spec['continuation']:
        row = rows[key]
        if set(changed[key]) != set(row['candidate_indices']):
            raise RuntimeError('Continuation pair action coverage mismatch')
        new = f.pareto_pair(*(changed[key][i] for i in row['candidate_indices']))
        reverse = new is not None and new[0] != row['preference'][0]
        reversals += reverse
        continuation_details.append(dict(row_id=key, original=row['preference'], changed=new,
                                         reversal=bool(reverse), outcomes=changed[key]))
    result = dict(status='PASS', method=x.METHOD, code_sha=d.read(run/'run_contract.json')['code_sha'],
                  gate=x.diagnosis_gate(dominated, reversals, len(continuation_details)),
                  missing_action_details=details, continuation_details=continuation_details,
                  reversals=reversals, artifacts=artifacts, training_authorized_only_by_gate=True,
                  labels_quarantined_from_training=True)
    # Preferences are tuples in pickle; action-keyed outcomes have int keys.
    # Freeze their JSON form so a completed diagnostic is repeatable on resume.
    result = json.loads(json.dumps(result))
    c.locked_json(run/'diagnose/report.json', result)
    return result


def query_bundle(run, arm, seed, step):
    require_pilot(run)
    request = data.freeze_queries(run, arm, seed, step)
    entries, prechecks = [], []
    blueprints = data.targets(run)
    for mode in x.MODES:
        subset = [r for r in request['requests'] if r['mode'] == mode]
        if subset:
            ids = f.ordered([q['row_id'] for q in subset],
                            f'decision-query-sentinel-{arm}-{seed}-{step}-{mode}')[:8]
            sentinel = make(run, f'query_sentinel_{arm}_s{seed}_t{step}_{mode}', 'train',
                            mode, 'shared_seed0', natural_requests(run, ids),
                            'shared_seed0', 'sentinel')
            references = {blueprints[i]['source_collection']['path'] for i in ids}
            if len(references) != 1:
                raise RuntimeError('Ambiguous query sentinel reference')
            prechecks.append(dict(sentinel=sentinel, reference=blueprints[ids[0]]['source_collection']))
            entries.append(make(run, f'query_{arm}_s{seed}_t{step}_{mode}', 'train', mode,
                                'shared_seed0', subset, 'shared_seed0', 'query'))
    result = dict(status='PASS', collections=entries, prechecks=prechecks, request=request)
    c.locked_json(run/'queries'/f'{arm}_seed{seed}'/f'g{request["generation"]}'/'collections.json', result)
    return result


def build_query_cache(run, arm, seed, generation):
    require_pilot(run)
    folder = run/'queries'/f'{arm}_seed{seed}'/f'g{generation}'
    bundle = d.read(folder/'collections.json')
    labelled, artifacts = {}, [d.artifact(folder/'collections.json')]
    for pair in bundle['prechecks']:
        artifacts.append(compare_sentinel(pair))
    expected = {q['row_id']: q for q in bundle['request']['requests']}
    for entry in bundle['collections']:
        result, audit = outcomes(entry, 'shared_seed0')
        artifacts.extend((entry, audit))
        for key, value in result.items():
            if (key not in expected or key in labelled or
                    value['candidate_index'] != expected[key]['candidate_index'] or
                    value['source_fingerprint'] != expected[key]['source_fingerprint']):
                raise RuntimeError('Wrong arm/action branch label')
            labelled[key] = value['outcome']
    return data.extend_cache(run, arm, seed, generation, labelled, artifacts)
