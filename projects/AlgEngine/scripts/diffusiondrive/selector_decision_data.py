"""Immutable source lineage and arm-isolated, partially labelled feedback caches."""
import copy
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_feedback_common as f
import selector_decision_common as x


def checked_inputs(run):
    value = d.read(run/'decision_inputs.json')
    if value['method'] != x.METHOD or value['status'] != 'PASS':
        raise RuntimeError('Invalid decision-feedback source')
    for entry in value['artifacts'].values():
        d.verify(entry)
    for entry in value['models'].values():
        d.verify(entry)
    for cohort in value['cohorts'].values():
        d.verify(cohort['scenario'])
    return value


def freeze(run, prior):
    from prepare_selector_feedback import model_entry
    from train_selector_feedback import feedback_rows
    run, prior = Path(run).resolve(), Path(prior).resolve()
    if run == prior or prior in run.parents:
        raise RuntimeError('New run must not modify the source run')
    output = run/'decision_inputs.json'
    if output.exists():
        return checked_inputs(run)
    old = d.read(prior/'feedback_inputs.json')
    ledger = d.read(prior/'decision_ledger.json')
    screen = d.read(prior/'screen_report.json')
    if (old['method'] != f.METHOD or ledger['active'] is not None
            or screen['status'] != 'PASS' or screen['decision'] != 'STOP_NO_MECHANISM_SIGNAL'):
        raise RuntimeError('Require the complete, stopped feedback-v2 negative screen')
    rows, lineage = feedback_rows(prior, 'T')
    blueprints = d.read(prior/'feedback/round2/T/targets.json')
    if len(blueprints['targets']) != 64:
        raise RuntimeError('Missing fixed T targets')
    targets = {x.row_id(t): t for t in blueprints['targets']}
    result_rows = []
    excluded = set(old['excluded_logs'])
    for index, original in enumerate(rows):
        row = dict(original, fresh=index >= 192)
        row.update(row_id=x.row_id(row), candidate_bank_sha=x.candidate_bank_hash(row))
        x.check_row(row)
        if row['origin_log'] in excluded:
            raise RuntimeError('Evaluation log in training cache')
        if row['fresh']:
            target = targets[row['row_id']]
            if target['candidate_indices'] != row['candidate_indices']:
                raise RuntimeError('Target/cache candidate mismatch')
            context = c.load_pickle(d.verify(target['context']))
            if f.fingerprint(context) != row['source_fingerprint']:
                raise RuntimeError('Source context fingerprint changed')
            if any(not np.array_equal(np.asarray(row[k]), np.asarray(context[k])) for k in f.FINGERPRINT_FIELDS):
                raise RuntimeError('Feedback features differ from intervention state')
            for entry in target['prefix'].values():
                d.verify(entry)
        result_rows.append(row)
    if len({r['row_id'] for r in result_rows}) != 256:
        raise RuntimeError('Duplicate state/continuation identity')
    cache = run/'data/initial.pkl'
    c.atomic_pickle(cache, dict(status='PASS', method=x.METHOD, rows=result_rows))
    # Legacy inputs/arrays are read-only dependencies, not regenerated training data.
    c.locked_json(run/'feedback_inputs.json', old)
    dense = d.read(prior/'dense/manifest.json')
    if (dense['inputs'] != d.artifact(prior/'feedback_inputs.json')
            or any(r['origin_log'] in excluded for r in dense['records'])):
        raise RuntimeError('Dense replay source/exclusion lineage changed')
    for source in dense['sources'].values():
        for entry in source['arrays'].values():
            d.verify(entry)
    c.locked_json(run/'dense/manifest.json', dense)
    models = dict(old['models'])
    models['shared_seed0'] = model_entry(prior, 'shared_seed0', old)
    models['T_source'] = model_entry(prior, 'T_seed0', old)
    artifacts = {k: d.artifact(prior/p) for k,p in dict(
        source_inputs='feedback_inputs.json', source_contract='run_contract.json',
        source_ledger='decision_ledger.json', source_screen='screen_report.json',
        round1_audit='feedback/round1/shared/cache_audit.json',
        round2_audit='feedback/round2/T/cache_audit.json',
        source_targets='feedback/round2/T/targets.json',
        dense_manifest='dense/manifest.json', parent0='train/shared_seed0/resume.pt',
        parent0_report='train/shared_seed0/report.json').items()}
    artifacts.update({f'lineage_{i}': entry for i,entry in enumerate(lineage)})
    artifacts['initial_cache'] = d.artifact(cache)
    artifacts['local_dense_manifest'] = d.artifact(run/'dense/manifest.json')
    artifacts['local_compatibility_inputs'] = d.artifact(run/'feedback_inputs.json')
    value = dict(status='PASS', method=x.METHOD, prior_run=str(prior), artifacts=artifacts,
                 cohorts=old['cohorts'], models=models, excluded_logs=old['excluded_logs'],
                 exposure=x.EXPOSURE, inference_uses_reward_or_q=False)
    c.locked_json(output, value)
    return value


def initial_rows(run):
    inputs = checked_inputs(run)
    return c.load_pickle(d.verify(inputs['artifacts']['initial_cache']))['rows']


def targets(run):
    return {x.row_id(t): t for t in d.verified_read(
        checked_inputs(run)['artifacts']['source_targets'])['targets']}


def data_entry(run, arm, seed, generation):
    if arm not in x.ARMS or seed not in x.SEEDS or generation not in (0, 1, 2):
        raise RuntimeError('Unknown training-data condition')
    if generation == 0:
        return checked_inputs(run)['artifacts']['initial_cache']
    if arm not in ('P2', 'P3'):
        raise RuntimeError('Static controls cannot consume extra labels')
    audit = d.read(run/'queries'/f'{arm}_seed{seed}'/f'g{generation}'/'cache_audit.json')
    if (audit['status'] != 'PASS' or audit['arm'] != arm or audit['seed'] != seed
            or audit['generation'] != generation
            or audit['parent'] != data_entry(run, arm, seed, generation-1)):
        raise RuntimeError('Arm/seed/generation cache ownership drift')
    for entry in audit['artifacts']:
        d.verify(entry)
    return audit['cache']


def load_rows(run, arm, seed, generation):
    entry = data_entry(run, arm, seed, generation)
    rows = c.load_pickle(d.verify(entry))['rows']
    if len(rows) != 256 or sum(r['fresh'] for r in rows) != 64:
        raise RuntimeError('Feedback row count changed')
    for row in rows:
        x.check_row(row)
    return rows, entry


def scores(model, rows, device='cpu'):
    import torch
    with torch.no_grad():
        return torch.cat([m.score(model, rows, list(range(i, min(i+16, len(rows)))), device, 'residual')
                          for i in range(0, len(rows), 16)]).cpu().numpy()


def preflight(run, device, full_baselines=False):
    """Actual cached new-objective backward; never take an optimizer step."""
    import torch
    from preflight_selector_feedback import preflight as legacy_preflight
    torch.manual_seed(20260912)
    legacy = legacy_preflight(run, device, full_baselines)
    inputs = checked_inputs(run)
    manifest = d.verified_read(inputs['artifacts']['source_inputs'])['artifacts']['scalar_manifest']
    model, _, _ = m.load_incumbent(d.verify(manifest))
    model.to(device).train().requires_grad_(True)
    rows = [r for r in initial_rows(run) if r['fresh']]
    probe = [r for mode in x.MODES for r in [v for v in rows if v['mode']==mode][:8]]
    logits = m.score(model,probe,list(range(16)),device,'residual')
    pair,nll = x.feedback_loss(logits,probe)
    (pair+nll).backward()
    norm = m.v3.clip_grad_norm_cpu_(model.parameters(),10.)
    if not torch.isfinite(pair+nll) or not np.isfinite(norm) or norm <= 0:
        raise RuntimeError('New objective real-cache backward failed')
    result = dict(status='PASS', device=device, code_sha=d.read(run/'run_contract.json')['code_sha'],
                  legacy=d.artifact(run/f'cached_preflight_{device}.json'),
                  inputs=d.artifact(run/'decision_inputs.json'), actual_new_loss=float((pair+nll).detach()),
                  gradient_norm=float(norm), optimizer_steps=0, rollout_performed=False,
                  training_performed=False, labels_from_training_only=True)
    c.atomic_json(run/f'decision_preflight_{device}.json',result)
    return result


def model_entry(run, policy):
    inputs = checked_inputs(run)
    if policy in inputs['models']:
        entry = inputs['models'][policy]
        d.verify(entry)
        return entry
    report = d.read(run/'train'/policy/'step_500_report.json')
    contract = d.read(run/'run_contract.json')
    if (report['status'] != 'PASS' or report['provenance']['method'] != x.METHOD
            or report['provenance']['code_sha'] != contract['code_sha']
            or report['provenance']['policy'] != policy or report['step'] != 500):
        raise RuntimeError('Invalid learned model provenance')
    d.verify(report['selector'])
    return dict(report['selector'], kind='cfpi', score_mode='residual', provenance=report['provenance'])


def freeze_diagnostic(run):
    import torch
    torch.set_num_threads(4)
    output = run/'diagnose/targets.json'
    if output.exists():
        value = d.read(output)
        for entry in value['artifacts']:
            d.verify(entry)
        return value
    rows = [r for r in initial_rows(run) if r['fresh']]
    entry = model_entry(run, 'T_source')
    model, _ = m.load_selector(d.verify(entry))
    z = scores(model, rows)
    unsupported, continuation, summary = [], [], {}
    for mode in x.MODES:
        indices = [i for i,r in enumerate(rows) if r['mode'] == mode]
        informative = [i for i in indices if rows[i]['preference'] is not None]
        correct, selected = 0, 0
        for i in informative:
            winner, loser, _ = rows[i]['preference']
            a,b = (rows[i]['candidate_indices'][k] for k in (winner,loser))
            correct += int(z[i,a] > z[i,b])
            selected += int(int(z[i].argmax()) == a)
        for i in indices:
            action = int(z[i].argmax())
            if action not in rows[i]['candidate_indices']:
                unsupported.append(dict(row_id=rows[i]['row_id'], mode=mode,
                                        scene_id=rows[i]['scene_id'], candidate_index=action))
        selected_ids = f.ordered([rows[i]['row_id'] for i in informative],
                                 'decision-feedback-continuation-'+mode)[:8]
        if len(selected_ids) != 8:
            raise RuntimeError('Insufficient informative continuation targets; no quota relaxation')
        continuation.extend(selected_ids)
        summary[mode] = dict(rows=len(indices), informative=len(informative), pair_correct=correct,
                             full_winner=selected, unsupported=sum(r['mode']==mode for r in unsupported))
    value = dict(status='PASS', unsupported=unsupported, continuation=continuation,
                 summary=summary, artifacts=[entry, checked_inputs(run)['artifacts']['initial_cache']],
                 labels_quarantined_from_training=True)
    c.locked_json(output, value)
    return value


def freeze_queries(run, arm, seed, step):
    import torch
    torch.set_num_threads(4)
    generation = x.QUERY_STEPS.index(step)+1
    folder = run/'queries'/f'{arm}_seed{seed}'/f'g{generation}'
    output = folder/'requests.json'
    rows, parent = load_rows(run, arm, seed, generation-1)
    checkpoint_report = d.read(run/'train'/f'{arm}_seed{seed}'/f'step_{step}_report.json')
    if (checkpoint_report['status'] != 'PASS' or checkpoint_report['step'] != step
            or checkpoint_report['provenance']['policy'] != f'{arm}_seed{seed}'
            or checkpoint_report['provenance']['code_sha'] != d.read(run/'run_contract.json')['code_sha']):
        raise RuntimeError('Query checkpoint identity changed')
    model_path = d.verify(checkpoint_report['selector'])
    if output.exists():
        value = d.read(output)
        if (value['parent'] != parent or value['checkpoint'] != checkpoint_report['selector']
                or value['arm'] != arm or value['seed'] != seed or value['step'] != step):
            raise RuntimeError('Frozen query provenance changed')
        for entry in value['artifacts']:
            d.verify(entry)
        return value
    artifacts = [parent, checkpoint_report['selector']]
    if arm == 'P3':
        model, _ = m.load_selector(model_path)
        requests = x.query_plan(rows, scores(model, rows), arm, seed, step)
    else:
        active_entry = d.artifact(run/'queries'/f'P3_seed{seed}'/f'g{generation}'/'requests.json')
        active = d.verified_read(active_entry)
        requests = x.query_plan(rows, None, arm, seed, step, active['requests'])
        artifacts.append(active_entry)  # schedule only, NEVER active outcomes/cache
    value = dict(status='PASS', method=x.METHOD, arm=arm, seed=seed, step=step,
                 generation=generation, parent=parent, checkpoint=checkpoint_report['selector'],
                 requests=requests, artifacts=artifacts, logical_queries=len(requests))
    c.locked_json(output, value)
    return value


def extend_cache(run, arm, seed, generation, labelled, outcome_artifacts):
    """labelled maps EXACT requested row IDs to their one newly queried outcome."""
    folder = run/'queries'/f'{arm}_seed{seed}'/f'g{generation}'
    request_entry = d.artifact(folder/'requests.json')
    request = d.verified_read(request_entry)
    rows, parent = load_rows(run, arm, seed, generation-1)
    if request['parent'] != parent or request['arm'] != arm or request['seed'] != seed:
        raise RuntimeError('Query data ownership mismatch')
    if set(labelled) != {r['row_id'] for r in request['requests']}:
        raise RuntimeError('Missing/extra labels; diagnostics and other arms are forbidden')
    updated = copy.deepcopy(rows)
    lookup = {r['row_id']: r for r in updated}
    for q in request['requests']:
        row = lookup[q['row_id']]
        if (not row['fresh'] or q['candidate_index'] in row['candidate_indices']
                or q['candidate_bank_sha'] != row['candidate_bank_sha']
                or q['source_fingerprint'] != row['source_fingerprint']
                or q['continuation_policy'] != row['continuation_policy']):
            raise RuntimeError('Query/context/continuation mismatch')
        row['candidate_indices'].append(q['candidate_index'])
        row['branch_outcomes'].append(labelled[q['row_id']])
        x.check_row(row)
    output = folder/'cache_audit.json'
    artifacts = [request_entry, parent, *outcome_artifacts]
    if output.exists():
        audit = d.read(output)
        if audit['artifacts'] != artifacts:
            raise RuntimeError('Cache label lineage drift')
        d.verify(audit['cache'])
        return audit
    path = folder/'cache.pkl'
    c.atomic_pickle(path, dict(status='PASS', method=x.METHOD, arm=arm, seed=seed,
                              generation=generation, rows=updated))
    result = dict(status='PASS', arm=arm, seed=seed, generation=generation,
                  parent=parent, cache=d.artifact(path), artifacts=artifacts,
                  logical_queries=len(labelled), no_missing_label_imputation=True)
    c.locked_json(output, result)
    return result
