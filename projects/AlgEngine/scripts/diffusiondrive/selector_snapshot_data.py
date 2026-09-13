"""Archive and export without training or inventing legacy training metadata."""
from pathlib import Path
import json
import numpy as np
import torch
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_snapshot_common as s
from materialize_grpo_selector_v3 import state_dict


def export_model(scalar_path, selector_path, output, provenance):
    checkpoint = torch.load(scalar_path, map_location='cpu')
    original = state_dict(checkpoint)
    payload = torch.load(selector_path, map_location='cpu')
    if payload.get('score_mode') != 'residual' or payload.get('inference_uses_reward_or_q') is not False:
        raise RuntimeError('Only native residual selector export is authorized')
    if payload['provenance']['policy'] != 'P1_seed2':
        raise RuntimeError('Wrong release policy')
    weights = payload['scene_selector_state']
    old_keys = {k for k in original if k.startswith(s.PREFIX)}
    if old_keys != {s.PREFIX+k for k in weights} or len(old_keys) != 54:
        raise RuntimeError('Selector key set changed')
    for key, value in weights.items():
        if value.shape != original[s.PREFIX+key].shape or not torch.isfinite(value).all():
            raise RuntimeError('Invalid exported selector tensor')
        original[s.PREFIX+key] = value.detach().cpu().clone()
    # Replace obsolete model-specific metadata, not the frozen reference weights.
    checkpoint.setdefault('meta', {}).pop('diffusiondrive_grpo_selector_v3', None)
    checkpoint['meta']['selector_snapshot'] = dict(method=s.METHOD, **provenance,
        scene_selector_config=payload['scene_selector_config'], source_training=payload['provenance'])
    output = Path(output)
    if output.exists():
        existing = torch.load(output, map_location='cpu')
        if existing.get('meta') != checkpoint.get('meta') or set(state_dict(existing)) != set(original):
            raise RuntimeError('Native checkpoint metadata/key drift')
        if any(not torch.equal(v, state_dict(existing)[k]) for k,v in original.items()):
            raise RuntimeError('Native checkpoint tensor drift')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(output)
    frozen = state_dict(torch.load(scalar_path, map_location='cpu'))
    nonselector = [k for k in frozen if not k.startswith(s.PREFIX)]
    if len(nonselector) != 963 or any(not torch.equal(frozen[k], original[k]) for k in nonselector):
        raise RuntimeError('Export changed frozen generator/perception/base selector')
    return dict(status='PASS', unchanged_nonselector_tensors=len(nonselector),
                selector_tensors=len(weights), checkpoint=d.artifact(output))


def freeze(run, source, canonical):
    run, source, canonical = map(Path, (run, source, canonical))
    report = d.read(source/'pilot_report.json')
    if report['status'] != 'PASS' or d.read(source/'decision_ledger.json')['active'] is not None:
        raise RuntimeError('Source must be complete and idle')
    trained = d.read(source/'train/P1_seed2/step_500_report.json')
    if (trained['status'] != 'PASS' or trained['step'] != 500
            or trained['provenance']['policy'] != 'P1_seed2'
            or trained['selector']['sha256'] != s.SELECTOR_SHA):
        raise RuntimeError('Wrong selected checkpoint')
    inherited = d.read(source/'decision_inputs.json')
    prior_inputs = d.verified_read(inherited['artifacts']['source_inputs'])
    scalar_manifest = d.verified_read(prior_inputs['artifacts']['scalar_manifest'])
    if scalar_manifest['checkpoint_sha256'] != c.CHECKPOINT_SHA256:
        raise RuntimeError('Wrong paired V3')
    files = dict(source_report=source/'pilot_report.json', source_contract=source/'run_contract.json',
                 source_ledger=source/'decision_ledger.json', source_inputs=source/'decision_inputs.json',
                 train_report=source/'train/P1_seed2/step_500_report.json',
                 selector=d.verify(trained['selector']), resume=d.verify(trained['resume']),
                 initial_cache=d.verify(inherited['artifacts']['initial_cache']),
                 scalar_manifest=d.verify(prior_inputs['artifacts']['scalar_manifest']))
    copied = {k:s.preserved_copy(v,run/'archive'/k/Path(v).name) for k,v in files.items()}
    scalar = s.preserved_copy(scalar_manifest['checkpoint'],run/'models/scalar_v3/checkpoint.pth')
    if scalar['sha256'] != scalar_manifest['checkpoint_sha256']:
        raise RuntimeError('Copied V3 checkpoint differs from frozen manifest')
    native = export_model(d.verify(scalar),d.verify(copied['selector']),
                          run/'models/P1_seed2/checkpoint.pth',
                          dict(source_selector_sha=s.SELECTOR_SHA, exposure=s.EXPOSURE))
    c.locked_json(run/'archive/native_export_audit.json',native)
    # Freeze exact historical membership; aggregate rows are NOT samples.
    historical = canonical/'experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_tuned_s0/summary.json'
    hs = d.read(historical)
    copied['historical_summary'] = d.artifact(historical)
    for key in ('openloop_navtest_pdm','openloop_navtest_ade','openloop_failures_pdm','closedloop_nr','closedloop_r'):
        copied['historical_'+key] = hs['input_files'][key]
        d.verify(copied['historical_'+key])
    nav = s.real_rows(d.verify(copied['historical_openloop_navtest_pdm']),valid=True)
    rare = s.real_rows(d.verify(copied['historical_openloop_failures_pdm']),valid=True)
    if len(nav) != 12146 or len(rare) != 288:
        raise RuntimeError('Historical real-token coverage drift')
    split = d.verified_read(prior_inputs['artifacts']['rare_split'])
    full_path = Path(split['source'])
    if c.sha256_file(full_path) != split['source_sha256']:
        raise RuntimeError('Full scenario drift')
    full = c.load_pickle(full_path)
    # The simulator consumes a dict keyed by scene ID.
    if not isinstance(full,dict) or len(full) != 288:
        raise RuntimeError('Unexpected full scenario format')
    rows = inherited['cohorts']['development']['rows']
    dev_ids = {v['scene_id'] for v in rows}
    common = inherited['cohorts']['common']
    common_scene = common.get('scenario',common.get('scenario_file'))
    if not isinstance(common_scene,dict):
        raise RuntimeError('Missing frozen common scenario')
    copied.update(full_scenario=d.artifact(full_path), common_scenario=common_scene,
                  split=d.artifact(Path(prior_inputs['artifacts']['rare_split']['path'])),
                  export_audit=d.artifact(run/'archive/native_export_audit.json'))
    if not dev_ids <= set(full):
        raise RuntimeError('Development is not contained in full288')
    # Source cohort log identities are authoritative; full split includes log names in IDs.
    full_rows = [dict(scene_id=k,origin_log=k.rsplit('-',1)[0]) for k in sorted(full)]
    bridge_ids = sorted(dev_ids,key=lambda k:s.digest(['snapshot_bridge_v1',k]))[:8]
    bridge_path = run/'archive/bridge8.pkl'
    if bridge_path.exists():
        if set(c.load_pickle(bridge_path)) != set(bridge_ids):
            raise RuntimeError('Bridge identity drift')
    else:
        c.atomic_pickle(bridge_path,{k:full[k] for k in bridge_ids})
    copied['bridge_scenario'] = d.artifact(bridge_path)
    inputs = dict(method=s.METHOD, source=str(source), exposure=s.EXPOSURE,
        source_decision=report['decision'], artifacts=copied,
        models=dict(scalar_v3=scalar,P1_seed2=native['checkpoint']),
        cohorts=dict(full=full_rows,development=rows,common=common['rows'],
                     bridge=[v for v in rows if v['scene_id'] in bridge_ids]),
        open_tokens=dict(open_navtest=sorted(v['token'] for v in nav),
                         open_rare=sorted(v['token'] for v in rare)),
        assets=dict(full=str(canonical/'data/sim_engine/assets/navtest_failures/assets'),
                    bridge=str(canonical/'data/sim_engine/assets/navtest_failures/assets'),
                    common=str(canonical/'data/sim_engine/assets/navtrain/assets')))
    for mode in ('NR','R'):
        for policy in s.POLICIES:
            p=source/'collections'/f'eval_development_{mode}_{policy}'/'collection_audit.json'
            inputs['artifacts'][f'bridge_reference_{policy}_{mode}']=d.artifact(p)
    c.locked_json(run/'inputs.json',inputs)
    return inputs


def parity(run, device='cpu'):
    torch.set_num_threads(4)
    inputs=s.checked_inputs(run)
    model,payload=m.load_selector(d.verify(inputs['artifacts']['selector']))
    native=state_dict(torch.load(d.verify(inputs['models']['P1_seed2']),map_location='cpu'))
    restored=m.v3.model_from_config(payload['scene_selector_config'])
    restored.load_state_dict({k[len(s.PREFIX):]:v for k,v in native.items() if k.startswith(s.PREFIX)},strict=True)
    model.to(device).eval();restored.to(device).eval()
    cache=c.load_pickle(d.verify(inputs['artifacts']['initial_cache']))
    rows=cache['rows'] if isinstance(cache,dict) else cache
    with torch.no_grad():
        a=m.score(model,rows,list(range(16)),device,'residual')
        b=m.score(restored,rows,list(range(16)),device,'residual')
    error=float((a-b).abs().max())
    if not np.isfinite(error) or error>c.MODEL_RECOMPUTE_TOLERANCE or not torch.equal(a.argmax(-1),b.argmax(-1)):
        raise RuntimeError('Native/cache selector parity failed')
    out=dict(status='PASS',device=device,score_max_abs=error,argmax_mismatches=0,
             optimizer_steps=0,inputs=d.artifact(Path(run)/'inputs.json'))
    c.locked_json(Path(run)/f'cached_parity_{device}.json',out)
    return out
