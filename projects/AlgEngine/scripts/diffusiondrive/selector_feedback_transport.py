"""Explicit training-only intervention. Never forge policy logits for a forced action."""
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
from oracle_r15_common import max_abs_error


def check_prefix(context, target, decision, selected=None):
    if decision > target['decision_step']:
        return
    original = c.load_pickle(d.verify(target['prefix'][str(decision)]))
    error = max(max_abs_error(context[k],original[k]) for k in f.FINGERPRINT_FIELDS)
    if error > c.ARRAY_TOLERANCE:
        raise RuntimeError(f'Feedback branch visible prefix/state drift: {error}')
    if decision < target['decision_step'] and selected is not None and selected != int(original['selected_index']):
        raise RuntimeError('Feedback visiting policy changed prefix action')
    if decision == target['decision_step'] and f.fingerprint(original) != target['source_fingerprint']:
        raise RuntimeError('Frozen target fingerprint drift')


def apply_intervention(result, route, decision, natural, expand, device):
    target = route.get('feedback_target')
    if not target:
        return natural, None
    context = result['diffusiondrive_rollout_context']
    check_prefix(context,target,decision,natural)
    if decision != target['decision_step']:
        return natural, None
    forced = target['forced_index']
    if type(forced) is not int or not 0<=forced<20:
        raise RuntimeError('Intervention candidate outside original20')
    if natural != target['visiting_index']:
        raise RuntimeError('Actual visiting action differs from source state')
    import torch
    trajectory = torch.as_tensor(context['candidate_trajectories_8'][forced:forced+1],dtype=torch.float32,device=device)
    result['trajectory'] = expand(trajectory)[0].cpu().numpy()
    result['chosen_ind'] = forced
    context['selected_indices'] = np.asarray(forced,dtype=np.int64)
    result.pop('ade_4s',None); result.pop('fde_4s',None)
    return forced, dict(training_only=True, natural_selected_index=natural, forced_index=forced,
                        source_fingerprint=target['source_fingerprint'],
                        source_collection_sha256=target['source_collection']['sha256'],
                        sentinel=target['sentinel'])


def validate_feedback(sidecar, collection, route):
    mode = collection['react_type']
    if mode not in f.MODES:
        raise RuntimeError('Invalid feedback react_type')
    contract = sidecar['selector_rollout_contract']
    cohort = collection['cohort']
    expected = dict(research_method=f.METHOD,react_type=mode,source_data_split=cohort,
                    development_consumed=cohort=='development',test_consumed=False,
                    legacy_exposed_benchmark=True,independent_unseen_test=False)
    if any(contract.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('Feedback mode/exposure contract drift')
    target = route.get('feedback_target')
    if target and cohort!='train':
        raise RuntimeError('Interventions forbidden on evaluation scenes')
    step, selected = int(sidecar['planner_step']), int(sidecar['selected_index'])
    intervention = sidecar['cfpi_deployment'].get('feedback_intervention')
    if not target or step != target['decision_step']:
        if intervention is not None:
            raise RuntimeError('Unscheduled or repeated intervention')
        if target:
            check_prefix(sidecar,target,step,selected)
        return False
    check_prefix(sidecar,target,step)
    natural = int(np.asarray(sidecar['current_logits']).argmax())
    expected = dict(training_only=True,natural_selected_index=target['visiting_index'],
                    forced_index=target['forced_index'],source_fingerprint=target['source_fingerprint'],
                    source_collection_sha256=target['source_collection']['sha256'],sentinel=target['sentinel'])
    if intervention != expected or natural != target['visiting_index'] or selected != target['forced_index']:
        raise RuntimeError('Forced/natural action identity mismatch')
    return True
