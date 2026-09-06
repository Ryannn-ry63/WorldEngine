"""Fixed posthoc OOF/refit comparison on identical cached states; no fitting."""
import argparse
from pathlib import Path
import numpy as np
import torch
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_rare_common as r
from selector_cfpi_deployment_router import load_bank, selector_scores


def diagnose(run, device):
    torch.set_num_threads(4)
    inputs = d.read(run/'rare_inputs.json')
    original = d.verified_read(inputs['artifacts']['old_inputs'])
    rows = c.validate_cache(c.load_pickle(d.verify(inputs['artifacts']['cache'])))
    result = {}
    for policy in r.BRIDGE:
        bank = {fold:load_bank(inputs['models'][f'{policy}_fold{fold}']).to(device) for fold in range(4)}
        refit = load_bank(inputs['models'][policy]).to(device)
        values = []
        for row in rows:
            a = selector_scores(bank[row['fold']],row,device,'residual')
            b = selector_scores(refit,row,device,'residual')
            labels = row['local_official_pdm' if policy.startswith('local') else 'branch_returns']
            if int(a.argmax())!=original['oof_predictions'][policy][row['scene_id']]:
                raise RuntimeError('Cached OOF action differs from original frozen prediction')
            advantage = (labels-np.mean(labels))/max(float(np.std(labels)),1e-6)
            def objective(scores):
                probs = np.exp(scores-scores.max())
                return -float(np.dot(probs/probs.sum(),advantage))
            values.append(dict(scene_id=row['scene_id'],fold=row['fold'],oof_action=int(a.argmax()),refit_action=int(b.argmax()),
                               score_max_abs=float(abs(a-b).max()),oof_selected_label=float(labels[a.argmax()]),
                               refit_selected_label=float(labels[b.argmax()]),
                               oof_exact_group_loss=objective(a),refit_exact_group_loss=objective(b),
                               oof_scores=a.tolist(),refit_scores=b.tolist()))
        result[policy] = dict(rows=values,action_disagreement=float(np.mean([x['oof_action']!=x['refit_action'] for x in values])))
    c.locked_json(run/'bridge_cached_diagnostic.json',dict(status='PASS',posthoc=True,conditions=result,
                  note='Refit values are in-sample; not a generalization estimate.',inputs=d.artifact(run/'rare_inputs.json')))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    diagnose(a.run_root,a.device)
