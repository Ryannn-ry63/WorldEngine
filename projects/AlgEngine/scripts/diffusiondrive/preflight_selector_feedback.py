"""CPU/GPU cached software checks, never a training run or efficacy claim."""
import argparse
from pathlib import Path
import numpy as np
import torch
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_feedback_common as f
from selector_cfpi_deployment_router import load_bank,selector_scores
from train_selector_feedback import pair_loss


def preflight(run, device, full_baselines=False):
    inputs = d.read(run/'feedback_inputs.json')
    torch.set_num_threads(4)
    if full_baselines:
        from prepare_selector_cfpi_deployment import verify_baselines
        c.locked_json(run/'deployment_inputs.json',inputs)
        verify_baselines(run,inputs)
    old = d.verified_read(inputs['artifacts']['prior_inputs'])
    # Real prior feedback cache tests visible inputs only; none of its Q enters new training.
    source = c.load_pickle(d.verify(old['artifacts']['cache']))
    rows = c.validate_cache(source)[:8]
    model,config,_ = m.load_incumbent(d.verify(inputs['artifacts']['scalar_manifest']))
    model.to(device)
    bank = load_bank(inputs['models']['scalar_v3']).to(device)
    maximum = 0.
    with torch.no_grad():
        direct = m.score(model,rows,list(range(8)),device,'residual').cpu().numpy()
        for i,row in enumerate(rows):
            routed = selector_scores(bank,row,device,'residual')
            maximum = max(maximum,float(np.max(np.abs(routed-direct[i]))))
            if routed.argmax()!=direct[i].argmax():
                raise RuntimeError('Real-cache bank action mismatch')
    if maximum>c.MODEL_RECOMPUTE_TOLERANCE:
        raise RuntimeError('Real-cache bank scoring drift')
    clone = m.initialize(model,'residual').requires_grad_(True).to(device)
    with torch.no_grad():
        if not torch.equal(m.score(model,rows,list(range(8)),device,'residual'),
                           m.score(clone,rows,list(range(8)),device,'residual')):
            raise RuntimeError('Initial V3 identity failed')
    logits = torch.tensor([[-1000.,1000.]+[0.]*18],requires_grad=True,device=device)
    loss = pair_loss(logits,[dict(candidate_indices=[0,1],preference=(0,1,.5))])
    loss.backward()
    if not torch.isfinite(loss) or abs(float(logits.grad[0,0]))<.49:
        raise RuntimeError('Suppressed preferred action gradient failed')
    from train_selector_feedback import Dense
    from selector_rare_model import teacher_kl
    dense = Dense(run).sample(np.random.default_rng(20260907))
    clone.train()
    student = m.score(clone,dense,list(range(16)),device,'residual')
    with torch.no_grad():
        teacher = m.score(model,dense,list(range(16)),device,'residual')
    rewards = torch.as_tensor(np.stack([r['candidate_rewards'] for r in dense]),dtype=torch.float32,device=device)
    valid = torch.as_tensor(np.stack([r['candidate_reward_valid_mask'] for r in dense]),dtype=torch.bool,device=device)
    real_loss = m.v3.exact_group_loss(student,teacher,rewards,valid,1.,0.)[0]+.01*teacher_kl(student,teacher)
    real_loss.backward()
    gradient_norm = m.v3.clip_grad_norm_cpu_(clone.parameters(),10.)
    if not np.isfinite(gradient_norm):
        raise RuntimeError('Real dense backward produced nonfinite gradients')
    # No optimizer step, no exported model, no feedback outcome used for this check.
    result = dict(status='PASS',device=device,inputs=d.artifact(run/'feedback_inputs.json'),
                  code_sha=d.read(run/'run_contract.json')['code_sha'],bank_max_abs=maximum,
                  real_contexts=8,initial_identity=True,finite_extreme_pair_gradient=True,
                  real_dense_backward_finite=True,real_dense_gradient_norm=gradient_norm,
                  training_performed=False,optimizer_steps=0,rollout_performed=False)
    c.atomic_json(run/f'cached_preflight_{device}.json',result)
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--full-baselines',action='store_true')
    a = p.parse_args()
    print(preflight(a.run_root,a.device,a.full_baselines),flush=True)
