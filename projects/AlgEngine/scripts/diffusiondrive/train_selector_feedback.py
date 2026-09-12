"""Same dense GRPO, pair loss and retention for all feedback-selection conditions."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_feedback_common as f
from selector_rare_model import teacher_kl


def pair_loss(logits, rows):
    terms = []
    for i,row in enumerate(rows):
        p = row['preference']
        if p is None:
            continue
        winner,loser,gap = p
        a,b = row['candidate_indices'][winner],row['candidate_indices'][loser]
        terms.append(float(gap)*F.softplus(logits[i,b]-logits[i,a]))
    # Normalize by ALL rows, not only informative pairs: no adaptive reweighting.
    return torch.stack(terms).sum()/len(rows) if terms else logits.sum()*0.


class Dense:
    def __init__(self, run):
        self.manifest = d.read(run/'dense/manifest.json')
        self.arrays = {name:{k:np.load(d.verify(v),mmap_mode='r',allow_pickle=False)
                            for k,v in source['arrays'].items()} for name,source in self.manifest['sources'].items()}
        self.records = self.manifest['records']
        if not self.records:
            raise RuntimeError('Empty dense replay')

    def sample(self, rng):
        rows = []
        for group in ('common','hard'):
            for index in rng.integers(len(self.records),size=8):
                variants = self.records[int(index)][group]
                source,offset = variants[int(rng.integers(len(variants)))]
                rows.append({k:np.asarray(v[offset]) for k,v in self.arrays[source].items()})
        return rows


def feedback_rows(run, arm):
    audits = [run/'feedback/round1/shared/cache_audit.json']
    if arm!='shared':
        audits.append(run/'feedback/round2'/arm/'cache_audit.json')
    rows, artifacts = [], []
    for path in audits:
        audit = d.read(path)
        if audit['status']!='PASS':
            raise RuntimeError('Feedback cache not audited')
        for item in audit['artifacts']:
            d.verify(item)
        cache = c.load_pickle(d.verify(audit['cache']))
        rows.extend(cache['rows'])
        artifacts.extend((d.artifact(path),audit['cache']))
    if len(rows)!=(192 if arm=='shared' else 256):
        raise RuntimeError('Feedback row count drift')
    excluded = set(d.read(run/'feedback_inputs.json')['excluded_logs'])
    if any(row['origin_log'] in excluded for row in rows):
        raise RuntimeError('Evaluation log in feedback')
    return rows,artifacts


def train(run, arm, seed, device):
    if arm not in ('shared',*f.ARMS) or seed not in (0,1,2):
        raise RuntimeError('Unregistered training')
    if arm in ('S','U') and seed!=0:
        raise RuntimeError('Only fixed controls seed0 are authorized')
    if seed>0:
        from report_selector_feedback import report
        if report(run,'screen').get('decision')!='PROCEED_T_SEEDS_1_2':
            raise RuntimeError('Additional seeds require the complete passing T screen')
    inputs = d.read(run/'feedback_inputs.json')
    feedback, sources = feedback_rows(run,arm)
    policy = f'{arm}_seed{seed}'
    folder = run/'train'/policy
    provenance = dict(method=f.METHOD,arm=arm,seed=seed,steps=500,settings=f.TRAIN,
                      inputs_sha256=c.sha256_file(run/'feedback_inputs.json'),
                      code_sha=d.read(run/'run_contract.json')['code_sha'],
                      feedback=sources,dense=d.artifact(run/'dense/manifest.json'),
                      feedback_collection_seed=0,total_optimizer_steps=500 if arm=='shared' else 1000)
    parent = run/'train'/f'shared_seed{seed}'/'resume.pt'
    if arm!='shared':
        parent_report = d.read(parent.parent/'report.json')
        if parent_report['status']!='PASS':
            raise RuntimeError('First round incomplete')
        provenance['parent'] = d.artifact(parent)
        provenance['parent_report'] = d.artifact(parent.parent/'report.json')
    c.locked_json(folder/'contract.json',provenance)
    if (folder/'report.json').exists():
        report = d.read(folder/'report.json')
        if report['status']!='PASS' or report['provenance']!=provenance:
            raise RuntimeError('Training resume provenance drift')
        d.verify(report['selector'])
        return report
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('No silent CPU fallback')
    incumbent,config,_ = m.load_incumbent(d.verify(inputs['artifacts']['scalar_manifest']))
    incumbent.to(device).eval().requires_grad_(False)
    model = m.initialize(incumbent,'residual').to(device).requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng,start = np.random.default_rng(seed),0
    if arm!='shared':
        state = torch.load(d.verify(provenance['parent']),map_location=device)
        if state['step']!=500 or state['provenance']['arm']!='shared' or state['provenance']['seed']!=seed:
            raise RuntimeError('Wrong round1 optimizer parent')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
    resume = folder/'resume.pt'
    losses = {}
    if resume.exists():
        state = torch.load(resume,map_location=device)
        if state['provenance']!=provenance:
            raise RuntimeError('Resume settings differ')
        model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
        start = state['step']
        losses = state['losses']
    dense = Dense(run)
    strata = {mode:[r for r in feedback if r['mode']==mode] for mode in f.MODES}
    model.train()
    for step in range(start+1,501):
        dense_rows = dense.sample(rng)
        pairs = [strata[mode][int(i)] for mode in f.MODES for i in rng.integers(len(strata[mode]),size=8)]
        batch = dense_rows+pairs
        student = m.score(model,batch,list(range(32)),device,'residual')
        with torch.no_grad():
            teacher = m.score(incumbent,batch,list(range(32)),device,'residual')
        rewards = torch.as_tensor(np.stack([r['candidate_rewards'] for r in dense_rows]),dtype=torch.float32,device=device)
        valid = torch.as_tensor(np.stack([r['candidate_reward_valid_mask'] for r in dense_rows]),dtype=torch.bool,device=device)
        policy_loss = m.v3.exact_group_loss(student[:16],teacher[:16],rewards,valid,1.,0.)[0]
        preference = pair_loss(student[16:],pairs)
        kl = teacher_kl(student,teacher)
        loss = policy_loss+preference+.01*kl
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training objective')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        m.v3.clip_grad_norm_cpu_(model.parameters(),10.)
        optimizer.step()
        losses = dict(total=float(loss.detach()),dense=float(policy_loss.detach()),
                      preference=float(preference.detach()),teacher_kl=float(kl.detach()))
        if step%25==0:
            temporary = resume.with_suffix('.tmp')
            torch.save(dict(provenance=provenance,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
                            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,losses=losses),temporary)
            temporary.replace(resume)
            print(dict(policy=policy,step=step,**losses),flush=True)
    model.eval()
    path = folder/'step_500.pt'
    m.save_selector(path,model,config,'residual',provenance)
    restored,_ = m.load_selector(path)
    restored.to(device)
    with torch.no_grad():
        probe = feedback[:16]
        a = m.score(model,probe,list(range(16)),device,'residual')
        b = m.score(restored,probe,list(range(16)),device,'residual')
        error = float((a-b).abs().max())
        if error>c.ARRAY_TOLERANCE or not torch.equal(a.argmax(-1),b.argmax(-1)):
            raise RuntimeError('Export score/action parity failed')
    report = dict(status='PASS',provenance=provenance,selector=d.artifact(path),losses=losses,
                  trainable_names=sorted(k for k,_ in model.named_parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters()),export_max_abs=error,
                  inference_uses_reward_or_q=False)
    c.locked_json(folder/'report.json',report)
    return report


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--arm',choices=('shared',*f.ARMS),required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    train(a.run_root,a.arm,a.seed,a.device)
