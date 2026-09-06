"""Three fixed arms; balanced replay; only final step500 is deployable."""
import argparse
from pathlib import Path
import numpy as np
import torch
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_rare_common as r
import selector_rare_model as rare
from selector_cfpi_objectives import exact_group
from train_selector_cfpi import validate_incumbent_recompute


def train(run, method, seed, device):
    inputs = d.read(run/'rare_inputs.json')
    if inputs['status']!='PASS' or method not in r.POLICIES or seed not in r.SEEDS:
        raise RuntimeError('Unregistered training')
    files = inputs['artifacts']
    rows = c.validate_cache(c.load_pickle(d.verify(files['cache'])))
    replay = c.load_pickle(d.verify(files['replay']))['rows']
    if len(rows)!=64 or len(replay)!=512:
        raise RuntimeError('Training coverage changed')
    folder = run/'train'/f'{method}_seed{seed}'
    provenance = dict(method=method,seed=seed,step=500,settings=r.TRAIN,
                      inputs_sha256=c.sha256_file(run/'rare_inputs.json'),
                      code_sha=d.read(run/'run_contract.json')['code_sha'])
    c.locked_json(folder/'contract.json',provenance)
    if (folder/'report.json').exists():
        report = d.read(folder/'report.json')
        if report['provenance']!=provenance or report['status']!='PASS':
            raise RuntimeError('Resume report drift')
        d.verify(report['selector'])
        return
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('No CPU fallback for formal training')
    incumbent, config, _ = m.load_incumbent(d.verify(files['scalar_manifest']))
    incumbent.to(device).eval().requires_grad_(False)
    parity = validate_incumbent_recompute(incumbent,rows,device)
    model = rare.initialize(incumbent,method).to(device).train()
    trainable = {k for k,p in model.named_parameters() if p.requires_grad}
    if method!='q_full' and any(not k.startswith('adapter.') for k in trainable):
        raise RuntimeError('Frozen V3 scope violated')
    frozen = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
              if method!='q_full' and k.startswith('incumbent.')}
    # Exact zero initialization on the identical cached scoring path.
    with torch.no_grad():
        model.eval()
        before = m.score(incumbent,rows,list(range(16)),device,'residual')
        initial = m.score(model,rows,list(range(16)),device,'residual')
        if not torch.equal(before,initial):
            raise RuntimeError('Zero-adapter/full-copy initial scoring parity failed')
        model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    rng, start, losses = np.random.default_rng(seed), 0, {}
    resume = folder/'resume.pt'
    if resume.exists():
        state = torch.load(resume,map_location=device)
        if state['provenance']!=provenance:
            raise RuntimeError('Optimizer resume drift')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
        start, losses = state['step'],state['losses']
    label = 'local_official_pdm' if method=='local_anchor' else 'branch_returns'
    labels = torch.tensor(np.stack([row[label] for row in rows]),dtype=torch.float32,device=device)
    for step in range(start+1,501):
        ids = rng.choice(64,16,replace=False).tolist()
        # Exactly 8 states per scene: uniform scene then uniform decision.
        replay_ids = (rng.choice(64,16,replace=False)*8+rng.integers(8,size=16)).tolist()
        student = m.score(model,replay,replay_ids,device,'residual')
        with torch.no_grad():
            teacher = m.score(incumbent,replay,replay_ids,device,'residual')
        correction = exact_group(m.score(model,rows,ids,device,'residual'),labels[ids],1.)
        kl = rare.teacher_kl(student,teacher)
        loss = correction+.01*kl
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite rare loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        m.v3.clip_grad_norm_cpu_([p for p in model.parameters() if p.requires_grad],10.)
        optimizer.step()
        losses = dict(total=float(loss.detach()),correction=float(correction.detach()),teacher_kl=float(kl.detach()))
        if step%25==0:
            tmp = resume.with_suffix('.tmp')
            torch.save(dict(provenance=provenance,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
                            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),losses=losses,
                            cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None),tmp)
            tmp.replace(resume)
    if any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in frozen.items()):
        raise RuntimeError('Frozen V3 tensors changed')
    path = folder/'step_500.pt'
    model.eval()
    rare.save(path,model,config,provenance)
    restored,_ = rare.load(path)
    restored.to(device)
    maximum = 0.
    with torch.no_grad():
        for offset in range(0,64,16):
            ids = list(range(offset,offset+16))
            a,b = (m.score(x,rows,ids,device,'residual') for x in (model,restored))
            maximum = max(maximum,float((a-b).abs().max()))
            if maximum>c.ARRAY_TOLERANCE or not torch.equal(a.argmax(-1),b.argmax(-1)):
                raise RuntimeError('Export score/action parity failed')
    c.atomic_json(folder/'report.json',dict(status='PASS',provenance=provenance,selector=d.artifact(path),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  trainable_names=sorted(trainable),frozen_tensors_verified=len(frozen),incumbent_parity=parity,
                  export_max_abs=maximum,losses=losses,inference_uses_reward_or_q=False))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--method',choices=r.POLICIES,required=True)
    p.add_argument('--seed',type=int,choices=r.SEEDS,required=True)
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    train(a.run_root.resolve(),a.method,a.seed,a.device)
