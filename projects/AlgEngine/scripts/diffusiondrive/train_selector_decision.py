"""Segmented training with arm-owned feedback and full optimizer/RNG continuation."""
import argparse
from pathlib import Path
import numpy as np
import torch
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_decision_common as x
import selector_decision_data as data
from selector_decision_collections import require_pilot
from train_selector_feedback import Dense, pair_loss
from selector_rare_model import teacher_kl


def parent_entry(run, seed):
    if seed == 0:
        return data.checked_inputs(run)['artifacts']['parent0']
    report = d.read(run/'train'/f'shared_seed{seed}'/'step_500_report.json')
    if (report['status'] != 'PASS' or report['provenance']['policy'] != f'shared_seed{seed}'
            or report['provenance']['code_sha'] != d.read(run/'run_contract.json')['code_sha']):
        raise RuntimeError('First-round parent missing or changed')
    d.verify(report['resume'])
    return report['resume']


def generation_for(arm, end):
    return ((0 if end == 100 else 1 if end == 300 else 2) if arm in ('P2', 'P3') else 0)


def train(run, arm, seed, end, device='cuda'):
    require_pilot(run)
    if (arm not in ('shared', *x.ARMS) or seed not in x.SEEDS
            or end not in (100, 300, 500) or arm == 'shared' and (seed == 0 or end != 500)):
        raise RuntimeError('Unregistered training segment')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('No silent CPU fallback')
    inputs = data.checked_inputs(run)
    generation = generation_for(arm, end)
    rows, cache_entry = data.load_rows(run, 'P0' if arm == 'shared' else arm, seed, generation)
    if arm == 'shared':
        rows = rows[:192]
    policy = f'{arm}_seed{seed}'
    folder = run/'train'/policy
    start = 0 if arm == 'shared' or end == 100 else 100 if end == 300 else 300
    parent = None
    if arm != 'shared':
        parent = parent_entry(run, seed) if start == 0 else d.read(folder/f'step_{start}_report.json')['resume']
    provenance = dict(method=x.METHOD, policy=policy, seed=seed, arm=arm,
                      code_sha=d.read(run/'run_contract.json')['code_sha'],
                      inputs=d.artifact(run/'decision_inputs.json'), data=cache_entry,
                      parent=parent, start=start, end=end, generation=generation,
                      training=x.TRAIN, inference_uses_reward_or_q=False)
    c.locked_json(folder/f'segment_{end}_contract.json', provenance)
    output = folder/f'step_{end}_report.json'
    if output.exists():
        report = d.read(output)
        if report['status'] != 'PASS' or report['provenance'] != provenance:
            raise RuntimeError('Training segment provenance changed')
        d.verify(report['selector']); d.verify(report['resume'])
        return report
    scalar_manifest = d.verified_read(inputs['artifacts']['source_inputs'])['artifacts']['scalar_manifest']
    incumbent, config, _ = m.load_incumbent(d.verify(scalar_manifest))
    incumbent.to(device).eval().requires_grad_(False)
    model = m.initialize(incumbent, 'residual').to(device).requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    def restore(state):
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)

    if parent is not None:
        state = torch.load(d.verify(parent), map_location=device)
        expected_step = 500 if start == 0 else start
        expected_arm = 'shared' if start == 0 else arm
        if (state['step'] != expected_step or state['provenance']['seed'] != seed
                or state['provenance']['arm'] != expected_arm):
            raise RuntimeError('Wrong optimizer parent seed/step/arm')
        restore(state)
    resume = folder/f'segment_{end}_resume.pt'
    current, losses = start, {}
    if resume.exists():
        state = torch.load(resume, map_location=device)
        if state['provenance'] != provenance or not start <= state['step'] <= end:
            raise RuntimeError('Resume segment identity drift')
        restore(state)
        current, losses = state['step'], state['losses']
    dense = Dense(run)
    strata = {mode: [r for r in rows if r['mode'] == mode] for mode in x.MODES}
    model.train()

    def save_resume(step):
        temp = resume.with_suffix('.tmp')
        torch.save(dict(provenance=provenance, step=step, model=model.state_dict(),
                        optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
                        losses=losses), temp)
        temp.replace(resume)

    for step in range(current+1, end+1):
        dense_rows = dense.sample(rng)
        pairs = [strata[mode][int(i)] for mode in x.MODES for i in rng.integers(len(strata[mode]), size=8)]
        batch = dense_rows+pairs
        student = m.score(model, batch, list(range(32)), device, 'residual')
        with torch.no_grad():
            teacher = m.score(incumbent, batch, list(range(32)), device, 'residual')
        rewards = torch.as_tensor(np.stack([r['candidate_rewards'] for r in dense_rows]), dtype=torch.float32, device=device)
        valid = torch.as_tensor(np.stack([r['candidate_reward_valid_mask'] for r in dense_rows]), dtype=torch.bool, device=device)
        grpo = m.v3.exact_group_loss(student[:16], teacher[:16], rewards, valid, 1., 0.)[0]
        if arm in ('P0', 'shared'):
            preference = pair_loss(student[16:], pairs)  # preserve legacy operation order
            nll = student.sum()*0.
        else:
            preference, nll = x.feedback_loss(student[16:], pairs)
        kl = teacher_kl(student, teacher)
        loss = grpo+preference+.01*kl if arm in ('P0', 'shared') else grpo+preference+nll+.01*kl
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        m.v3.clip_grad_norm_cpu_(model.parameters(), 10.)
        optimizer.step()
        losses = dict(total=float(loss.detach()), dense=float(grpo.detach()), preference=float(preference.detach()),
                      winner_nll=float(nll.detach()), teacher_kl=float(kl.detach()))
        if step % 25 == 0:
            save_resume(step)
            print(dict(policy=policy, step=step, **losses), flush=True)
    model.eval()
    selector = folder/f'step_{end}.pt'
    m.save_selector(selector, model, config, 'residual', provenance)
    restored, _ = m.load_selector(selector)
    restored.to(device)
    with torch.no_grad():
        a = m.score(model, rows[:16], list(range(16)), device, 'residual')
        b = m.score(restored, rows[:16], list(range(16)), device, 'residual')
    error = float((a-b).abs().max())
    if error > c.ARRAY_TOLERANCE or not torch.equal(a.argmax(-1), b.argmax(-1)):
        raise RuntimeError('Export parity failed')
    parity = None
    if arm == 'P0' and seed == 0 and end == 500:
        original, _ = m.load_selector(d.verify(inputs['models']['T_source']))
        parameter_error = max(float((v.detach().cpu()-original.state_dict()[k]).abs().max())
                              for k,v in model.state_dict().items())
        original.to(device)
        old_scores, new_scores = data.scores(original, rows, device), data.scores(restored, rows, device)
        score_error = float(np.max(np.abs(old_scores-new_scores)))
        parity = dict(parameter_max_abs=parameter_error, score_max_abs=score_error,
                      action_mismatches=int(np.sum(old_scores.argmax(1) != new_scores.argmax(1))))
        if (parameter_error > c.ARRAY_TOLERANCE or score_error > c.MODEL_RECOMPUTE_TOLERANCE
                or parity['action_mismatches']):
            raise RuntimeError('P0 does not reproduce frozen T seed0: '+str(parity))
    report = dict(status='PASS', provenance=provenance, step=end, selector=d.artifact(selector),
                  resume=d.artifact(resume), losses=losses, export_max_abs=error, legacy_P0_parity=parity,
                  trainable_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_names=sorted(k for k,_ in model.named_parameters()),
                  inference_uses_reward_or_q=False)
    c.locked_json(output, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--arm', choices=('shared', *x.ARMS), required=True)
    p.add_argument('--seed', type=int, choices=x.SEEDS, required=True)
    p.add_argument('--end', type=int, choices=(100, 300, 500), required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    train(a.run_root, a.arm, a.seed, a.end, a.device)
