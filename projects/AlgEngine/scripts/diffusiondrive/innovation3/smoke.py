"""Synthetic-context learner smoke with a real registered V3 checkpoint.

This deliberately does NOT certify rendering, candidate generation, dynamics,
reactive agents, reward computation, or a real closed-loop rollout.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from grpo_selector_v3_cached_common import SceneConditionedTrajectorySetSelector
from .learner import (FROZEN_V3_PLUS_ZERO_RESIDUAL, PARAMETERIZATIONS,
                      OnlineV3Learner)
from .paths import checked_path, sha256_file


def run(settings, device):
    cfg = json.loads(Path(settings).read_text())
    path = checked_path(cfg['selector_state'])
    digest = sha256_file(path)
    if digest != cfg['expected_sha256']['selector_state']:
        raise ValueError('Registered V3 selector SHA256 mismatch')
    payload = torch.load(path, map_location='cpu')
    if payload.get('method') != 'scene_conditioned_exact_group_grpo' or payload.get('schema_version') != 3:
        raise ValueError('Expected standard V3 selector schema/method')
    model = SceneConditionedTrajectorySetSelector(**payload['scene_selector_config']).to(device)
    model.load_state_dict(payload['scene_selector_state'], strict=True)
    parameterization = cfg.get('online_parameterization', FROZEN_V3_PLUS_ZERO_RESIDUAL)
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError('Unknown online_parameterization: ' + str(parameterization))
    source = dict(kind='offline_selector_file', path=str(path), sha256=digest,
                  payload_schema=payload['schema_version'], method=payload['method'],
                  scene_selector_config=payload['scene_selector_config'])
    learner = OnlineV3Learner(model, parameterization=parameterization,
                              initialization_source=source)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    # Synthetic features: all reports below explicitly retain this limitation.
    with torch.inference_mode():
        context = dict(candidate_features=torch.randn(1,20,256,device=device),
            candidate_trajectories=torch.randn(1,20,8,3,device=device),
            route_bev_features=torch.randn(1,20,8,256,device=device),
            status_token=torch.randn(1,1,256,device=device),
            ego_query=torch.randn(1,1,256,device=device),
            agents_query=torch.randn(1,30,256,device=device))
        base = torch.randn(1,20,device=device)
    initial = copy.deepcopy(learner.reference.state_dict())
    initial_selector = copy.deepcopy(learner.selector.state_dict())
    with torch.no_grad():
        correction = learner.selector(**context)
    residual_initially_zero = bool(torch.equal(correction, torch.zeros_like(correction)))
    if parameterization == FROZEN_V3_PLUS_ZERO_RESIDUAL and not residual_initially_zero:
        raise RuntimeError("Legacy online residual is not initially zero")
    reports=[]
    for step in range(8):
        _, probabilities, version = learner.choose(context,base,str(step))
        if step == 0:
            with torch.no_grad():
                expected=(base+learner.reference(**context)).softmax(-1)
            if not torch.equal(expected, probabilities):
                raise RuntimeError('Step-zero V3 parity failed')
        rewards = torch.roll(torch.arange(20,device=device,dtype=torch.float32),step)[None,:] / 19
        report=learner.update(rewards,str(step))
        if report['policy_version'] != version + 1:
            raise RuntimeError('Policy version did not advance')
        reports.append(report)
    restored = OnlineV3Learner(copy.deepcopy(model), parameterization=parameterization,
                                initialization_source=source)
    # A real serialization round-trip without writing an intermediate training file.
    import io
    buffer=io.BytesIO(); torch.save(learner.state_dict(),buffer);buffer.seek(0)
    restored.load_state_dict(torch.load(buffer,map_location=device))
    left=learner.choose(context,base,'resume');right=restored.choose(context,base,'resume')
    if left[0] != right[0] or not torch.equal(left[1],right[1]):
        raise RuntimeError('Serialized resume parity failed')
    rewards=torch.arange(20,device=device,dtype=torch.float32)[None,:]
    learner.update(rewards,'resume');restored.update(rewards,'resume')
    if not all(torch.equal(v,restored.selector.state_dict()[k]) for k,v in learner.selector.state_dict().items()):
        raise RuntimeError('Optimizer resume parity failed')
    if not all(torch.equal(v,learner.reference.state_dict()[k]) for k,v in initial.items()):
        raise RuntimeError('Frozen V3 reference changed')
    if not all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items()):
        raise RuntimeError('Input V3 weights changed')
    changed=sum(not torch.equal(v,learner.selector.state_dict()[k]) for k,v in initial_selector.items())
    if changed == 0: raise RuntimeError('No selector parameter changed')
    if device.type == 'cuda': torch.cuda.synchronize()
    return dict(status='PASS_SYNTHETIC_CONTEXT_LEARNER_ONLY', selector_sha256=digest,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        updates=reports, changed_tensor_count=changed, step0_v3_parity=True,
        parameterization=parameterization, residual_initially_zero=residual_initially_zero,
        online_selector_initial_output_zero=residual_initially_zero, input_v3_unchanged=True,
        initialization_provenance=learner.provenance(),
        serialized_optimizer_rng_resume_parity=True, frozen_reference_unchanged=True,
        real_observation=False, real_generator=False, real_simulation=False,
        real_reward=False, closed_loop_verified=False, formal_ready=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    output=checked_path(a.output,must_exist=False)
    code=Path(__file__).resolve().parents[5]
    if output == code or code in output.parents: p.error('Reports must be outside code')
    if output.exists(): p.error('Use a fresh report path')
    try:
        report=run(a.settings,torch.device(a.device)); status=0
    except Exception as error:
        report=dict(status='FAIL_LEARNER_SMOKE',error=repr(error),closed_loop_verified=False);status=1
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))
    return status


if __name__=='__main__': sys.exit(main())
