"""Actual cached input/teacher/zero-adapter/online-bank parity; no optimization."""
import argparse
from pathlib import Path
import numpy as np
import torch
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_rare_model as rare
from selector_cfpi_deployment_router import selector_scores
from train_selector_cfpi import validate_incumbent_recompute


def check(run,device):
    torch.set_num_threads(4)
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA preflight unavailable')
    inputs = d.read(run/'rare_inputs.json')
    rows = c.validate_cache(c.load_pickle(d.verify(inputs['artifacts']['cache'])))
    replay = c.load_pickle(d.verify(inputs['artifacts']['replay']))['rows']
    incumbent,_,_ = m.load_incumbent(d.verify(inputs['artifacts']['scalar_manifest']))
    incumbent.to(device)
    parity = validate_incumbent_recompute(incumbent,rows,device)
    anchor = rare.AnchoredSelector(incumbent).to(device).eval()
    max_teacher,max_online = 0.,0.
    with torch.no_grad():
        for group in (rows,replay):
            for offset in range(0,len(group),16):
                ids = list(range(offset,min(offset+16,len(group))))
                base = m.score(incumbent,group,ids,device,'residual')
                scores = m.score(anchor,group,ids,device,'residual')
                if not torch.equal(base,scores):
                    raise RuntimeError('Zero adapter changed V3 scores/actions')
                stored = torch.tensor(np.stack([group[i]['v3_logits'] for i in ids]),device=device)
                max_teacher = max(max_teacher,float((base-stored).abs().max()))
                if max_teacher>c.MODEL_RECOMPUTE_TOLERANCE or not torch.equal(base.argmax(-1),stored.argmax(-1)):
                    raise RuntimeError('Replay/correction teacher parity failed')
            # Compare cached training scoring and the actual online bank path.
            for index in (0,len(group)-1):
                offline = m.score(anchor,group,[index],device,'residual')[0].cpu().numpy()
                online = selector_scores(anchor,group[index],device,'residual')
                max_online = max(max_online,float(abs(offline-online).max()))
                if max_online>c.ARRAY_TOLERANCE or offline.argmax()!=online.argmax():
                    raise RuntimeError('Cached/online score semantics differ')
    report = dict(status='PASS',device=device,rows=len(rows),replay_states=len(replay),
                  zero_adapter_scores_exact=True,teacher_max_abs=max_teacher,online_max_abs=max_online,
                  incumbent_parity=parity,training_performed=False,rollout_performed=False,
                  inputs=d.artifact(run/'rare_inputs.json'))
    c.locked_json(run/f'rare_cached_preflight_{device}.json',report)
    print(report)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a = p.parse_args()
    check(a.run_root,a.device)
