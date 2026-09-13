"""Supervised CPU helpers and native open-loop worker."""
import argparse
import os
from pathlib import Path
import subprocess
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_snapshot_common as s


def checked_open(run, policy, block):
    root=Path(run)/'openloop'/policy/block
    if not (root/'audit.json').exists():
        return None
    a=d.read(root/'audit.json');inputs=s.checked_inputs(run)
    for entry in a['artifacts']:
        d.verify(entry)
    if (a['status']!='PASS' or a['policy']!=policy or a['block']!=block
            or a['count']!=len(inputs['open_tokens'][block])
            or a['checkpoint']!=inputs['models'][policy] or a['noise']!=s.FORMAL_NOISE
            or a['contract']!=d.artifact(Path(run)/'run_contract.json')):
        raise RuntimeError('Open-loop checkpoint/protocol drift')
    if a['metrics']!=s.metrics(d.verify(a['pdm']),inputs['open_tokens'][block],valid=True):
        raise RuntimeError('Open-loop metric drift')
    return a


def openloop(run,policy,block):
    old=checked_open(run,policy,block)
    if old is not None:
        return old
    inputs=s.checked_inputs(run)
    checkpoint=d.verify(inputs['models'][policy])
    alg=Path(os.environ['ALGENGINE_ROOT'])
    config=alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_p1_snapshot.py'
    script=alg/'scripts'/('e2e_dist_eval.sh' if block=='open_navtest' else 'e2e_dist_eval_navtest_failures.sh')
    # Private per-condition checkpoint directory isolates all native evaluator outputs,
    # including interrupted attempts, from both the archive and the other block.
    folder=Path(run)/'openloop'/policy/block
    attempt=folder/'attempts'
    attempt.mkdir(parents=True,exist_ok=True)
    import tempfile
    current=Path(tempfile.mkdtemp(prefix='attempt_',dir=attempt))
    local=s.preserved_copy(checkpoint,current/'checkpoint.pth')
    env=dict(os.environ,SELECTOR_SNAPSHOT_NOISE=s.FORMAL_NOISE,SELECTOR_SNAPSHOT_MODE='OPEN',
        SELECTOR_SNAPSHOT_OPEN='1',SELECTOR_SNAPSHOT_CONDITION=policy+'_'+block,
        NAVSIM_OFFICIAL_RESCORE='always',PYTHON_BIN=os.environ['ALGENGINE_PYTHON'],
        MASTER_PORT='28620' if block=='open_navtest' else '28621')
    subprocess.run(['bash',str(script),str(config),local['path'],'4'],cwd=alg,env=env,check=True)
    pdms=list((current/'test').glob('*_official_pdms/pdm_scores_merged.csv'))
    if len(pdms)!=1:
        raise RuntimeError('Expected exactly one official PDM artifact in this attempt')
    values=s.metrics(pdms[0],inputs['open_tokens'][block],valid=True)
    raw=Path(str(pdms[0].parent).removesuffix('_official_pdms')+'.csv')
    rows=s.real_rows(raw,inputs['open_tokens'][block])
    ade={}
    if block=='open_navtest':
        for key in ('ade_4s','fde_4s'):
            array=np.asarray([float(v[key]) for v in rows])
            if not np.isfinite(array).all():
                raise RuntimeError('Nonfinite native ADE/FDE')
            ade[key]=float(array.mean())
    report=dict(status='PASS',policy=policy,block=block,noise=s.FORMAL_NOISE,
        checkpoint=inputs['models'][policy],contract=d.artifact(Path(run)/'run_contract.json'),
        pdm=d.artifact(pdms[0]),raw=d.artifact(raw),metrics=values,ade_fde=ade,
        count=len(values),artifacts=[d.artifact(pdms[0]),d.artifact(raw),local])
    c.locked_json(folder/'audit.json',report)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=('parity','audit','bridge','open','report','config'))
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--collection',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--policy',choices=s.POLICIES)
    p.add_argument('--block',choices=('open_navtest','open_rare'))
    a=p.parse_args()
    if a.task=='config':
        from mmcv import Config
        config=Config.fromfile(str(Path(os.environ['ALGENGINE_ROOT'])/'configs/diffusiondrive/e2e_diffusiondrive_selector_p1_snapshot.py'))
        if (config.model.planning_head.online_reward is not None or config.get('cfpi_deployment_routing')
                or config.model.planning_head.candidate_noise_namespace!=s.FORMAL_NOISE):
            raise RuntimeError('Native evaluation config mismatch')
        _=config.pretty_text
        result=dict(status='PASS')
    elif a.task=='parity':
        from selector_snapshot_data import parity
        result=parity(a.run,a.device)
    elif a.task=='audit':
        from selector_snapshot_audit import audit
        result=audit(a.collection)
    elif a.task=='bridge':
        from selector_snapshot_audit import bridge_report
        result=bridge_report(a.run)
    elif a.task=='report':
        from report_selector_snapshot import report
        result=report(a.run)
    else:
        result=openloop(a.run,a.policy,a.block)
    print({k:result[k] for k in ('status','decision','count','scenes') if k in result},flush=True)


if __name__=='__main__':
    main()
