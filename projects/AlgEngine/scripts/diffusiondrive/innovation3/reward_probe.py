"""CPU causal H1 reward acceptance; no visual inference, optimizer or formal score."""
import argparse
import json
import logging
from pathlib import Path
import pickle
import subprocess
import time
import traceback
from .paths import checked_path, sha256_file
from .snapshot_probe import IDMFailures, runtime_evidence, save_parity_failure
from .visual_assets import visual_asset
from worldengine.online.headless import HeadlessSimulator, diagnostic_candidates
from worldengine.online.reward import CONTRACT
from worldengine.online.reward_runtime import RewardSession
from worldengine.online.state import SnapshotParityError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args=parser.parse_args()
    if not 1<=args.steps<=8 or args.seed<0:
        parser.error('Use steps 1..8 and a nonnegative seed')
    output=checked_path(args.output.absolute(), must_exist=False)
    code=Path(__file__).resolve().parents[5]
    if code in output.parents or output==code:
        parser.error('Private output must be outside code')
    if any(output.with_suffix(s).exists() for s in ('.json','.events.jsonl','.failure')):
        parser.error('Choose a fresh report basename')
    cfg=json.loads(checked_path(args.settings.absolute()).read_text())
    report=dict(status='RUNNING', contract=CONTRACT, groups=[],
        real_generator=False, online_update=False, official_pdm_reward=False,
        real_closed_loop=False, formal_ready=False, reward_adapter_verified=False,
        candidates='current_state_analytic_diagnostic_not_generator', seed=args.seed,
        code_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip())
    started=time.monotonic(); sim=None
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(report,f,indent=2)
    try:
        report['runtime']=runtime_evidence()
        sources=list(Path(__file__).parent.glob('*.py'))+list((code/'projects/SimEngine/worldengine/online').glob('*.py'))
        sources += [Path(__file__).parent.parent/'innovation3_runtime.py']
        report['source_sha256']={str(p.relative_to(code)):sha256_file(p) for p in sorted(sources)}
        source=checked_path(Path(cfg['scenario_root'])/'original/navtrain_failures_per1/all_scenarios.pkl')
        report['source']=dict(path=str(source),bytes=source.stat().st_size,sha256=sha256_file(source))
        with source.open('rb') as f: scenes=pickle.load(f)
        scene_id,_,_=visual_asset(cfg,scenes)
        scene=scenes[scene_id]; del scenes
        report['scene']=scene_id
        root=checked_path(cfg['map_root'])
        metadata=checked_path(root/'nuplan-maps-v1.0.json')
        version=json.loads(metadata.read_text())[scene['map']]['version']
        mapfile=checked_path(root/scene['map']/version/'map.gpkg')
        report['map']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256_file(p)) for p in (metadata,mapfile)]
        failure=IDMFailures(); logging.getLogger().addHandler(failure)
        for reaction in ('NR','R'):
            sim=HeadlessSimulator(scene_id,scene,reaction,13+args.steps,args.seed)
            session=RewardSession(sim,root)
            for _ in range(13): session.warmup(diagnostic_candidates(sim)[12])
            for index in range(args.steps):
                group=session.group(diagnostic_candidates(sim),12)
                if failure.count: raise RuntimeError('IDM fallback invalidates reward probe')
                group.update(reaction=reaction,decision=index,planner_step=sim.engine.episode_step)
                report['groups'].append(group)
                output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(json.dumps(dict(reaction=reaction,decision=index,reward_std=group['reward_std'],
                    main=group['main'])),flush=True)
            sim.close();sim=None
        report.update(status='PASS_CAUSAL_H1_REWARD_ADAPTER_ONLY',reward_adapter_verified=True,
            candidate_branches=len(report['groups'])*20,idm_fallbacks=failure.count,
            groups_with_signal=sum(g['reward_std']>1e-6 for g in report['groups']))
    except BaseException as error:
        if isinstance(error,SnapshotParityError): save_parity_failure(error,output)
        report.update(status='FAIL_CAUSAL_H1_REWARD_ADAPTER',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if sim is not None: sim.close()
        report['elapsed_seconds']=time.monotonic()-started
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=report['status'],report=str(output))),flush=True)

if __name__=='__main__': main()
