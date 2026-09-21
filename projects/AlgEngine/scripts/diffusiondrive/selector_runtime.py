#!/usr/bin/env python3
"""Portable explicit-path launcher; no project migration wrapper or implicit shell setup."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

CODE=Path(__file__).resolve().parents[4]
ALG=CODE/'projects/AlgEngine';SIM=CODE/'projects/SimEngine';TOOLS=Path(__file__).parent


def environment(settings, devices):
    cfg=json.loads(Path(settings).read_text())
    required=('algengine_python','simengine_python','data_root','baseline','navsim_root',
              'nuplan_root','metric_cache_train','metric_cache_eval','test_images','train_images','anchors')
    missing=[key for key in required if not cfg.get(key)]
    if missing:raise ValueError('Missing runtime settings: '+str(missing))
    for key in required:
        if not Path(cfg[key]).is_absolute():raise ValueError('Use absolute paths: '+key)
    ids=devices.split(',')
    if not ids or any(not x.isdigit() for x in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid devices')
    env={k:v for k,v in os.environ.items() if not k.startswith(("WORLDENGINE_", "DIFFUSIONDRIVE_", "NAVSIM_", "PAPER_", "PROOT_")) and k not in ("PYTHONHOME", "SELECTOR_ABLATION")}
    # Do not inherit paths from an earlier project or previous experimentation.
    paths=[SIM/'scripts/diffusiondrive/algengine_worker_bootstrap',TOOLS,ALG,SIM,
           Path(cfg['navsim_root']),Path(cfg['nuplan_root'])]
    paths.extend(Path(x) for x in cfg.get('extra_python_paths',[]))
    env.update(WORLDENGINE_ROOT=str(CODE),ALGENGINE_ROOT=str(ALG),SIMENGINE_ROOT=str(SIM),
        WORLDENGINE_DATA_ROOT=cfg['data_root'],WORLDENGINE_TEST_IMAGES=cfg['test_images'],
        WORLDENGINE_TRAIN_IMAGES=cfg['train_images'],DIFFUSIONDRIVE_ANCHORS=cfg['anchors'],
        DIFFUSIONDRIVE_SELECTOR_INIT_CHECKPOINT=cfg['baseline'],
        NAVSIM_DEVKIT_ROOT=cfg['navsim_root'],NAVSIM_METRIC_CACHE_PATH_TRAIN=cfg['metric_cache_train'],
        NAVSIM_METRIC_CACHE_PATH_EVAL=cfg['metric_cache_eval'],NAVSIM_METRIC_CACHE_PATH=cfg['metric_cache_eval'],
        PYTHON_BIN=cfg['algengine_python'],ALGENGINE_PYTHON=cfg['algengine_python'],
        SIMENGINE_PYTHON=cfg['simengine_python'],PYTHONPATH=os.pathsep.join(map(str,paths)),
        CUDA_VISIBLE_DEVICES=devices,OMP_NUM_THREADS=str(cfg.get('omp_threads',4)),
        NAVSIM_OFFICIAL_RESCORE='always',NAVSIM_RESCORE_SHARDS=str(len(ids)))
    env['PATH']=str(Path(cfg['algengine_python']).parent)+os.pathsep+env.get('PATH','')
    for kind in ('mmcv','gsplat'):
        env.pop('WORLDENGINE_DIFFUSIONDRIVE_'+kind.upper()+'_BOOTSTRAP',None)
        if cfg.get(kind+'_extension'):
            env['WORLDENGINE_DIFFUSIONDRIVE_'+kind.upper()+'_BOOTSTRAP']='1'
            env['WORLDENGINE_'+kind.upper()+'_EXTENSION']=cfg[kind+'_extension']
    allowed_method_env = {'SELECTOR_ABLATION', 'DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE',
                          'DIFFUSIONDRIVE_GRPO_KL_WEIGHT', 'DIFFUSIONDRIVE_GRPO_V3_MODEL_DIM'}
    for key, value in cfg.get('method_env', {}).items():
        if key not in allowed_method_env:
            raise ValueError('Unsupported method environment key: ' + key)
        env[key] = str(value)
    return cfg,env


def check(cfg):
    checks=[]
    for key,value in cfg.items():
        if isinstance(value,str) and value.startswith('/'):
            p=Path(value);checks.append(dict(key=key,path=value,resolved=str(p.resolve()),exists=p.exists()))
    failed=[x for x in checks if not x['exists']]
    print(json.dumps(dict(status='FAIL_PATH_CHECK' if failed else 'PASS_PATH_CHECK_ONLY',checks=checks,
          data_coverage_verified=False,gpu_verified=False,formal_ready=False),indent=2))
    if failed:raise RuntimeError('Missing runtime paths; see checks above')


def run_commands(commands, out, env, timeout, dry=False):
    print(json.dumps([dict(argv=c,cwd=str(w)) for c,w,_ in commands],indent=2),flush=True)
    if dry:return
    # A fresh output per attempt avoids changing completed work or guessing how to resume it.
    out.mkdir(parents=True,exist_ok=False)
    (out/'commands.json').write_text(json.dumps([dict(argv=c,cwd=str(w),log=n) for c,w,n in commands],indent=2))
    children=[];files=[];start=time.monotonic();status=dict(status='RUNNING',pid=os.getpid(),started=time.time())
    (out/'process_status.json').write_text(json.dumps(status,indent=2))
    def stopped(signum,frame):raise InterruptedError('Signal '+str(signum))
    previous={s:signal.signal(s,stopped) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for cmd,cwd,name in commands:
            f=(out/name).open('w');files.append(f)
            children.append(subprocess.Popen(cmd,cwd=str(cwd),env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
        while True:
            codes=[p.poll() for p in children]
            if any(c is not None and c!=0 for c in codes):raise RuntimeError('Child failed: '+str(codes)+'; logs='+str(out))
            if all(c==0 for c in codes):break
            if time.monotonic()-start>timeout:raise TimeoutError('Stage timeout; logs='+str(out))
            print(json.dumps(dict(elapsed_seconds=int(time.monotonic()-start),child_codes=codes,logs=str(out))),flush=True)
            time.sleep(30)
        status['status']='PASS_PROCESSES_ONLY'
    except BaseException as e:
        status.update(status='FAILED',error=repr(e));raise
    finally:
        # Terminate only sessions created by this launcher, including their descendants.
        for p in children:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for p in children:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
        for f in files:f.close()
        status.update(finished=time.time(),formal_table_ready=False)
        (out/'process_status.json').write_text(json.dumps(status,indent=2))
        for s,handler in previous.items():signal.signal(s,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,required=True);p.add_argument('--devices',default='0')
    p.add_argument('--dry-run',action='store_true')
    sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('check')
    tool=sub.add_parser('tool');tool.add_argument('name');tool.add_argument('arguments',nargs=argparse.REMAINDER)
    for mode in ('openloop','closedloop'):
        q=sub.add_parser(mode);q.add_argument('--config',type=Path,required=True);q.add_argument('--checkpoint',type=Path,required=True)
        q.add_argument('--output',type=Path,required=True);q.add_argument('--eval-seed',type=int,default=0)
        q.add_argument('--timeout',type=int,default=43200)
        if mode=='openloop':q.add_argument('--split',choices=['navtest','rare'],required=True)
        else:q.add_argument('--reaction',choices=['NR','R'],required=True);q.add_argument('--scenarios',type=Path,required=True);q.add_argument('--assets',type=Path,required=True)
    a=p.parse_args();cfg,env=environment(a.settings,a.devices)
    if a.mode=='check':return check(cfg)
    if a.mode=='tool':
        if Path(a.name).name!=a.name:raise ValueError('Tool must be a basename')
        script=TOOLS/(a.name.removesuffix('.py')+'.py')
        if not script.is_file():raise FileNotFoundError(script)
        cmd=[cfg['algengine_python'],str(script),*a.arguments]
        if a.dry_run:print(json.dumps(cmd));return
        os.chdir(ALG);os.execve(cmd[0],cmd,env)
    for f in (a.config,a.checkpoint):
        if not f.is_file():raise FileNotFoundError(f)
    out=a.output.resolve();n=len(a.devices.split(','))
    if CODE==out or CODE in out.parents:raise ValueError('Output must be outside the code repository')
    env['DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE']='formal_navtest_seed'+str(a.eval_seed)
    config=str(a.config.resolve());checkpoint=str(a.checkpoint.resolve())
    if a.mode=='openloop':
        # The upstream evaluator writes beside its checkpoint. Use a per-attempt link.
        link=out.parent/(out.name+'_inputs')/'checkpoint.pth'
        env['WORLDENGINE_EVAL_OUTPUT']=str(out/'work')
        env['WORLDENGINE_EVAL_SEED']=str(a.eval_seed)
        env['WORLDENGINE_COLLECT_TMPDIR']=str(link.parent/'distributed_collect')
        script='e2e_dist_eval.sh' if a.split=='navtest' else 'e2e_dist_eval_navtest_failures.sh'
        cmds=[(['bash',str(ALG/'scripts'/script),config,str(link),str(n)],ALG,'openloop.log')]
        if not a.dry_run:
            if out.exists() or link.parent.exists():raise FileExistsError('Choose a new attempt output')
            link.parent.mkdir(parents=True);link.symlink_to(checkpoint)
            (link.parent/'distributed_collect').mkdir()
        run_commands(cmds,out,env,a.timeout,a.dry_run)
    else:
        for f in (a.scenarios,a.assets):
            if not f.exists():raise FileNotFoundError(f)
        root=str(out/'simulation');worker=root+'/__WORKER_ID__'
        cmd=[cfg['simengine_python'],str(SIM/'worldengine/runner/run_simulation.py'),
             'debug_mode=True','debug_scene_name=null','data_file_path='+str(a.scenarios.resolve()),
             'asset_folder_path='+str(a.assets.resolve()),'output_dir='+worker+'/WE_output',
             'use_planner_actions=true','ego_policy=env_input_policy','ego_client=navformer_client',
             'ego_controller=log_play_controller','ego_navigation=trajectory_navigation',
             'planner_data_path='+worker+'/plan_traj','planner_client_folder='+worker+'/frames',
             'with_metric_manager=true','with_dense_reward_manager=false','distributed_mode=SCENARIO_BASED',
             'worker=ray_distributed','worker_id_prefix=split_','enable_resume=false',
             'nuplan_map_root='+str(Path(cfg['data_root'])/'raw/nuplan/dataset/maps'),
             'exit_on_failure=true']
        if a.reaction=='R':cmd+=['agent_policy=idm_policy','agent_navigation=idm_navigation']
        cmds=[(cmd,SIM,'simulator.log')]
        # Worker GPU IDs are selected explicitly in a tiny subprocess environment wrapper.
        for index,gpu in enumerate(a.devices.split(',')):
            w=Path(root)/('split_'+str(index))
            cmd=['env','CUDA_VISIBLE_DEVICES='+gpu,cfg['algengine_python'],str(ALG/'closed_loop/sim_test.py'),
                 config,checkpoint,'--log-dir',str(w),'--seed',str(a.eval_seed),'--cfg-options',
                 'sim.monitored_folder='+str(w/'frames'),'sim.plan_save_path='+str(w/'plan_traj'),
                 'sim.merged_ann_save_dir='+str(w/'merged_ann_files'),'sim.clean_temp_files=True',
                 'sim.clean_record_data=False','data_root='+str(w/'WE_output/openscene_format/')+'/']
            cmds.append((cmd,ALG,'planner_'+str(index)+'.log'))
        run_commands(cmds,out,env,a.timeout,a.dry_run)
        if not a.dry_run:
            subprocess.run([cfg['simengine_python'],str(SIM/'scripts/merge_simulation_results.py'),
                '--test_path',root,'--react_type',a.reaction,'--num-splits',str(n)],env=env,cwd=SIM,check=True)
    print('DRY_RUN: no processes started.' if a.dry_run else 'Process stage ended. Full table acceptance remains a separate exact-coverage check.')
if __name__=='__main__':main()
