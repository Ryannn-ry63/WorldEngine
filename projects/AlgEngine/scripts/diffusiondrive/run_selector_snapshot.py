"""Four-card native observed-best snapshot. No training or effect gate."""
import argparse
import fcntl
import math
from pathlib import Path
import re
import signal
import time
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import selector_snapshot_common as s
from run_selector_cfpi import ROOT,SCRIPT,CANONICAL,environment,implementation_inventory
from run_selector_cfpi_deployment import Runner as BaseRunner,archive_incomplete_collection
from run_selector_feedback import Runner as FeedbackRunner
import selector_snapshot_data as data
import selector_snapshot_audit as audit

SOURCE=CANONICAL/'experiments/worktrees/WorldEngine-selector-decision-feedback-v1/experiments/diffusiondrive/selector_decision_feedback_v1/runs/decision_feedback_20260912_192h_v1'


def inventory():
    result=implementation_inventory()
    for folder in ('projects/AlgEngine/scripts','projects/AlgEngine/mmdet3d_plugin',
                   'projects/AlgEngine/configs','research/p1_snapshot'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.sh','.json','.md','.yaml'):
                result[str(path.relative_to(ROOT))]=c.sha256_file(path)
    result['run_diffusiondrive_selector_p1_snapshot_4h100.sh']=c.sha256_file(ROOT/'run_diffusiondrive_selector_p1_snapshot_4h100.sh')
    return result


class Runner(BaseRunner):
    charge_idle=FeedbackRunner.charge_idle

    def __init__(self,args):
        self.args=args;self.phase='snapshot'
        self.phase_limits={'snapshot':args.gpu_hours}
        self.account_idle=args.stage not in ('audit','report')
        self.idle_since=time.monotonic()
        self.run=ROOT/'experiments/diffusiondrive/selector_p1_snapshot_v1/runs'/args.run_id
        self.run.mkdir(parents=True,exist_ok=True)
        self.lock=(self.run/'runner.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.env=environment(args.source_worldengine_root)
        self.devices=f.gpu_devices(self.env,4)
        self.env.update(CUDA_VISIBLE_DEVICES=','.join(self.devices),
            WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP=','.join(self.devices),
            DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE=s.FORMAL_NOISE,
            NAVSIM_OFFICIAL_RESCORE='always',PYTHON_BIN=self.env['ALGENGINE_PYTHON'])
        exp=args.source_worldengine_root.parent/'exp'
        self.env.update(NAVSIM_EXP_ROOT=str(exp),NAVSIM_METRIC_CACHE_PATH=str(exp/'metric_cache_navtest'),
            NAVSIM_METRIC_CACHE_PATH_EVAL=str(exp/'metric_cache_navtest'),
            NAVSIM_METRIC_CACHE_PATH_TRAIN=str(exp/'metric_cache_trainval'),
            NAVSIM_DEVKIT_ROOT=str(args.source_worldengine_root.parent/'DiffusionDrive'),
            NAVSIM_RESCORE_SHARDS='4',NAVSIM_RESCORE_SPLIT='navtest',SELECTOR_SNAPSHOT_OPEN='0')
        self.alg=Path(self.env['ALGENGINE_ROOT']);self.sim=Path(self.env['SIMENGINE_ROOT'])
        self.config=self.alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_p1_snapshot.py'
        self.checkpoint_manifest=args.source_worldengine_root/'experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json'
        scalar_manifest=d.read(self.checkpoint_manifest)
        self.env.update(DIFFUSIONDRIVE_SELECTOR_INIT_CHECKPOINT=scalar_manifest['baseline'],
                        DIFFUSIONDRIVE_SELECTOR_INIT_SHA256=scalar_manifest['baseline_sha256'])
        self.inventory=inventory();self.code_sha=s.digest(self.inventory)
        self.env['DIFFUSIONDRIVE_ROLLOUT_CODE_SHA']=self.code_sha
        contract=dict(method=s.METHOD,code_sha=self.code_sha,implementation=self.inventory,
            source=str(args.source_run),source_report=d.artifact(args.source_run/'pilot_report.json'),
            source_contract=d.artifact(args.source_run/'run_contract.json'),
            source_ledger=d.artifact(args.source_run/'decision_ledger.json'),
            gpu_count=4,gpu_hour_limit=args.gpu_hours,exposure=s.EXPOSURE,
            selector_sha=s.SELECTOR_SHA,formal_noise=s.FORMAL_NOISE,bridge_noise=c.NOISE_NAMESPACE,
            source_worldengine_root=str(args.source_worldengine_root),
            native_environment={k:self.env[k] for k in ('NAVSIM_EXP_ROOT','NAVSIM_METRIC_CACHE_PATH',
                'NAVSIM_METRIC_CACHE_PATH_EVAL','NAVSIM_METRIC_CACHE_PATH_TRAIN','NAVSIM_DEVKIT_ROOT',
                'NAVSIM_RESCORE_SHARDS','DIFFUSIONDRIVE_SELECTOR_INIT_CHECKPOINT','DIFFUSIONDRIVE_SELECTOR_INIT_SHA256')},
            scorer=d.artifact(Path(self.env['NAVSIM_DEVKIT_ROOT'])/'navsim/planning/script/run_pdm_score_from_submission.py'),
            policies=list(s.POLICIES),conditions=[list(x) for x in s.conditions()],
            accounting='All four allocated GPUs including CPU helpers; offline audit/report uncharged',
            performance_gate_applied=False,no_training=True)
        c.locked_json(self.run/'run_contract.json',contract)
        self.ledger_path=self.run/'ledger.json'
        self.ledger=d.read(self.ledger_path) if self.ledger_path.exists() else dict(
            gpu_hours_used=0.,phase_gpu_hours={'snapshot':0.},phase_limits=self.phase_limits,active=None,events=[])
        if (self.ledger['phase_limits']!=self.phase_limits
                or set(self.ledger['phase_gpu_hours'])!={'snapshot'}
                or not math.isfinite(self.ledger['gpu_hours_used']) or self.ledger['gpu_hours_used']<0
                or abs(self.ledger['gpu_hours_used']-self.ledger['phase_gpu_hours']['snapshot'])>1e-6):
            raise RuntimeError('Ledger identity/charge drift')
        self.children=[]
        self.recover_accounting()
        signal.signal(signal.SIGINT,self.interrupt);signal.signal(signal.SIGTERM,self.interrupt)
        self.idle_since=time.monotonic()

    def execute(self,jobs,stage,gpus=0,timeout=21600):
        self.charge_idle()
        try:
            return BaseRunner.execute(self,jobs,stage,gpus,timeout)
        finally:
            self.idle_since=time.monotonic()

    def helper(self,task,*arguments,env=None):
        self.python(SCRIPT/'selector_snapshot_tasks.py',[task,'--run',self.run,*arguments],
                    stage=task+'_'+s.digest(list(map(str,arguments)))[:12],env=env,timeout=43200)

    def cpu_audit(self):
        data.freeze(self.run,self.args.source_run,self.args.source_worldengine_root)
        code={}
        for relative,expected in self.inventory.items():
            entry=s.preserved_copy(ROOT/relative,self.run/'archive/code'/relative)
            if entry['sha256']!=expected:
                raise RuntimeError('Code changed during archive: '+relative)
            code[relative]=entry
        c.locked_json(self.run/'archive/code_manifest.json',code)
        c.locked_json(self.run/'archive/reproduce.json',dict(
            worktree=str(ROOT),launcher='run_diffusiondrive_selector_p1_snapshot_4h100.sh',
            run_id=self.args.run_id,source_run=str(self.args.source_run),
            source_worldengine_root=str(self.args.source_worldengine_root),
            stages=['audit','preflight','all','report'],gpus=4,gpu_hours=self.args.gpu_hours,
            command_template='bash run_diffusiondrive_selector_p1_snapshot_4h100.sh STAGE '
                +self.args.run_id+' --gpus 4 --gpu-hours '+str(self.args.gpu_hours)
                +' --source-run '+str(self.args.source_run)
                +' --source-worldengine-root '+str(self.args.source_worldengine_root),
            contract=d.artifact(self.run/'run_contract.json')))
        data.parity(self.run,'cpu')
        if not Path(self.env['NAVSIM_METRIC_CACHE_PATH']).is_dir():
            raise RuntimeError('Missing official navtest metric cache')
        for policy in s.POLICIES:
            for cohort in ('bridge','full','common'):
                for mode in ('NR','R'):
                    audit.make_collection(self.run,policy,cohort,mode)
        result=dict(status='PASS',inputs=d.artifact(self.run/'inputs.json'),
            export=d.artifact(self.run/'archive/native_export_audit.json'),
            cached_parity=d.artifact(self.run/'cached_parity_cpu.json'),code_sha=self.code_sha,
            code_archive=d.artifact(self.run/'archive/code_manifest.json'),
            reproduce=d.artifact(self.run/'archive/reproduce.json'),
            expected_formal_conditions=12,expected_bridge_conditions=4,
            gpu_hours_limit=self.args.gpu_hours,formal_gpu_experiments_started=False,
            actual_training_steps=0,forecast_note='Full native runtime not yet measured; cap is not an ETA.')
        c.locked_json(self.run/'audit_report.json',result)
        self.record('audit','COMPLETE',result)

    def collection(self,path):
        path=Path(path);root=path.parent;co=d.read(path)
        if root.parent!=self.run/'collections' or co['contract']!=d.artifact(self.run/'run_contract.json'):
            raise RuntimeError('Foreign collection cannot execute')
        if (root/'audit.json').exists():
            audit.checked(root,full=True);self.record(co['id'],'REUSED_AUDITED');return
        archived=archive_incomplete_collection(root)
        if archived:
            self.record(co['id'],'ARCHIVED_PARTIAL',archived)
        env=dict(self.env,SELECTOR_SNAPSHOT_NOISE=co['noise'],SELECTOR_SNAPSHOT_MODE=co['mode'],
                 SELECTOR_SNAPSHOT_CONDITION=co['id'])
        command=[env['SIMENGINE_PYTHON'],self.sim/'worldengine/runner/run_simulation.py',
            'debug_mode=True','debug_scene_name=null',f"data_file_path={co['scenario']['path']}",
            f"asset_folder_path={co['asset_folder']}",f'output_dir={root}/__WORKER_ID__/WE_output',
            f"job_name=snapshot_{co['id']}",'use_planner_actions=true','ego_policy=env_input_policy',
            'ego_client=navformer_client','ego_controller=log_play_controller','ego_navigation=trajectory_navigation',
            'agent_policy='+('idm_policy' if co['mode']=='R' else 'trajectory_policy'),
            'agent_navigation='+('idm_navigation' if co['mode']=='R' else 'trajectory_navigation'),
            f'planner_data_path={root}/__WORKER_ID__/plan_traj',
            f'planner_client_folder={root}/__WORKER_ID__/frames',
            'with_metric_manager=true','with_dense_reward_manager=false','selector_snapshot=true',
            f'selector_snapshot_manifest={path}',f'selector_snapshot_manifest_sha256={c.sha256_file(path)}',
            f'diffusiondrive_candidate_sidecar_path={root}/__WORKER_ID__/diffusiondrive_candidate_sidecars',
            'diffusiondrive_sidecar_timeout_s=300','distributed_mode=SCENARIO_BASED',
            'worker=ray_distributed','worker_id_prefix=split_','enable_resume=false',
            f'completed_scenarios_dir={root}/__WORKER_ID__/completed_scenarios']
        jobs=[(command,self.sim,env,root/'logs/simulator.log')]
        for index,device in enumerate(self.devices):
            worker=root/f'split_{index}'
            for name in ('frames','plan_traj','merged_ann_files','diffusiondrive_candidate_sidecars'):
                (worker/name).mkdir(parents=True,exist_ok=True)
            cmd=[env['ALGENGINE_PYTHON'],self.alg/'closed_loop/sim_test.py',self.config,
                 d.verify(co['checkpoint']),'--seed','0','--log-dir',worker,'--cfg-options',
                 f'sim.monitored_folder={worker}/frames',f'sim.plan_save_path={worker}/plan_traj',
                 f'sim.merged_ann_save_dir={worker}/merged_ann_files',
                 f'sim.diffusiondrive_rollout_sidecar_path={worker}/diffusiondrive_candidate_sidecars',
                 'sim.clean_temp_files=False','sim.clean_record_data=False',
                 f'data_root={worker}/WE_output/openscene_format/']
            jobs.append((cmd,self.alg,dict(env,CUDA_VISIBLE_DEVICES=device),root/f'logs/planner{index}.log'))
        self.execute(jobs,co['id'],timeout=43200)
        self.helper('audit','--collection',path)

    def bridge(self):
        for policy in s.POLICIES:
            for mode in ('NR','R'):
                self.collection(self.run/'collections'/f'bridge_{mode}_{policy}'/'collection.json')
        self.helper('bridge')
        self.record('bridge','COMPLETE','Native export matches prior routed deployment')

    def evaluate(self):
        b=d.read(self.run/'bridge_report.json')
        if b['status']!='PASS' or b['inputs']!=d.artifact(self.run/'inputs.json'):
            raise RuntimeError('Bridge required before full evaluation')
        for entry in b['conditions']:
            d.verify(entry)
        for policy,block in s.conditions():
            if block.startswith('open'):
                self.helper('open','--policy',policy,'--block',block)
            else:
                self.collection(self.run/'collections'/f'{block}_{policy}'/'collection.json')
        self.helper('report')
        self.record('eval','COMPLETE','Complete snapshot retained irrespective of effect')

    def dispatch(self):
        if self.args.stage=='audit':
            return self.cpu_audit()
        if self.args.stage=='report':
            from report_selector_snapshot import report
            result=report(self.run);print({k:v for k,v in result.items() if k in ('status','decision','missing')});return
        a=d.read(self.run/'audit_report.json')
        if a['status']!='PASS' or a['code_sha']!=self.code_sha:
            raise RuntimeError('Current CPU audit required')
        for key in ('inputs','export','cached_parity'):
            d.verify(a[key])
        s.checked_inputs(self.run)
        d.verify(d.read(self.run/'run_contract.json')['scorer'])
        self.preflight()
        self.helper('parity','--device','cuda')
        env=dict(self.env,SELECTOR_SNAPSHOT_NOISE=s.FORMAL_NOISE,
                 SELECTOR_SNAPSHOT_MODE='NR',SELECTOR_SNAPSHOT_CONDITION='preflight')
        self.helper('config',env=env)
        if self.args.stage=='preflight':
            self.record('preflight','COMPLETE','No formal rollout/training');return
        if self.args.stage in ('bridge','all'):
            self.bridge()
        if self.args.stage in ('eval','all'):
            self.evaluate()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('audit','preflight','bridge','eval','report','all'))
    p.add_argument('run_id')
    p.add_argument('--source-run',type=Path,default=SOURCE)
    p.add_argument('--source-worldengine-root',type=Path,default=CANONICAL)
    p.add_argument('--gpus',type=int,choices=(4,),default=4)
    p.add_argument('--gpu-hours',type=float,default=192.)
    a=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}',a.run_id):
        p.error('Safe run ID required')
    if not math.isfinite(a.gpu_hours) or a.gpu_hours<=0:
        p.error('Positive finite budget required')
    a.source_run=a.source_run.resolve();a.source_worldengine_root=a.source_worldengine_root.resolve()
    runner=Runner(a)
    try:
        runner.dispatch()
    except BaseException as error:
        runner.stop_children();runner.record(a.stage,'STOPPED',str(error));raise
    finally:
        runner.charge_idle(enforce=False)


if __name__=='__main__':
    main()
