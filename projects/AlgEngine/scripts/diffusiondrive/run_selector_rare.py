"""Independent staged rare-first runner: bounded 64 GPUh, immutable three-seed winner."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import time
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_rare_common as r
from run_selector_cfpi import ROOT, SCRIPT, CANONICAL, environment, implementation_inventory
from run_selector_cfpi_deployment import Runner as DeploymentRunner, check_budget

OLD = CANONICAL/'experiments/worktrees/WorldEngine-selector-cfpi-v1/experiments/diffusiondrive/selector_cfpi_deployment_v1/runs/cfpi_continuous_20260906_v1'


def duration_samples(old):
    from datetime import datetime
    pending, samples = {}, []
    for event in d.read(old/'decision_ledger.json')['events']:
        stage = event['stage']
        if event['status']=='RUNNING':
            pending[stage] = event['time']
        if event['status']=='COMPLETE' and stage in pending:
            path = old/'collections'/stage/'deployment_collection.json'
            if path.exists():
                v = d.read(path)
                n = len(d.verified_read(v['routing'])['routes'])
                seconds = (datetime.fromisoformat(event['time'])-datetime.fromisoformat(pending[stage])).total_seconds()
                samples.append(dict(scenes=n,seconds=seconds))
    if not samples:
        raise RuntimeError('No audited historical runtimes for forecast')
    return samples


def forecast(samples, scenes):
    # Explicit conservative startup + per-scene service estimate; no scene/seed truncation.
    import numpy as np
    rate = float(np.quantile([max(0.,x['seconds']-120.)/x['scenes'] for x in samples],.9))
    return 1.15*(120.+rate*scenes)*8/3600.+.15


class Runner(DeploymentRunner):
    def __init__(self,args):
        self.args = args
        self.run = ROOT/'experiments/diffusiondrive/selector_rare_v1/runs'/args.run_id
        self.run.mkdir(parents=True,exist_ok=True)
        self.lock = (self.run/'runner.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.env = environment(args.source_worldengine_root)
        self.alg,self.sim = Path(self.env['ALGENGINE_ROOT']),Path(self.env['SIMENGINE_ROOT'])
        self.config = self.alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_rare.py'
        self.checkpoint_manifest = args.source_worldengine_root/'experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json'
        self.gate_manifest = args.source_worldengine_root/'experiments/diffusiondrive/grpo_selector_v3_gate_conditioned_v1/models/seed0/checkpoint_manifest.json'
        self.phase = args.stage if args.stage in ('screen','confirm') else 'bridge'
        self.inventory = implementation_inventory()
        for path in (ROOT/'research/rare').rglob('*'):
            if path.is_file() and path.suffix in ('.py','.md','.json','.sh'):
                self.inventory[str(path.relative_to(ROOT))] = c.sha256_file(path)
        self.inventory['run_diffusiondrive_selector_rare_8h100.sh'] = c.sha256_file(ROOT/'run_diffusiondrive_selector_rare_8h100.sh')
        self.code_sha = hashlib.sha256(json.dumps(self.inventory,sort_keys=True).encode()).hexdigest()
        contract = dict(method=r.METHOD,transport_method=d.METHOD,code_sha=self.code_sha,implementation=self.inventory,
                        source_worldengine_root=str(args.source_worldengine_root),prior_deployment=str(args.deployment_run),
                        checkpoint_sha256=c.CHECKPOINT_SHA256,noise_namespace=c.NOISE_NAMESPACE,
                        gpu_count=8,gpu_hour_limit=args.gpu_hours,phase_limits=r.LIMITS,policies=list(r.POLICIES),
                        seeds=list(r.SEEDS),training=r.TRAIN,exposure=r.EXPOSURE,base_commit='26ab020',
                        effect_thresholds=dict(rare_pdm=.03,common_max_drop=.01,positive_seeds=2),
                        budget_accounting='All 8 reserved GPUs during launched stages; CPU-only audit/report outside allocation uncharged.',
                        buffer_rule='4 GPUh shared phase overrun/retry reserve; never resets total64.',
                        independent_stage_boundaries=True,new_branch_feedback_authorized=False,
                        model_selection='Only step500; one method/all three seeds; no refit/replacement after selection')
        c.locked_json(self.run/'run_contract.json',contract)
        self.ledger_path = self.run/'decision_ledger.json'
        self.ledger = d.read(self.ledger_path) if self.ledger_path.exists() else dict(
            run_id=args.run_id,gpu_hours_used=0.,phase_gpu_hours={k:0. for k in r.LIMITS},
            phase_limits=r.LIMITS,events=[],active=None)
        if (self.ledger.get('phase_limits')!=r.LIMITS
                or set(self.ledger['phase_gpu_hours'])!=set(r.LIMITS)
                or any(not math.isfinite(v) or v<0 for v in self.ledger['phase_gpu_hours'].values())
                or not math.isfinite(self.ledger['gpu_hours_used'])
                or abs(sum(self.ledger['phase_gpu_hours'].values())-self.ledger['gpu_hours_used'])>1e-6):
            raise RuntimeError('Cumulative ledger invalid; refusing budget reset')
        self.children = []
        self.recover_accounting()
        signal.signal(signal.SIGINT,self.interrupt)
        signal.signal(signal.SIGTERM,self.interrupt)

    def prepare_rare(self,stage):
        self.python(SCRIPT/'prepare_selector_rare.py',[stage,'--run-root',self.run,
                    '--deployment-run',self.args.deployment_run,'--canonical',self.args.source_worldengine_root],
                    stage='prepare_'+stage)

    def preflight(self):
        assets = self.args.source_worldengine_root/'data/sim_engine/assets/navtest_failures/assets'
        if not assets.is_dir():
            raise RuntimeError('Missing rare assets: '+str(assets))
        return super().preflight()

    def do_report(self,phase):
        from report_selector_rare import report
        return report(self.run,phase)

    def collection(self,entry):
        contract = d.verified_read(entry)
        root = Path(entry['path']).parent
        if not (root/'collection_audit.json').exists():
            samples = duration_samples(self.args.deployment_run)+self.ledger.get('runtime_samples',[])
            scenes = len(d.verified_read(contract['routing'])['routes'])
            estimate = forecast(samples,scenes)
            self.record(contract['collection_id'],'FORECAST',dict(gpu_hours=estimate,scenes=scenes))
            check_budget(self.ledger,self.phase,self.args.gpu_hours,estimate)
            start = time.monotonic()
            original_env = self.env
            self.env = dict(original_env,DIFFUSIONDRIVE_RARE_EVAL_COHORT=contract['cohort'])
            try:
                super().collection(entry)
            finally:
                self.env = original_env
            self.ledger.setdefault('runtime_samples',[]).append(dict(scenes=scenes,seconds=time.monotonic()-start))
            c.atomic_json(self.ledger_path,self.ledger)
        else:
            # Full read-only hash validation without burning a new rollout.
            from report_selector_cfpi_deployment import checked_collection
            if checked_collection(entry) is None:
                raise RuntimeError('Incomplete resume audit')
            self.record(contract['collection_id'],'REUSED_AUDITED')

    def dispatch(self):
        from prepare_selector_rare import freeze
        if self.args.stage=='report':
            results = [self.do_report(p) for p in ('bridge','screen','confirm')]
            print(json.dumps([{k:v for k,v in x.items() if k in ('status','phase','decision','winner')} for x in results],indent=2))
            return
        if self.args.stage=='audit':
            inputs = freeze(self.run,self.args.deployment_run,self.args.source_worldengine_root)
            samples = duration_samples(self.args.deployment_run)
            estimates = dict(bridge=3*forecast(samples,64),screen=17*forecast(samples,58)+9*forecast(samples,64),
                             confirm=5*forecast(samples,230))
            c.atomic_json(self.run/'budget_forecast.json',dict(rollout_gpu_hours=estimates,includes_training=False,
                          warning='Forecast is not permission to exceed64; stop before conditions that do not fit.'))
            self.record('audit','COMPLETE',dict(replay=inputs['replay_capacity'],forecast=estimates))
            return
        # Prerequisites checked BEFORE taking an instance through expensive preflight.
        if self.args.stage=='screen':
            if not (self.run/'bridge_report.json').exists() or self.do_report('bridge')['status']!='PASS':
                raise RuntimeError('Screen requires all three completed/audited bridge conditions')
        if self.args.stage=='confirm':
            screen = self.do_report('screen')
            if screen.get('decision')!='PROCEED_FIXED_WINNER':
                self.record('confirm','NOT_AUTHORIZED','No fully passing fixed winner')
                return
            winner = d.read(self.run/'winner.json')
            d.verify(winner['screen_report'])
            for model in winner['models'].values():
                d.verify(model)
        self.preflight()
        self.prepare_rare('freeze')
        self.prepare_rare('baselines')
        self.python(SCRIPT/'preflight_selector_rare.py',['--run-root',self.run],stage='rare_cached_preflight',timeout=1200)
        if self.args.stage=='preflight':
            self.record('preflight','COMPLETE','No rollout/training launched; next boundary is bridge')
            return
        if self.args.stage=='bridge':
            self.python(SCRIPT/'diagnose_selector_rare.py',['--run-root',self.run],stage='bridge_cached_diagnostic',timeout=1200)
        if self.args.stage=='screen':
            devices = self.env.get('CUDA_VISIBLE_DEVICES',','.join(map(str,range(8)))).split(',')
            jobs = []
            for method in r.POLICIES:
                for seed in r.SEEDS:
                    policy = f'{method}_seed{seed}'
                    env = dict(self.env,CUDA_VISIBLE_DEVICES=devices[len(jobs)%8])
                    cmd = [env['ALGENGINE_PYTHON'],SCRIPT/'train_selector_rare.py','--run-root',self.run,
                           '--method',method,'--seed',str(seed)]
                    jobs.append((cmd,ROOT,env,self.run/'logs'/f'train_{policy}.log'))
            for offset in range(0,len(jobs),8):
                check_budget(self.ledger,self.phase,self.args.gpu_hours,.5)
                self.execute(jobs[offset:offset+8],f'train_batch_{offset//8}',timeout=7200)
        self.prepare_rare(self.args.stage)
        for entry in d.read(self.run/f'{self.args.stage}_collections.json')['collections']:
            self.collection(entry)
        # On-allocation reporting remains budget charged and can later be recovered CPU-only.
        self.python(SCRIPT/'report_selector_rare.py',['--run-root',self.run,'--phase',self.args.stage],
                    stage=self.args.stage+'_report')
        result = d.read(self.run/f'{self.args.stage}_report.json')
        self.record(self.args.stage,'COMPLETE',result['decision']+'; stopped at requested stage boundary')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('audit','preflight','bridge','screen','confirm','report'))
    p.add_argument('run_id')
    p.add_argument('--deployment-run',type=Path,default=OLD)
    p.add_argument('--source-worldengine-root',type=Path,default=CANONICAL)
    p.add_argument('--gpus',type=int,default=8)
    p.add_argument('--gpu-hours',type=float,default=64.)
    a = p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}',a.run_id) or a.gpus!=8:
        p.error('Safe new run ID and exactly8 H100 GPUs required')
    if not math.isfinite(a.gpu_hours) or not 0<a.gpu_hours<=64:
        p.error('Budget must be finite and in (0,64]')
    a.deployment_run,a.source_worldengine_root = a.deployment_run.resolve(),a.source_worldengine_root.resolve()
    runner = Runner(a)
    try:
        runner.dispatch()
    except BaseException as error:
        runner.stop_children()
        runner.record(a.stage,'STOPPED',str(error))
        raise


if __name__=='__main__':
    main()
