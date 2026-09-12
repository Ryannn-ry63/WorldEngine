"""4/8-H100, explicit cumulative budget; fixed two-round experiment with gated unattended continuation."""
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
import selector_feedback_common as f
from run_selector_cfpi import ROOT,SCRIPT,CANONICAL,environment,implementation_inventory
from run_selector_cfpi_deployment import Runner as BaseRunner,check_budget,charge
from run_selector_rare import duration_samples,forecast

PRIOR = CANONICAL/'experiments/worktrees/WorldEngine-selector-rare-v1/experiments/diffusiondrive/selector_rare_v1/runs/rare_retention_20260906_v1'


class Runner(BaseRunner):
    def __init__(self,args):
        self.args = args
        self.phase_limits = f.budget_limits(args.gpu_hours)
        self.run = ROOT/'experiments/diffusiondrive/selector_feedback_v2/runs'/args.run_id
        self.run.mkdir(parents=True,exist_ok=True)
        self.lock = (self.run/'runner.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.env = environment(args.source_worldengine_root)
        self.devices = f.gpu_devices(self.env,args.gpus)
        self.env.update(CUDA_VISIBLE_DEVICES=','.join(self.devices),
                        WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP=','.join(self.devices))
        self.collection_timeout = 21600*8//args.gpus
        self.alg,self.sim = Path(self.env['ALGENGINE_ROOT']),Path(self.env['SIMENGINE_ROOT'])
        self.config = self.alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_feedback.py'
        self.checkpoint_manifest = args.source_worldengine_root/'experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json'
        self.inventory = implementation_inventory()
        for p in (ROOT/'research/feedback').rglob('*'):
            if p.is_file() and p.suffix in ('.md','.py','.json','.sh'):
                self.inventory[str(p.relative_to(ROOT))] = c.sha256_file(p)
        self.inventory['run_diffusiondrive_selector_feedback_8h100.sh'] = c.sha256_file(ROOT/'run_diffusiondrive_selector_feedback_8h100.sh')
        self.inventory['run_diffusiondrive_selector_feedback.sh'] = c.sha256_file(ROOT/'run_diffusiondrive_selector_feedback.sh')
        self.code_sha = hashlib.sha256(json.dumps(self.inventory,sort_keys=True).encode()).hexdigest()
        contract = dict(method=f.METHOD,source_worldengine_root=str(args.source_worldengine_root),
            prior_run=str(args.prior_run),code_sha=self.code_sha,implementation=self.inventory,
            base_commit='0c237e2',gpu_count=args.gpus,gpu_hour_limit=args.gpu_hours,phase_limits=self.phase_limits,
            collection_timeout_seconds=self.collection_timeout,
            training=f.TRAIN,modes=list(f.MODES),arms=list(f.ARMS),feedback_collection_seed=0,
            source_strata=f.STRATA,source_maximum_per_log=2,noise_namespace=c.NOISE_NAMESPACE,
            checkpoint_sha256=c.CHECKPOINT_SHA256,exposure=f.EXPOSURE,
            endpoint='final_development_report',confirmation_authorized=False,
            accounting=f'{args.gpus} reserved GPUs during all launched stages; CPU audit outside allocation is uncharged')
        # Round-trip JSON avoids tuple/list mismatches on resumed contracts.
        c.locked_json(self.run/'run_contract.json',json.loads(json.dumps(contract)))
        self.ledger_path = self.run/'decision_ledger.json'
        self.ledger = d.read(self.ledger_path) if self.ledger_path.exists() else dict(
            run_id=args.run_id,gpu_hours_used=0.,phase_gpu_hours={k:0. for k in self.phase_limits},
            phase_limits=self.phase_limits,events=[],active=None)
        if (self.ledger['phase_limits']!=self.phase_limits or set(self.ledger['phase_gpu_hours'])!=set(self.phase_limits)
            or any(not math.isfinite(x) or x<0 for x in self.ledger['phase_gpu_hours'].values())
            or not math.isfinite(self.ledger['gpu_hours_used'])
            or abs(sum(self.ledger['phase_gpu_hours'].values())-self.ledger['gpu_hours_used'])>1e-6):
            raise RuntimeError('Invalid cumulative ledger; budget cannot reset')
        self.children = []
        self.phase = 'feedback'
        self.recover_accounting()
        signal.signal(signal.SIGINT,self.interrupt)
        signal.signal(signal.SIGTERM,self.interrupt)
        old = d.read(args.prior_run/'rare_inputs.json')
        self.prior_deployment = Path(old['artifacts']['old_inputs']['path']).parent
        self.samples = duration_samples(self.prior_deployment)
        self.account_idle = args.stage not in ('audit','report')
        self.idle_since = time.monotonic()

    def charge_idle(self, enforce=True):
        if not self.account_idle:
            return
        now = time.monotonic()
        charge(self.ledger,self.phase,now-self.idle_since,self.args.gpus)
        self.idle_since = now
        c.atomic_json(self.ledger_path,self.ledger)
        if enforce:
            check_budget(self.ledger,self.phase,self.args.gpu_hours)

    def execute(self,jobs,stage,gpus=0,timeout=21600):
        self.charge_idle()
        try:
            return super().execute(jobs,stage,gpus,timeout)
        finally:
            # BaseRunner charged the complete child interval, including shutdown.
            self.idle_since = time.monotonic()

    def prepare_feedback(self,stage):
        self.python(SCRIPT/'prepare_selector_feedback.py',[stage,'--run-root',self.run,'--prior-run',self.args.prior_run,
                    '--canonical',self.args.source_worldengine_root],stage='prepare_'+stage,timeout=3600)

    def build(self,stage,round_name,arm,collection=None):
        argv = [stage,'--run-root',self.run,'--round',round_name,'--arm',arm]
        if collection:
            argv += ['--collection',collection]
        self.python(SCRIPT/'build_selector_feedback.py',argv,stage=f'{round_name}_{arm}_{stage}',timeout=3600)

    def collection(self,entry):
        from report_selector_cfpi_deployment import checked_collection
        contract = d.verified_read(entry)
        root = Path(entry['path']).parent
        if (root/'collection_audit.json').exists():
            # Recheck bytes/identities; do not start new simulator workers.
            checked_collection(entry)
            self.record(root.name,'REUSED_AUDITED')
            return
        scenes = len(d.verified_read(contract['routing'])['routes'])
        samples = self.samples+self.ledger.get('runtime_samples',[])
        estimate = forecast(f.normalized_runtime_samples(samples),scenes)*(1.2 if contract['react_type']=='NR' else 1.)
        check_budget(self.ledger,self.phase,self.args.gpu_hours,estimate)
        self.record(root.name,'FORECAST',dict(gpu_hours=estimate,wall_hours=estimate/self.args.gpus,
                    gpu_count=self.args.gpus,scenes=scenes,mode=contract['react_type']))
        start = time.monotonic()
        super().collection(entry)
        self.ledger.setdefault('runtime_samples',[]).append(dict(scenes=scenes,seconds=time.monotonic()-start,
                                                              gpu_count=self.args.gpus))
        c.atomic_json(self.ledger_path,self.ledger)

    def training(self,arm,seed):
        folder = self.run/'train'/f'{arm}_seed{seed}'
        if (folder/'report.json').exists():
            report = d.read(folder/'report.json')
            if report['status']!='PASS' or report['provenance']['code_sha']!=self.code_sha:
                raise RuntimeError('Training result identity drift')
            d.verify(report['selector'])
            self.record(folder.name,'REUSED_AUDITED')
            return
        old_phase,self.phase = self.phase,'train'
        try:
            check_budget(self.ledger,self.phase,self.args.gpu_hours,.35)
            self.python(SCRIPT/'train_selector_feedback.py',['--run-root',self.run,'--arm',arm,'--seed',str(seed)],
                        stage='train_'+folder.name,env=dict(self.env,CUDA_VISIBLE_DEVICES=self.devices[0]),timeout=7200)
        finally:
            self.phase = old_phase

    def feedback_round(self,round_name,arm):
        self.build('targets',round_name,arm)
        self.build('collections',round_name,arm)
        inventory = d.read(self.run/'feedback'/round_name/arm/'collections.json')
        # Each mode's visiting sentinel MUST pass before either treatment is run.
        for entry in inventory['collections']:
            self.collection(entry)
            name = Path(entry['path']).parent.name
            if name.endswith('_sentinel'):
                self.build('sentinel',round_name,arm,name)
        self.build('cache',round_name,arm)

    def feedback(self):
        self.phase = 'feedback'
        self.prepare_feedback('baseline')
        for entry in d.read(self.run/'baseline_collections.json')['collections']:
            self.collection(entry)
        self.feedback_round('round1','shared')
        self.training('shared',0)
        self.prepare_feedback('visit')
        for entry in d.read(self.run/'visit_collections.json')['collections']:
            self.collection(entry)
        # Freeze S/U/T acquisition decisions before seeing ANY round2 treatment return.
        for arm in f.ARMS:
            self.build('targets','round2',arm)
        for arm in f.ARMS:
            self.feedback_round('round2',arm)
        self.record('feedback','COMPLETE','Two feedback rounds frozen; no third round authorized')

    def do_report(self,phase):
        self.python(SCRIPT/'report_selector_feedback.py',['--run-root',self.run,'--phase',phase],stage=phase+'_report',timeout=1800)
        return d.read(self.run/(phase+'_report.json'))

    def screen(self):
        self.phase = 'eval'
        for arm in f.ARMS:
            self.training(arm,0)
        self.prepare_feedback('screen')
        for entry in d.read(self.run/'screen_collections.json')['collections']:
            self.collection(entry)
        return self.do_report('screen')

    def final(self):
        from report_selector_feedback import report
        screen = report(self.run,'screen')
        if screen.get('decision')!='PROCEED_T_SEEDS_1_2':
            self.record('final','NOT_AUTHORIZED',screen['decision'])
            return
        self.phase = 'eval'
        for seed in (1,2):
            self.training('shared',seed)
            self.training('T',seed)
        self.prepare_feedback('final')
        for entry in d.read(self.run/'final_collections.json')['collections']:
            self.collection(entry)
        result = self.do_report('final')
        self.record('final','COMPLETE',result['decision']+'; no confirmation automatically authorized')

    def dispatch(self):
        from prepare_selector_feedback import freeze,dense_cache
        from report_selector_feedback import report
        if self.args.stage=='report':
            for phase in ('screen','final'):
                result = report(self.run,phase)
                print({k:result[k] for k in ('status','phase','decision')},flush=True)
            return
        if self.args.stage=='audit':
            inputs = freeze(self.run,self.args.prior_run,self.args.source_worldengine_root)
            dense_cache(self.run,self.args.source_worldengine_root)
            from preflight_selector_feedback import preflight
            preflight(self.run,'cpu',full_baselines=True)
            # Does not promise that all stages fit: startup cost of small interventions matters.
            estimate = dict(feedback=sum(2.2*forecast(self.samples,n) for n in
                            ([128]*2+[8,96,96]+[8,32,32]*3)),
                            evaluation_screen=2.2*5*forecast(self.samples,58),
                            evaluation_final_incremental=2.2*(2*forecast(self.samples,58)+4*forecast(self.samples,64)))
            c.atomic_json(self.run/'budget_forecast.json',dict(gpu_hours=estimate,training_budget=4.,
                          gpu_count=self.args.gpus,wall_hours={k:v/self.args.gpus for k,v in estimate.items()},
                          total_wall_hours_with_training_allowance=(sum(estimate.values())+4.)/self.args.gpus,
                          forecast_basis='Historical 8-GPU GPUh; allocation-normalized live samples; four-GPU scaling not yet measured.',
                          gpu_hour_limit=self.args.gpu_hours,phase_limits=self.phase_limits,
                          includes_all_cpu_overhead=False,
                          warning=f'Fixed quotas; hard stop at {self.args.gpu_hours:g} GPUh. No automatic budget expansion.'))
            self.record('audit','COMPLETE',dict(source=inputs['source_counts'],forecast=estimate))
            return
        if not (self.run/'cached_preflight_cpu.json').exists() or not (self.run/'dense/manifest.json').exists():
            raise RuntimeError('Run CPU audit first, before reserving the GPU instance')
        if self.args.stage=='screen':
            for arm in f.ARMS:
                if not (self.run/'feedback/round2'/arm/'cache_audit.json').exists():
                    raise RuntimeError('All three feedback conditions required before screen')
        if self.args.stage=='final' and report(self.run,'screen').get('decision')!='PROCEED_T_SEEDS_1_2':
            self.record('final','NOT_AUTHORIZED','No passing T screen; GPU preflight not launched')
            return
        self.phase = 'feedback' if self.args.stage in ('preflight','feedback','all') else 'eval'
        self.preflight()
        self.python(SCRIPT/'preflight_selector_feedback.py',['--run-root',self.run,'--device','cuda'],
                    stage='cached_cuda_preflight',timeout=900)
        if self.args.stage=='preflight':
            self.record('preflight','COMPLETE','No simulation or training launched')
            return
        if self.args.stage in ('feedback','all'):
            self.feedback()
        if self.args.stage in ('screen','all'):
            result = self.screen()
            if result['decision']!='PROCEED_T_SEEDS_1_2':
                self.record('all','STOPPED_AT_EFFECT_GATE',result['decision'])
                return
        if self.args.stage in ('final','all'):
            self.final()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('audit','preflight','feedback','screen','final','all','report'))
    p.add_argument('run_id')
    p.add_argument('--prior-run',type=Path,default=PRIOR)
    p.add_argument('--source-worldengine-root',type=Path,default=CANONICAL)
    p.add_argument('--gpus',type=int,default=8,choices=(4,8),
                   help='Reserved H100 count; keep identical across stages (default: 8)')
    p.add_argument('--gpu-hours',type=float,default=64.,
                   help='Explicit cumulative GPUh cap (default: 64); phase allowances scale proportionally. Keep identical across stages.')
    a = p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}',a.run_id):
        p.error('Safe run ID required')
    try:
        f.budget_limits(a.gpu_hours)
    except ValueError as error:
        p.error(str(error))
    a.prior_run,a.source_worldengine_root = a.prior_run.resolve(),a.source_worldengine_root.resolve()
    runner = Runner(a)
    try:
        runner.dispatch()
    except BaseException as error:
        runner.stop_children()
        runner.record(a.stage,'STOPPED',str(error))
        raise
    finally:
        runner.charge_idle(enforce=False)


if __name__=='__main__':
    main()
