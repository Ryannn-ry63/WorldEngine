"""Independent four-H100 diagnosis -> fixed four-arm pilot. Never runs confirmation."""
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
import selector_decision_common as x
import selector_decision_data as data
import selector_decision_collections as collections
from run_selector_cfpi import ROOT, SCRIPT, CANONICAL, environment, implementation_inventory
from run_selector_cfpi_deployment import Runner as BaseRunner, check_budget
from run_selector_feedback import Runner as FeedbackRunner
from run_selector_rare import forecast
from report_selector_cfpi_deployment import checked_collection

PRIOR = CANONICAL/'experiments/worktrees/WorldEngine-selector-feedback-v2/experiments/diffusiondrive/selector_feedback_v2/runs/rare_feedback_20260907_4gpu_v1'


def history_samples(prior):
    ledger = d.read(prior/'decision_ledger.json')
    samples = ledger.get('runtime_samples', [])
    if not samples:
        raise RuntimeError('No source-run allocation-aware runtime samples')
    return f.normalized_runtime_samples(samples)


def estimate_collection(samples, scenes, mode, allocation=4):
    # Historical model gives 8-GPU-normalized cost; small batches use fewer workers
    # but still charge the WHOLE reserved allocation. Conservative, not a promise.
    workers = min(allocation, scenes)
    if workers <= 0:
        return 0.
    return forecast(samples, scenes)*(1.2 if mode == 'NR' else 1.)*allocation/workers


class Runner(BaseRunner):
    charge_idle = FeedbackRunner.charge_idle

    def __init__(self, args):
        self.args = args
        self.phase_limits = x.budget_limits(args.gpu_hours)
        self.run = ROOT/'experiments/diffusiondrive/selector_decision_feedback_v1/runs'/args.run_id
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run/'runner.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.phase = 'diagnose'
        self.account_idle = args.stage not in ('audit','report')
        self.idle_since = time.monotonic()
        self.env = environment(args.source_worldengine_root)
        self.devices = f.gpu_devices(self.env, args.gpus)
        self.env.update(CUDA_VISIBLE_DEVICES=','.join(self.devices),
                        WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP=','.join(self.devices),
                        DIFFUSIONDRIVE_FEEDBACK_RESEARCH_METHOD=x.METHOD)
        self.alg, self.sim = Path(self.env['ALGENGINE_ROOT']), Path(self.env['SIMENGINE_ROOT'])
        self.config = self.alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_decision_feedback.py'
        self.collection_timeout = 43200
        self.checkpoint_manifest = args.source_worldengine_root/'experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json'
        self.inventory = implementation_inventory()
        for folder in ('research/decision_feedback','research/feedback'):
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and p.suffix in ('.py','.md','.json','.sh'):
                    self.inventory[str(p.relative_to(ROOT))] = c.sha256_file(p)
        launcher = ROOT/'run_diffusiondrive_selector_decision_feedback.sh'
        self.inventory[launcher.name] = c.sha256_file(launcher)
        self.code_sha = hashlib.sha256(json.dumps(self.inventory,sort_keys=True).encode()).hexdigest()
        source_snapshot = {p:d.artifact(args.prior_run/p) for p in ('run_contract.json','screen_report.json','decision_ledger.json')}
        contract = dict(method=x.METHOD, base_snapshot_commit='aab4752', code_sha=self.code_sha,
                        implementation=self.inventory, source_snapshot=source_snapshot,
                        source_worldengine_root=str(args.source_worldengine_root), prior_run=str(args.prior_run),
                        gpu_count=args.gpus, gpu_hour_limit=args.gpu_hours, phase_limits=self.phase_limits,
                        training=x.TRAIN, diagnosis=x.DIAGNOSIS, arms=list(x.ARMS), seeds=list(x.SEEDS),
                        modes=list(x.MODES), exposure=x.EXPOSURE, inference_uses_reward_or_q=False,
                        confirmation_authorized=False, endpoint='pilot_report',
                        accounting='All four reserved GPUs during allocation stages, including CPU work; audit/report outside allocation uncharged')
        c.locked_json(self.run/'run_contract.json',contract)
        self.ledger_path = self.run/'decision_ledger.json'
        self.ledger = d.read(self.ledger_path) if self.ledger_path.exists() else dict(
            run_id=args.run_id, gpu_hours_used=0., phase_gpu_hours={k:0. for k in self.phase_limits},
            phase_limits=self.phase_limits, events=[], active=None)
        if (self.ledger['phase_limits'] != self.phase_limits
                or set(self.ledger['phase_gpu_hours']) != set(self.phase_limits)
                or any(not math.isfinite(v) or v < 0 for v in self.ledger['phase_gpu_hours'].values())
                or not math.isfinite(self.ledger['gpu_hours_used'])
                or abs(sum(self.ledger['phase_gpu_hours'].values())-self.ledger['gpu_hours_used']) > 1e-6):
            raise RuntimeError('Invalid ledger; budget never resets')
        self.children = []
        self.recover_accounting()
        signal.signal(signal.SIGINT,self.interrupt)
        signal.signal(signal.SIGTERM,self.interrupt)
        self.samples = history_samples(args.prior_run)
        self.idle_since = time.monotonic()

    def execute(self, jobs, stage, gpus=0, timeout=21600):
        # Explicit base invocation: reusing FeedbackRunner.execute via assignment
        # would bind its zero-argument super() to the wrong class.
        self.charge_idle()
        try:
            return BaseRunner.execute(self,jobs,stage,gpus,timeout)
        finally:
            self.idle_since = time.monotonic()

    def set_phase(self, phase):
        self.charge_idle()
        self.phase = phase

    def task(self, action, *argv):
        # Absolute source paths can exceed NAME_MAX when embedded in a log name.
        label = action+'_'+x.digest(list(map(str,argv)))[:12]
        self.python(SCRIPT/'selector_decision_tasks.py', [action,'--run-root',self.run,*argv],
                    stage=label,
                    timeout=7200)

    def collection(self, entry):
        contract = d.verified_read(entry)
        root = Path(entry['path']).parent
        if root.parent != self.run/'collections' or contract['research_method'] != x.METHOD:
            raise RuntimeError('Cannot execute or rewrite a historical/foreign collection')
        if (root/'collection_audit.json').exists():
            checked_collection(entry)
            self.record(root.name,'REUSED_AUDITED')
            return
        scenes = len(d.verified_read(contract['routing'])['routes'])
        estimate = estimate_collection(self.samples, scenes, contract['react_type'], self.args.gpus)
        check_budget(self.ledger,self.phase,self.args.gpu_hours,estimate)
        self.record(root.name,'FORECAST',dict(gpu_hours=estimate,wall_hours=estimate/self.args.gpus,
                    workers=contract['worker_count'],reserved_gpus=self.args.gpus,scenes=scenes))
        start = time.monotonic()
        BaseRunner.collection(self,entry)
        self.ledger.setdefault('runtime_samples',[]).append(dict(scenes=scenes,seconds=time.monotonic()-start,
                          gpu_count=self.args.gpus,worker_count=contract['worker_count']))
        c.atomic_json(self.ledger_path,self.ledger)

    def prechecks(self, pairs):
        for index, pair in enumerate(pairs):
            reference = d.verified_read(pair['reference'])
            if reference.get('research_method') == x.METHOD:
                self.collection(pair['reference'])
            elif checked_collection(pair['reference']) is None:
                raise RuntimeError('Historical sentinel reference incomplete')
            self.collection(pair['sentinel'])
            self.task('sentinel','--entry',pair['sentinel']['path'],'--reference',pair['reference']['path'])

    def diagnose(self):
        self.set_phase('diagnose')
        self.task('diagnostic_bundle')
        bundle = d.read(self.run/'diagnose/collections.json')
        self.prechecks(bundle['prechecks'])
        for entry in bundle['treatments']:
            self.collection(entry)
        self.task('diagnostic_report')
        result = d.read(self.run/'diagnose/report.json')
        self.record('diagnose','COMPLETE',result['gate']['decision'])
        return result

    def training(self, arm, seed, end):
        self.set_phase('train')
        report = self.run/'train'/f'{arm}_seed{seed}'/f'step_{end}_report.json'
        if not report.exists():
            check_budget(self.ledger,self.phase,self.args.gpu_hours,.35)
        self.python(SCRIPT/'train_selector_decision.py',
                    ['--run-root',self.run,'--arm',arm,'--seed',str(seed),'--end',str(end)],
                    stage=f'train_{arm}_s{seed}_t{end}',
                    env=dict(self.env,CUDA_VISIBLE_DEVICES=self.devices[0]),timeout=7200)

    def pilot(self):
        collections.require_pilot(self.run)
        for seed in x.SEEDS:
            if seed:
                self.training('shared',seed,500)
            for end in (100,300,500):
                for arm in x.ARMS:
                    self.training(arm,seed,end)
                if end in x.QUERY_STEPS:
                    self.set_phase('query')
                    # Freeze both schedules BEFORE collecting either arm's returns.
                    for arm in ('P3','P2'):
                        self.task('query_bundle','--arm',arm,'--seed',str(seed),'--step',str(end))
                    for arm in ('P3','P2'):
                        generation = x.QUERY_STEPS.index(end)+1
                        bundle = d.read(self.run/'queries'/f'{arm}_seed{seed}'/f'g{generation}'/'collections.json')
                        self.prechecks(bundle['prechecks'])
                        for entry in bundle['collections']:
                            self.collection(entry)
                        self.task('query_cache','--arm',arm,'--seed',str(seed),'--generation',str(generation))
        self.set_phase('eval')
        self.task('evaluation_bundle')
        for entry in d.read(self.run/'pilot_collections.json')['collections']:
            self.collection(entry)
        self.task('report')
        result = d.read(self.run/'pilot_report.json')
        self.record('pilot','COMPLETE',result['decision']+'; no confirmation authorized')

    def audit(self):
        data.freeze(self.run,self.args.prior_run)
        data.preflight(self.run,'cpu',full_baselines=True)
        spec = data.freeze_diagnostic(self.run)
        bundle = collections.diagnostic_bundle(self.run)
        entries = [p['sentinel'] for p in bundle['prechecks']]+bundle['treatments']
        entries += [p['reference'] for p in bundle['prechecks']
                    if d.verified_read(p['reference'])['research_method'] == x.METHOD]
        diagnostic_cost = sum(estimate_collection(self.samples,len(d.verified_read(d.verified_read(e)['routing'])['routes']),
                                  d.verified_read(e)['react_type']) for e in entries)
        inputs = data.checked_inputs(self.run)
        eval_cost = sum(estimate_collection(self.samples,len(inputs['cohorts'][cohort]['rows']),mode)
                        for cohort,mode,_ in x.evaluation_conditions())
        query_max = sum(estimate_collection(self.samples,n,mode)
                        for _ in range(3*2*2) for mode in x.MODES for n in (8,32))
        forecast_report = dict(status='PASS', diagnostic_targets=spec['summary'],
            gpu_hours=dict(diagnose=diagnostic_cost, query_upper_estimate=query_max, training_allowance=8., evaluation=eval_cost),
            total_upper_estimate=diagnostic_cost+query_max+8.+eval_cost,
            total_upper_wall_hours=(diagnostic_cost+query_max+8.+eval_cost)/4,
            gpu_hour_limit=self.args.gpu_hours, phase_limits=self.phase_limits, gpu_count=4,
            diagnostic_wall_hours=diagnostic_cost/4, source_samples=len(self.samples),
            fits_upper_estimate=diagnostic_cost+query_max+8.+eval_cost < self.args.gpu_hours,
            note='Query counts adaptive; small-batch startup estimated conservatively. No automatic budget expansion or truncation.')
        c.locked_json(self.run/'budget_forecast.json',forecast_report)
        c.locked_json(self.run/'audit_report.json',dict(status='PASS',code_sha=self.code_sha,
                      inputs=d.artifact(self.run/'decision_inputs.json'),
                      cpu_preflight=d.artifact(self.run/'decision_preflight_cpu.json'),
                      source_screen=d.artifact(self.args.prior_run/'screen_report.json'),
                      forecast=d.artifact(self.run/'budget_forecast.json'), formal_training_performed=False))
        self.record('audit','COMPLETE',forecast_report)

    def dispatch(self):
        if self.args.stage == 'audit':
            return self.audit()
        if self.args.stage == 'report':
            from report_selector_decision import report
            result = report(self.run)
            print(json.dumps(result if result['status'] != 'PASS' else
                  {k:result[k] for k in ('status','decision','effects')}),flush=True)
            return
        audit = d.read(self.run/'audit_report.json')
        if audit['status'] != 'PASS' or audit['code_sha'] != self.code_sha:
            raise RuntimeError('Run CPU audit before reserving GPUs')
        for key in ('inputs','cpu_preflight','source_screen','forecast'):
            d.verify(audit[key])
        if self.args.stage == 'pilot':
            collections.require_pilot(self.run)  # fail before CUDA launch
        prior_report = self.run/'diagnose/report.json'
        if self.args.stage == 'all' and prior_report.exists() and not d.read(prior_report)['gate']['passed']:
            self.record('all','STOPPED_AT_DIAGNOSTIC_GATE',d.read(prior_report)['gate']['decision'])
            return
        self.preflight()
        self.task('cached_preflight','--device','cuda')
        if self.args.stage == 'preflight':
            self.record('preflight','COMPLETE','No training or simulation launched')
            return
        if self.args.stage in ('diagnose','all'):
            result = self.diagnose()
            if not result['gate']['passed']:
                self.record(self.args.stage,'STOPPED_AT_DIAGNOSTIC_GATE',result['gate']['decision'])
                return
        if self.args.stage in ('pilot','all'):
            self.pilot()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('audit','preflight','diagnose','pilot','all','report'))
    p.add_argument('run_id')
    p.add_argument('--prior-run',type=Path,default=PRIOR)
    p.add_argument('--source-worldengine-root',type=Path,default=CANONICAL)
    p.add_argument('--gpus',type=int,choices=(4,),default=4)
    p.add_argument('--gpu-hours',type=float,default=128.)
    a = p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}',a.run_id):
        p.error('Safe run ID required')
    try:
        x.budget_limits(a.gpu_hours)
    except ValueError as error:
        p.error(str(error))
    a.prior_run, a.source_worldengine_root = a.prior_run.resolve(),a.source_worldengine_root.resolve()
    runner = Runner(a)
    try:
        runner.dispatch()
    except BaseException as error:
        runner.stop_children()
        runner.record(a.stage,'STOPPED',str(error))
        raise
    finally:
        runner.charge_idle(enforce=False)


if __name__ == '__main__':
    main()
