"""Compose the FROZEN screen and confirm CLIs; never mutate their contract or budget.

This file deliberately lives outside the experiment's implementation inventory.
Automation provenance is recorded separately. All scientific/gating checks remain
in the original child runner, including its exclusive lock and cumulative ledger.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT/'projects/AlgEngine/scripts/diffusiondrive'
RUNS = ROOT/'experiments/diffusiondrive/selector_rare_v1/runs'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(item):
    path = Path(item['path']).resolve()
    if sha(path)!=item['sha256']:
        raise RuntimeError('Frozen automation input changed: '+str(path))
    return path


def gate(run, contract):
    """A quick fail-closed boundary check; confirm repeats the full original audit."""
    screen = read(run/'screen_report.json')
    if screen.get('status')!='PASS' or screen.get('engineering_pass') is not True or screen.get('phase')!='screen':
        raise RuntimeError('Screen incomplete or engineering audit failed; confirm not started')
    if screen.get('decision')=='STOP_NO_WINNER' and screen.get('winner') is None:
        return 'STOP_NO_WINNER'
    if screen.get('decision')!='PROCEED_FIXED_WINNER':
        raise RuntimeError('Screen did not authorize a fixed winner')
    winner = read(run/'winner.json')
    method = winner.get('method')
    if (method not in contract['policies'] or method!=screen.get('winner')
            or winner.get('refit_authorized') is not False
            or screen['methods'][method]['effect_gate'].get('passed') is not True):
        raise RuntimeError('Invalid winner or effect gate')
    if verify(winner['screen_report'])!=(run/'screen_report.json').resolve():
        raise RuntimeError('Winner belongs to a different screen report')
    if set(winner['models'])!={f'{method}_seed{s}' for s in contract['seeds']}:
        raise RuntimeError('Winner must retain all fixed seeds')
    for item in winner['models'].values():
        verify(item)
    return 'PROCEED_FIXED_WINNER'


class Automation:
    def __init__(self, run_id):
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}',run_id):
            raise ValueError('Invalid run ID')
        self.run_id, self.run = run_id, RUNS/run_id
        # A completed prior audit is required; this entrypoint never creates a new experiment.
        self.contract = read(self.run/'run_contract.json')
        if self.contract.get('method')!='selector_rare_retention_v1' or not (self.run/'rare_inputs.json').exists():
            raise RuntimeError('Complete the original audit first')
        self.contract_sha = sha(self.run/'run_contract.json')
        self.lock = (self.run/'overnight.lock').open('a+')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        self.child, self.cancelled = None, False

    def event(self, status, **detail):
        item = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    stage='screen_then_confirm',status=status,detail=detail)
        with (self.run/'overnight_events.jsonl').open('a') as stream:
            stream.write(json.dumps(item,sort_keys=True)+'\n')
        print(json.dumps(item),flush=True)

    def stop(self, signum, _frame):
        self.cancelled = True
        if self.child is not None and self.child.poll() is None:
            try:
                # The original runner gracefully stops only its own process groups.
                self.child.send_signal(signum)
            except ProcessLookupError:
                pass

    def stage(self, name):
        if self.cancelled:
            raise RuntimeError('Automation cancelled; no next phase')
        if sha(self.run/'run_contract.json')!=self.contract_sha:
            raise RuntimeError('Original run contract changed')
        args = [sys.executable,SCRIPT/'run_selector_rare.py',name,self.run_id,
                '--gpus',str(self.contract['gpu_count']),
                '--gpu-hours',str(self.contract['gpu_hour_limit']),
                '--deployment-run',self.contract['prior_deployment'],
                '--source-worldengine-root',self.contract['source_worldengine_root']]
        self.event('START_CHILD',child_stage=name)
        self.child = subprocess.Popen(list(map(str,args)),cwd=ROOT)
        # Handle a signal arriving between the prelaunch check and child assignment.
        if self.cancelled:
            self.stop(signal.SIGTERM,None)
        code = self.child.wait()
        self.child = None
        if code or self.cancelled:
            raise RuntimeError(f'{name} stopped (exit={code}); no subsequent phase started')

    def execute(self):
        self.event('START',wrapper_sha256=sha(__file__),run_contract_sha256=self.contract_sha,
                   authorized_sequence=['screen','confirm'],budget_reset=False)
        if not (self.run/'screen_report.json').exists():
            bridge_path = self.run/'bridge_report.json'
            if not bridge_path.exists() or read(bridge_path).get('status')!='PASS':
                raise RuntimeError('Bridge not complete: wait for bridge COMPLETE before starting this entrypoint')
            self.stage('screen')
        else:
            # The original confirm CLI rechecks all screen dependencies before using them.
            self.event('EXISTING_SCREEN_REPORT',note='Original confirm will revalidate full hashes and outcomes')
        decision = gate(self.run,self.contract)
        if decision=='STOP_NO_WINNER':
            self.event(decision,confirm_started=False)
            return
        if self.cancelled:
            raise RuntimeError('Automation cancelled before confirm')
        sys.path.insert(0,str(SCRIPT))
        if (self.run/'confirm_report.json').exists():
            from report_selector_rare import report
            result = report(self.run,'confirm')  # CPU-only complete dependency validation.
            if result.get('status')!='PASS':
                raise RuntimeError('Existing confirmation report is incomplete')
            self.event('ALREADY_COMPLETE',decision=result['decision'])
            return
        from run_selector_cfpi_deployment import check_budget
        ledger = read(self.run/'decision_ledger.json')
        if ledger.get('active'):
            raise RuntimeError('A child stage is still active or needs original-runner recovery; no automatic confirm')
        used = ledger['gpu_hours_used']
        if (not math.isfinite(used) or used<0 or ledger.get('phase_limits')!=self.contract['phase_limits']
                or any(not math.isfinite(v) or v<0 for v in ledger['phase_gpu_hours'].values())
                or abs(sum(ledger['phase_gpu_hours'].values())-used)>1e-6):
            raise RuntimeError('Invalid cumulative ledger')
        try:
            check_budget(ledger,'confirm',self.contract['gpu_hour_limit'],15*self.contract['gpu_count']/3600.)
        except RuntimeError as error:
            self.event('STOP_BUDGET',reason=str(error),confirm_started=False)
            return
        self.event('ADVANCE_TO_CONFIRM',winner=read(self.run/'winner.json')['method'],
                   gpu_hours_used=used,note='Original per-condition forecasts and budget stops remain enforced')
        self.stage('confirm')
        result = read(self.run/'confirm_report.json')
        if result.get('status')!='PASS':
            raise RuntimeError('Confirmation did not finish its engineering audit')
        self.event('COMPLETE',decision=result['decision'],gpu_hours_used=read(self.run/'decision_ledger.json')['gpu_hours_used'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id',help='The SAME audited run ID, after bridge COMPLETE')
    args = parser.parse_args()
    runner = Automation(args.run_id)
    signal.signal(signal.SIGINT,runner.stop)
    signal.signal(signal.SIGTERM,runner.stop)
    try:
        runner.execute()
    except BaseException as error:
        runner.event('STOPPED',reason=str(error))
        raise
    finally:
        try:
            if runner.child is not None and runner.child.poll() is None:
                runner.stop(signal.SIGTERM,None)
                try:
                    runner.child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    runner.event('CHILD_SHUTDOWN_PENDING',pid=runner.child.pid,
                                 note='Original runner still owns its lock; no global process cleanup attempted')
        finally:
            runner.lock.close()


if __name__=='__main__':
    main()
