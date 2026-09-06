"""Synthetic orchestration only: never launches the real experiment."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
spec = importlib.util.spec_from_file_location('rare_overnight',ROOT/'selector_rare_overnight.py')
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)
import selector_rare_common as r


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def artifact(path):
    return dict(path=str(path.resolve()),sha256=auto.sha(path))


class OvernightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'run1'
        self.contract = dict(method=r.METHOD,policies=list(r.POLICIES),seeds=[0,1,2],
                             phase_limits=r.LIMITS,gpu_count=8,gpu_hour_limit=64.,
                             prior_deployment='/frozen/old',source_worldengine_root='/frozen/source')
        write(self.run/'run_contract.json',self.contract)
        write(self.run/'rare_inputs.json',{'status':'PASS'})
        write(self.run/'bridge_report.json',dict(status='PASS'))
        write(self.run/'decision_ledger.json',dict(active=None,gpu_hours_used=20.,phase_limits=r.LIMITS,
              phase_gpu_hours=dict(bridge=4.,screen=16.,confirm=0.,buffer=0.)))
        self.runs_patch = patch.object(auto,'RUNS',self.root)
        self.runs_patch.start()
        self.addCleanup(self.runs_patch.stop)
        self.runner = auto.Automation('run1')
        self.addCleanup(self.runner.lock.close)
        self.runner.event = Mock()

    def screen(self,passed=True):
        write(self.run/'screen_report.json',dict(status='PASS',engineering_pass=True,phase='screen',
              decision='PROCEED_FIXED_WINNER' if passed else 'STOP_NO_WINNER',
              winner='q_anchor' if passed else None,methods={'q_anchor':dict(effect_gate=dict(passed=passed))}))
        if not passed:
            return
        models = {}
        for seed in range(3):
            path = self.run/f'model{seed}.pt'
            write(path,{'synthetic':seed})
            models[f'q_anchor_seed{seed}'] = artifact(path)
        write(self.run/'winner.json',dict(method='q_anchor',models=models,refit_authorized=False,
              screen_report=artifact(self.run/'screen_report.json')))

    def stages(self,passed=True,fail=False):
        def stage(name):
            if fail:
                raise RuntimeError('synthetic screen failure')
            if name=='screen':
                self.screen(passed)
            else:
                write(self.run/'confirm_report.json',dict(status='PASS',decision='INSUFFICIENT_EVIDENCE'))
        return Mock(side_effect=stage)

    def test_pass_automatically_runs_confirm_and_preserves_contract_ledger(self):
        before = [(self.run/name).read_bytes() for name in ('run_contract.json','decision_ledger.json')]
        self.runner.stage = self.stages()
        self.runner.execute()
        self.assertEqual([x.args[0] for x in self.runner.stage.call_args_list],['screen','confirm'])
        self.assertEqual(before,[(self.run/name).read_bytes() for name in ('run_contract.json','decision_ledger.json')])

    def test_no_winner_stops_without_confirm(self):
        self.runner.stage = self.stages(passed=False)
        self.runner.execute()
        self.assertEqual([x.args[0] for x in self.runner.stage.call_args_list],['screen'])
        self.runner.event.assert_any_call('STOP_NO_WINNER',confirm_started=False)

    def test_screen_failure_never_advances(self):
        self.runner.stage = self.stages(fail=True)
        with self.assertRaisesRegex(RuntimeError,'synthetic screen failure'):
            self.runner.execute()
        self.assertEqual(self.runner.stage.call_count,1)

    def test_incomplete_screen_cannot_advance(self):
        self.screen()
        screen = auto.read(self.run/'screen_report.json')
        screen['engineering_pass'] = False
        write(self.run/'screen_report.json',screen)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            auto.gate(self.run,self.contract)

    def test_model_hash_drift_fails_closed(self):
        self.screen()
        write(self.run/'model0.pt',{'different':True})
        with self.assertRaisesRegex(RuntimeError,'changed'):
            auto.gate(self.run,self.contract)

    def test_missing_seed_fails_closed(self):
        self.screen()
        winner = auto.read(self.run/'winner.json')
        winner['models'].pop('q_anchor_seed2')
        write(self.run/'winner.json',winner)
        with self.assertRaisesRegex(RuntimeError,'all fixed seeds'):
            auto.gate(self.run,self.contract)

    def test_frozen_screen_hash_is_checked(self):
        self.screen()
        screen = auto.read(self.run/'screen_report.json')
        screen['unexpected_change'] = True
        write(self.run/'screen_report.json',screen)
        with self.assertRaisesRegex(RuntimeError,'changed'):
            auto.gate(self.run,self.contract)

    def test_refit_permission_is_not_inferred(self):
        self.screen()
        winner = auto.read(self.run/'winner.json')
        winner['refit_authorized'] = True
        write(self.run/'winner.json',winner)
        with self.assertRaisesRegex(RuntimeError,'Invalid winner'):
            auto.gate(self.run,self.contract)

    def test_budget_stop_no_confirm(self):
        self.screen()
        ledger = auto.read(self.run/'decision_ledger.json')
        ledger.update(gpu_hours_used=64.,phase_gpu_hours=dict(bridge=6.,screen=34.,confirm=24.,buffer=0.))
        write(self.run/'decision_ledger.json',ledger)
        self.runner.stage = self.stages()
        self.runner.execute()
        self.runner.stage.assert_not_called()
        self.assertIn('STOP_BUDGET',[x.args[0] for x in self.runner.event.call_args_list])

    def test_resume_skips_finished_screen(self):
        self.screen()
        self.runner.stage = self.stages()
        self.runner.execute()
        self.runner.stage.assert_called_once_with('confirm')

    def test_completed_confirmation_revalidated_without_new_stages(self):
        self.screen()
        write(self.run/'confirm_report.json',dict(status='PASS',decision='FAIL'))
        self.runner.stage = self.stages()
        with patch('report_selector_rare.report',return_value=dict(status='PASS',decision='FAIL')) as check:
            self.runner.execute()
        check.assert_called_once_with(self.run,'confirm')
        self.runner.stage.assert_not_called()

    def test_active_ledger_blocks_automatic_confirm(self):
        self.screen()
        ledger = auto.read(self.run/'decision_ledger.json')
        ledger['active'] = dict(stage='other')
        write(self.run/'decision_ledger.json',ledger)
        self.runner.stage = self.stages()
        with self.assertRaisesRegex(RuntimeError,'still active'):
            self.runner.execute()
        self.runner.stage.assert_not_called()

    def test_pending_bridge_does_not_start_screen(self):
        write(self.run/'bridge_report.json',dict(status='INCOMPLETE'))
        self.runner.stage = self.stages()
        with self.assertRaisesRegex(RuntimeError,'Bridge not complete'):
            self.runner.execute()
        self.runner.stage.assert_not_called()

    def test_duplicate_automation_rejected(self):
        with self.assertRaises(BlockingIOError):
            auto.Automation('run1')

    def test_forward_signal_only_to_owned_child_and_no_next_phase(self):
        child = Mock()
        child.poll.return_value = None
        self.runner.child = child
        self.runner.stop(signal.SIGTERM,None)
        child.send_signal.assert_called_once_with(signal.SIGTERM)
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            self.runner.stage('confirm')

    def test_original_command_uses_same_run_and_budget(self):
        child = Mock()
        child.wait.return_value = 0
        with patch.object(auto.subprocess,'Popen',return_value=child) as launch:
            self.runner.stage('screen')
        args = launch.call_args.args[0]
        self.assertEqual(args[2:4],['screen','run1'])
        self.assertEqual(args[args.index('--gpu-hours')+1],'64.0')
        self.assertEqual(args[args.index('--deployment-run')+1],'/frozen/old')

    def test_child_failure_cannot_be_treated_as_completion(self):
        child = Mock()
        child.wait.return_value = 1
        with patch.object(auto.subprocess,'Popen',return_value=child),self.assertRaisesRegex(RuntimeError,'exit=1'):
            self.runner.stage('screen')


if __name__=='__main__':
    unittest.main()
