"""CPU-only invariants; no experiments or model selection on real outcomes."""
import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_rare_common as r
import selector_rare_model as rare
from selector_cfpi_deployment_router import load_bank, selector_scores
from run_selector_cfpi_deployment import check_budget
from run_selector_rare import forecast
from report_selector_rare import average, summarize, report
import report_selector_rare as reporting
import train_selector_rare as training


class ToyTrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.linspace(-.1,.1,20))


def metric(score=.5,success=1.,ep=.5):
    v = {k:1. for k in d.METRICS}
    v.update(score=score,success=success,ego_progress=ep,no_at_fault_collisions=success)
    return v


def visible():
    rng = np.random.default_rng(3)
    row = {key:rng.normal(size=shape).astype(np.float32) for key,shape in m.INPUTS.values()}
    row['reference_logits'] = rng.normal(size=20).astype(np.float32)
    return row


class Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.base = m.v3.model_from_config({}).eval()
        self.row = visible()

    def test_zero_exact_and_online(self):
        anchor = rare.AnchoredSelector(self.base).eval()
        a = m.score(self.base,[self.row],[0],'cpu','residual').detach()
        b = m.score(anchor,[self.row],[0],'cpu','residual').detach()
        self.assertTrue(torch.equal(a,b))
        online = selector_scores(anchor,self.row,'cpu','residual')
        np.testing.assert_array_equal(b.numpy()[0],online)

    def test_update_freezes_all_incumbent_tensors(self):
        anchor = rare.AnchoredSelector(self.base).train()
        frozen = {k:v.clone() for k,v in anchor.incumbent.state_dict().items()}
        self.assertFalse(anchor.incumbent.training)
        self.assertTrue(all(not p.requires_grad for p in anchor.incumbent.parameters()))
        self.assertEqual(sum(p.numel() for p in anchor.parameters() if p.requires_grad),33025)
        optimizer = torch.optim.AdamW([p for p in anchor.parameters() if p.requires_grad],lr=1e-3)
        m.score(anchor,[self.row],[0],'cpu','residual').square().mean().backward()
        optimizer.step()
        self.assertTrue(all(torch.equal(v,anchor.incumbent.state_dict()[k]) for k,v in frozen.items()))
        self.assertGreater(float(anchor.adapter[-1].weight.abs().sum()),0)

    def test_full_updates_old_selector_not_extra_v3(self):
        full = rare.initialize(self.base,'q_full')
        self.assertEqual(set(full.state_dict()),set(self.base.state_dict()))
        self.assertTrue(all(p.requires_grad for p in full.parameters()))
        self.assertTrue(torch.equal(m.score(full,[self.row],[0],'cpu','residual'),
                                    m.score(self.base,[self.row],[0],'cpu','residual')))

    def test_allowlist_ignores_privileged_labels(self):
        a = m.visible_batch([self.row],[0],'cpu')
        self.row.update(branch_returns=object(),future_collision=object(),origin_log=object(),fold=object())
        b = m.visible_batch([self.row],[0],'cpu')
        self.assertEqual(set(a),set(m.INPUTS))
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))

    def test_missing_visible_fails(self):
        self.row.pop('candidate_features')
        with self.assertRaises(KeyError):
            m.visible_batch([self.row],[0],'cpu')

    def test_teacher_kl_direction_and_no_teacher_grad(self):
        teacher = torch.tensor([[2.,0.,-1.]],requires_grad=True)
        student = torch.tensor([[0.,1.,3.]],requires_grad=True)
        loss = rare.teacher_kl(student,teacher)
        expected = (teacher.softmax(-1)*(teacher.log_softmax(-1)-student.log_softmax(-1))).sum()
        self.assertTrue(torch.allclose(loss,expected))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertAlmostEqual(float(rare.teacher_kl(teacher,teacher)),0.)

    def test_export_load_bank_parity_and_final_only(self):
        for method in r.POLICIES:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model.pt'
                model = rare.initialize(self.base,method).eval()
                provenance = dict(method=method,step=500)
                rare.save(path,model,{},provenance)
                bank = load_bank(dict(d.artifact(path),kind='rare',score_mode='residual',provenance=provenance))
                np.testing.assert_array_equal(selector_scores(model,self.row,'cpu','residual'),
                                              selector_scores(bank,self.row,'cpu','residual'))
                rare.save(path,model,{},dict(method=method,step=25))
                with self.assertRaises(RuntimeError):
                    rare.load(path)


class Protocol(unittest.TestCase):
    def test_replay_deterministic_log_cap_and_exclusions(self):
        data = {f'{log}:{i}':metric() for log in range(40) for i in range(4)}
        origin = lambda s:s.split(':')[0]
        a,counts = r.choose_replay(data,{'0','1'},origin)
        b,_ = r.choose_replay(dict(reversed(list(data.items()))),{'0','1'},origin)
        self.assertEqual(a,b)
        self.assertEqual(len(a),64)
        self.assertFalse({origin(s) for s in a}&{'0','1'})
        self.assertTrue(all(sum(origin(s)==log for s in a)<=2 for log in {origin(s) for s in a}))
        self.assertEqual(counts['eligible_scenes'],152)

    def test_replay_ineligible_and_no_quota_relaxation(self):
        data = {'a:1':metric(success=0),'b:1':metric(ep=.19)}
        with self.assertRaises(RuntimeError):
            r.choose_replay(data,set(),lambda s:s.split(':')[0],1)

    def test_aggregate_row_is_not_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'m.csv'
            with path.open('w',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=['token',*d.METRICS])
                writer.writeheader()
                writer.writerow(dict(token='s',**metric(.3)))
                writer.writerow(dict(token='overall_average',**metric(.9)))
            self.assertEqual(set(d.metrics_csv(path)),{'s'})
            self.assertEqual(d.metrics_csv(path)['s']['score'],.3)

    def test_scene_weighted_cluster_not_equal_log(self):
        rows = [dict(scene_id=str(i),origin_log='A' if i<9 else 'B') for i in range(10)]
        out = r.cluster_interval({str(i):0. if i<9 else 1. for i in range(10)},rows)
        self.assertAlmostEqual(out['scene_mean'],.1)
        self.assertAlmostEqual(out['equal_log_mean'],.5)
        self.assertEqual(out['logs'],2)

    def test_bootstrap_rejects_missing_and_duplicates(self):
        with self.assertRaises(RuntimeError):
            r.cluster_interval({'s':.1},[dict(scene_id='s',origin_log='a')]*2)

    def test_seed_average_first(self):
        result = average([{'s':metric(x)} for x in (.1,.2,.3)])
        self.assertAlmostEqual(result['s']['score'],.2)
        with self.assertRaises(RuntimeError):
            average([{'s':metric()}]*2)

    def test_gate_all_controls_required(self):
        gain = {k:0. for k in d.METRICS}
        gain['score'] = .031
        retention = {k:0. for k in d.METRICS}
        self.assertTrue(r.effect_gate([gain]*3,[gain]*3,[retention]*3)['passed'])
        for key in ('score','success','no_at_fault_collisions','drivable_area_compliance'):
            bad = dict(retention,**{key:-.011})
            self.assertFalse(r.effect_gate([gain]*3,[gain]*3,[bad]*3)['passed'])
        weak = dict(gain,score=.02)
        self.assertFalse(r.effect_gate([gain]*3,[weak]*3,[retention]*3)['passed'])
        with self.assertRaises(RuntimeError):
            r.effect_gate([gain]*2,[gain]*3,[retention]*3)

    def test_rare_safety_and_progress_gate(self):
        gain = {k:0. for k in d.METRICS}
        gain['score'] = .04
        for k in ('success','ego_progress'):
            bad = dict(gain,**{k:-.001})
            self.assertFalse(r.effect_gate([bad]*3,[gain]*3,[gain]*3)['passed'])

    def test_budget_shared_buffer_not_per_phase_reset(self):
        ledger = dict(gpu_hours_used=10.,phase_limits=r.LIMITS,
                      phase_gpu_hours=dict(bridge=8.,screen=2.,confirm=0.,buffer=0.))
        check_budget(ledger,'screen',64.,31.)
        with self.assertRaises(RuntimeError):
            check_budget(ledger,'screen',64.,32.)
        ledger['gpu_hours_used'] = 63.
        with self.assertRaises(RuntimeError):
            check_budget(ledger,'confirm',64.,1.)

    def test_forecast_positive_monotonic(self):
        samples = [dict(scenes=64,seconds=900),dict(scenes=128,seconds=1300)]
        self.assertGreater(forecast(samples,230),forecast(samples,58))
        self.assertGreater(forecast(samples,58),0)

    def test_locked_contract_cannot_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'f.json'
            c.locked_json(path,{'seed':0})
            c.locked_json(path,{'seed':0})
            with self.assertRaises(RuntimeError):
                c.locked_json(path,{'seed':1})

    def test_missing_results_do_not_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(report(Path(folder),'screen')['status'],'INCOMPLETE')

    def test_exact_condition_counts_and_no_unfrozen_confirm(self):
        self.assertEqual(len(r.expected_conditions('bridge')),3)
        self.assertEqual(len(r.expected_conditions('screen')),26)
        self.assertEqual(len(r.expected_conditions('confirm','q_anchor')),5)
        with self.assertRaises(RuntimeError):
            r.expected_conditions('confirm')

    def test_real_report_cli_missing_safe(self):
        import subprocess
        out = subprocess.run([sys.executable,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive/report_selector_rare.py'),
                              '--run-root','/tmp/nonexistent-rare-test','--phase','screen'],capture_output=True,text=True)
        self.assertEqual(out.returncode,0,out.stderr)
        self.assertIn('INCOMPLETE',out.stdout)

    def test_synthetic_500_step_resume_exact(self):
        # Synthetic 20 parameters only, not an experiment on a real planner.
        rng = np.random.default_rng(4)
        rows = [dict(branch_returns=rng.random(20).astype(np.float32)) for _ in range(64)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outputs = []
            for interrupted in (False,True):
                run = root/str(interrupted)
                c.atomic_pickle(run/'correction.pkl',rows)
                c.atomic_pickle(run/'replay.pkl',dict(rows=[{}]*512))
                c.atomic_json(run/'scalar.json',{})
                c.atomic_json(run/'rare_inputs.json',dict(status='PASS',artifacts={
                    'cache':d.artifact(run/'correction.pkl'),'replay':d.artifact(run/'replay.pkl'),
                    'scalar_manifest':d.artifact(run/'scalar.json')}))
                c.atomic_json(run/'run_contract.json',dict(code_sha='synthetic'))
                original_replace = Path.replace
                def interrupt_after_atomic_resume(path,target):
                    result = original_replace(path,target)
                    if Path(target).name=='resume.pt' and torch.load(target,map_location='cpu')['step']==25:
                        raise RuntimeError('injected after atomic checkpoint')
                    return result
                def reload(path):
                    payload = torch.load(path,map_location='cpu')
                    model = ToyTrain()
                    model.load_state_dict(payload['scene_selector_state'])
                    return model,payload
                def score(model,rows,ids,device,mode):
                    return model.logits[None].expand(len(ids),-1)
                with patch.object(training.c,'validate_cache',side_effect=lambda x:x), \
                     patch.object(training,'validate_incumbent_recompute',return_value={'synthetic':True}), \
                     patch.object(training.m,'load_incumbent',side_effect=lambda _: (ToyTrain(),{},{})), \
                     patch.object(training.m,'score',side_effect=score), \
                     patch.object(training.rare,'load',side_effect=reload):
                    if interrupted:
                        with patch.object(Path,'replace',new=interrupt_after_atomic_resume),self.assertRaisesRegex(RuntimeError,'injected'):
                            training.train(run,'q_full',0,'cpu')
                    training.train(run,'q_full',0,'cpu')
                    training.train(run,'q_full',0,'cpu')
                saved = d.read(run/'train/q_full_seed0/report.json')
                outputs.append(torch.load(saved['selector']['path'],map_location='cpu')['scene_selector_state']['logits'])
            self.assertTrue(torch.equal(outputs[0],outputs[1]))

    def test_full_screen_freezes_one_winner_then_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            rows = [dict(scene_id=f's{i}',origin_log=f'log{i}') for i in range(3)]
            base = {x['scene_id']:metric(.5) for x in rows}
            improved = {x['scene_id']:metric(.55) for x in rows}
            inputs = dict(cohorts={k:dict(rows=rows) for k in ('development','common','confirmation')},common_reuse={})
            checked = {}
            for policy in r.REFERENCES:
                path = run/'old'/policy/'collection.json'
                c.atomic_json(path,dict(cohort='common',policy=policy))
                entry = d.artifact(path)
                inputs['common_reuse'][policy] = dict(collection=entry,audit=entry)
                checked[str(path)] = dict(collection=d.read(path),audit=entry,metrics=base)
            c.atomic_json(run/'rare_inputs.json',inputs)
            for method in r.POLICIES:
                for seed in r.SEEDS:
                    path = run/'train'/f'{method}_seed{seed}'
                    c.atomic_json(path/'step_500.pt',{'synthetic':True})
                    c.atomic_json(path/'report.json',dict(trainable_parameters=33025 if method!='q_full' else 999999,
                                                         selector=d.artifact(path/'step_500.pt')))
            def prepare(phase,winner=None,drop=False):
                entries = []
                for cohort,policy in r.expected_conditions(phase,winner):
                    path = run/'collections'/f'{phase}_{cohort}_{policy}.json'
                    c.atomic_json(path,dict(cohort=cohort,policy=policy))
                    entry = d.artifact(path)
                    entries.append(entry)
                    gain = policy.startswith(r.POLICIES) and cohort!='common'
                    checked[str(path)] = dict(collection=d.read(path),audit=entry,metrics=improved if gain else base)
                c.atomic_json(run/f'{phase}_collections.json',dict(collections=entries[:-1] if drop else entries))
            prepare('screen',drop=True)
            with patch.object(reporting,'checked_collection',side_effect=lambda entry:checked[entry['path']]):
                with self.assertRaisesRegex(RuntimeError,'inventory'):
                    report(run,'screen')
                prepare('screen')
                result = report(run,'screen')
                self.assertEqual(result['decision'],'PROCEED_FIXED_WINNER')
                winner = d.read(run/'winner.json')
                self.assertEqual(winner['method'],'local_anchor')
                self.assertEqual(len(winner['models']),3)
                self.assertFalse(winner['refit_authorized'])
                self.assertEqual(report(run,'screen'),result)
                prepare('confirm','local_anchor')
                result = report(run,'confirm')
                self.assertEqual(result['decision'],'CONFIRMED_LEGACY_EXPOSED')
                self.assertFalse(result['winner_replacement_authorized'])

    def test_bridge_checks_initial_context_not_diverged_future(self):
        sys.path.insert(0,str(ROOT/'research/cfpi/tests'))
        import test_continuous_deployment as legacy
        fixture = legacy.DeploymentTests()
        fixture.setUp()
        try:
            fixture.manifest['routes'][0]['start_decision'] = 4
            router = fixture.router()
            collection = fixture.collection(router)
            source = fixture.root/'initial.pkl'
            c.atomic_pickle(source,legacy.context())
            collection['initial_contexts'] = {fixture.scene:d.artifact(source)}
            sidecar = fixture.sidecar(router,4)
            legacy.validate_sidecar(sidecar,collection,router.manifest,'code')
            sidecar['candidate_features'][0,0] = 1.
            with self.assertRaisesRegex(RuntimeError,'initial context'):
                legacy.validate_sidecar(sidecar,collection,router.manifest,'code')
            later = fixture.sidecar(router,7)
            later['candidate_features'][0,0] = 1.
            legacy.validate_sidecar(later,collection,router.manifest,'code')
        finally:
            fixture.tearDown()

    def test_rare_exposure_metadata_is_required(self):
        sys.path.insert(0,str(ROOT/'research/cfpi/tests'))
        import test_continuous_deployment as legacy
        fixture = legacy.DeploymentTests()
        fixture.setUp()
        try:
            router = fixture.router()
            collection = dict(fixture.collection(router),research_method=r.METHOD,cohort='confirmation')
            sidecar = fixture.sidecar(router)
            with self.assertRaisesRegex(RuntimeError,'exposure'):
                legacy.validate_sidecar(sidecar,collection,router.manifest,'code')
            sidecar['selector_rollout_contract'].update(research_method=r.METHOD,source_data_split='confirmation',
                development_consumed=False,test_consumed=True,legacy_exposed_benchmark=True,independent_unseen_test=False)
            legacy.validate_sidecar(sidecar,collection,router.manifest,'code')
        finally:
            fixture.tearDown()


if __name__=='__main__':
    unittest.main()
