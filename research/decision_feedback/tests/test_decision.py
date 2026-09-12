"""CPU tests only: synthetic models/labels, no driving experiment or GPU allocation."""
import copy
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
sys.path.insert(0,str(ROOT/'research/cfpi/tests'))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import selector_decision_common as x
import selector_decision_data as data
import selector_decision_collections as q
import train_selector_decision as trainer
import run_selector_decision as runner
import report_selector_decision as reporter
import test_continuous_deployment as fixtures
from train_selector_feedback import pair_loss
from audit_selector_cfpi_deployment import validate_sidecar
from run_selector_cfpi_deployment import Runner as BaseRunner, charge, check_budget


def metric(score=.8, safe=True):
    return dict(score=score, no_at_fault_collisions=float(safe), drivable_area_compliance=1.,
                success=float(safe), ego_progress=.5, time_to_collision_within_bound=1.,
                comfort=1., driving_direction_compliance=1.)


def row(index=0, mode='NR', fresh=True):
    r = dict(scene_id=f'log-{index}-'+format(index,'016x'), origin_log=f'log-{index}',
             mode=mode, decision_step=6, source_fingerprint=str(index),
             continuation_policy='shared_seed0', candidate_indices=[0,1],
             branch_outcomes=[metric(.2),metric(.8)], preference=(1,0,.6), fresh=fresh,
             candidate_trajectories_8=np.zeros((20,8,3),np.float32))
    r.update(row_id=x.row_id(r), candidate_bank_sha=x.candidate_bank_hash(r))
    return r


class RuleTests(unittest.TestCase):
    def test_pair_only_is_exact_legacy_loss_and_gradient(self):
        rows = [row(),row(1)]
        rows[1]['branch_outcomes']=[metric(.5),metric(.5)]
        rows[1]['preference']=None
        a = torch.randn(2,20,requires_grad=True)
        b = a.detach().clone().requires_grad_()
        expected = pair_loss(a,rows)
        actual,nll = x.feedback_loss(b,rows,False)
        self.assertTrue(torch.equal(expected,actual)); self.assertEqual(float(nll),0.)
        expected.backward(); actual.backward()
        self.assertTrue(torch.equal(a.grad,b.grad))

    def test_winner_nll_has_extreme_gradient_and_no_fake_pair(self):
        z = torch.tensor([[1000.,-1000.]+[0.]*18],requires_grad=True)
        pair,nll = x.feedback_loss(z,[row()])
        (pair+nll).backward()
        self.assertTrue(torch.isfinite(pair+nll)); self.assertLess(float(z.grad[0,1]),-1.)
        self.assertEqual(x.preferences(row()),[(1,0,.6000000000000001)])

    def test_all_pairs_normalized_by_possible_not_informative(self):
        r=row()
        r['candidate_indices']=[0,1,2]
        r['branch_outcomes']=[metric(.2),metric(.8),metric(.8)]
        z=torch.zeros(1,20,requires_grad=True)
        pair,nll=x.feedback_loss(z,[r])
        self.assertAlmostEqual(float(pair),2*.6*np.log(2)/3,places=6)
        self.assertAlmostEqual(float(nll),2*.6*np.log(20)/3,places=6)
        self.assertEqual(len(x.preferences(r)),2)

    def test_ties_and_conflicts_have_no_fake_supervision(self):
        for values in ([metric(.5),metric(.5)], [metric(.8,False),metric(.5,True)]):
            r=row();r['branch_outcomes']=values
            z=torch.randn(1,20,requires_grad=True)
            pair,nll=x.feedback_loss(z,[r])
            self.assertEqual(float(pair+nll),0.)
            (pair+nll).backward()
            self.assertEqual(float(z.grad.abs().sum()),0.)

    def test_identity_and_candidate_validation(self):
        for mutate in (
            lambda r:r['candidate_indices'].append(3),
            lambda r:r.update(candidate_indices=[0,0]),
            lambda r:r.update(candidate_indices=[0,20]),
            lambda r:r.update(continuation_policy='other'),
            lambda r:r['candidate_trajectories_8'].__setitem__((0,0,0),1),
            lambda r:r['branch_outcomes'][0].update(success=0),
        ):
            r=row();mutate(r)
            with self.assertRaises(RuntimeError):x.check_row(r)

    def test_active_random_queries_match_states_counts_and_are_deterministic(self):
        rows=[row(i,'NR' if i<32 else 'R') for i in range(64)]
        logits=np.zeros((64,20));logits[:,0]=1;logits[:17,7]=2
        active=x.query_plan(rows,logits,'P3',1,100)
        random=x.query_plan(rows,None,'P2',1,100,active)
        self.assertEqual(len(active),17)
        self.assertEqual([r['row_id'] for r in active],[r['row_id'] for r in random])
        self.assertEqual(random,x.query_plan(rows,None,'P2',1,100,active))
        self.assertTrue(all(q['candidate_index'] not in (0,1) for q in random))
        self.assertEqual(x.query_plan(rows,np.zeros((64,20)),'P3',0,100),[])
        with self.assertRaises(RuntimeError):x.query_plan(rows,None,'P2',0,100,None)
        with self.assertRaises(RuntimeError):x.query_plan(rows,None,'P2',0,100,active+active)
        with self.assertRaises(RuntimeError):x.query_plan(rows,logits,'P3',0,200)

    def test_diagnostic_threshold_edges(self):
        dominated=[dict(origin_log=str(i),mode='R' if i<2 else 'NR') for i in range(3)]
        self.assertTrue(x.diagnosis_gate(dominated,4,16)['passed'])
        self.assertFalse(x.diagnosis_gate(dominated,5,16)['passed'])
        self.assertFalse(x.diagnosis_gate(dominated[:2],0,16)['passed'])
        with self.assertRaises(RuntimeError):x.diagnosis_gate(dominated,0,15)

    def test_gate_both_modes_all_controls_and_common(self):
        effects={mode:dict(scalar=dict(metric(),score=.01,success=0.,no_at_fault_collisions=0.,drivable_area_compliance=0.),
                           gate={'score':-.02},P0={'score':.005},P1={'score':.005},P2={'score':.005},
                           common={k:-.01 for k in ('score','success','no_at_fault_collisions','drivable_area_compliance')},
                           seed_score_gains=[.1,.1,-.1]) for mode in x.MODES}
        self.assertTrue(x.point_gate(effects)['passed'])
        effects['R']['scalar']['success']=-1/58
        self.assertFalse(x.point_gate(effects)['passed'])
        self.assertEqual(len(x.evaluation_conditions()),56)
        self.assertFalse(any(c=='confirmation' for c,_,_ in x.evaluation_conditions()))

    def test_budget_small_batches_charge_full_allocation(self):
        limits=x.budget_limits(128)
        ledger=dict(gpu_hours_used=0.,phase_gpu_hours={k:0. for k in limits},phase_limits=limits)
        charge(ledger,'query',3600,4)
        self.assertEqual(ledger['gpu_hours_used'],4.)
        self.assertEqual(runner.estimate_collection([dict(scenes=32,seconds=1000)],1,'R'),
                         4*runner.forecast([dict(scenes=32,seconds=1000)],1))
        for value in (-1,0,float('nan'),float('inf')):
            with self.assertRaises(ValueError):x.budget_limits(value)
        ledger['gpu_hours_used']=128.
        with self.assertRaises(RuntimeError):check_budget(ledger,'query',128.)


class IntegrationTests(unittest.TestCase):
    def test_long_sentinel_paths_do_not_overflow_log_filename(self):
        obj=runner.Runner.__new__(runner.Runner);obj.run=Path('/tmp/synthetic');obj.python=Mock()
        obj.task('sentinel','--entry','/source/'+'a'*200,'--reference','/prior/'+'b'*200)
        self.assertLess(len(obj.python.call_args.kwargs['stage']),80)
        self.assertEqual(len(obj.python.call_args.args[1]),7)

    def test_diagnostic_report_roundtrips_and_gates_from_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); marker=root/'marker';c.atomic_json(marker,{})
            rows=[row(i,'NR' if i<8 else 'R') for i in range(16)]
            spec=dict(unsupported=[dict(row_id=rows[i]['row_id'],candidate_index=7)
                                   for i in (0,8,9)],continuation=[r['row_id'] for r in rows])
            c.atomic_json(root/'diagnose/targets.json',spec)
            c.atomic_json(root/'run_contract.json',dict(code_sha='toy'))
            entries=[];by_path={}
            for slot in ('missing',0,1):
                path=root/f'{slot}.json'
                continuation='shared_seed0' if slot=='missing' else 'T_source'
                c.atomic_json(path,dict(continuation_policy=continuation))
                entries.append(d.artifact(path))
                selected=[rows[i] for i in (0,8,9)] if slot=='missing' else rows
                by_path[str(path)]={r['row_id']:dict(
                    candidate_index=7 if slot=='missing' else slot,
                    source_fingerprint=r['source_fingerprint'],
                    outcome=metric(.1) if slot=='missing' else r['branch_outcomes'][slot])
                    for r in selected}
            c.atomic_json(root/'diagnose/collections.json',dict(prechecks=[],treatments=entries))
            with patch.object(data,'freeze_diagnostic',return_value=spec),patch.object(
                    data,'initial_rows',return_value=rows),patch.object(q,'outcomes',
                    side_effect=lambda entry,continuation:(by_path[entry['path']],d.artifact(marker))):
                first=q.diagnostic_report(root)
                second=q.diagnostic_report(root)
            self.assertEqual(first,second);self.assertTrue(first['gate']['passed'])
            self.assertEqual(q.require_pilot(root)['gate']['decision'],'PROCEED_FIXED_PILOT')
            corrupt=copy.deepcopy(first);corrupt['gate']['checks']['R_decisions']=False
            c.atomic_json(root/'diagnose/report.json',corrupt)
            with self.assertRaises(RuntimeError):q.require_pilot(root)

    def test_complete_report_all_conditions_and_repeatability(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rows=[dict(scene_id=f'log{i}-'+format(i,'016x'),origin_log=f'log{i}') for i in range(2)]
            inputs=dict(cohorts={k:dict(rows=rows) for k in ('development','common')})
            c.atomic_json(root/'decision_inputs.json',inputs);c.atomic_json(root/'run_contract.json',{})
            checked={}
            for cohort,mode,policy in x.evaluation_conditions():
                path=root/'collections'/reporter.name(cohort,mode,policy)/'deployment_collection.json'
                routing=path.parent/'routing.json'
                c.atomic_json(routing,dict(models={policy:{'policy':policy}},
                                           routes=[dict(scene_id=r['scene_id']) for r in rows]))
                condition=dict(cohort=cohort,react_type=mode,policy=policy,research_method=x.METHOD,
                    role='evaluation',continuation_policy=None,run_contract=d.artifact(root/'run_contract.json'),
                    inputs=d.artifact(root/'decision_inputs.json'),routing=d.artifact(routing))
                c.atomic_json(path,condition)
                score={'scalar_v3':.5,'gate_v3':.51,'P0':.51,'P1':.52,'P2':.525,'P3':.55}[policy.split('_seed')[0]]
                checked[str(path)]=dict(collection=condition,audit=d.artifact(path),
                    metrics={r['scene_id']:metric(score) for r in rows})
            with patch.object(data,'checked_inputs',return_value=inputs),patch.object(
                    reporter,'checked_collection',side_effect=lambda e:checked[e['path']]),patch.object(
                    data,'model_entry',side_effect=lambda run,p:{'policy':p}),patch.object(
                    reporter,'summarize_feedback',return_value={'synthetic':True}):
                first=reporter.report(root);second=reporter.report(root)
            self.assertEqual(first,second)
            self.assertEqual(first['decision'],'REQUIRES_SEPARATE_CONFIRMATION_PLAN')
            self.assertFalse(first['confirmation_authorized'])
            self.assertEqual(len(first['artifacts']),113)
            self.assertEqual(len(first['summaries']['R']),4)

    def test_rescue_counts_are_mean_of_binary_seed_events(self):
        rows=[dict(scene_id='a',origin_log='log')]
        result=reporter.seed_summary([{'a':metric(.8,True)},{'a':metric(.1,False)},
                                      {'a':metric(.8,True)}],{'a':metric(.1,False)},rows)
        self.assertAlmostEqual(result['mean_rescued_failed'],2/3)
        self.assertNotIn('rescued_failed',result)

    def test_pilot_freezes_both_query_schedules_before_any_new_return(self):
        obj=runner.Runner.__new__(runner.Runner);obj.run=Path('/tmp/synthetic-not-created')
        calls=[]
        obj.set_phase=lambda p:calls.append(('phase',p))
        obj.training=lambda *args:calls.append(('train',*args))
        obj.task=lambda *args:calls.append(('task',*args))
        obj.prechecks=lambda p:None
        obj.collection=lambda e:calls.append(('collection',e))
        obj.record=Mock()
        def read(path):
            if str(path).endswith('pilot_report.json'):return dict(decision='STOP_EFFECT_FAIL')
            return dict(prechecks=[],collections=['synthetic'])
        with patch.object(q,'require_pilot',return_value={}),patch.object(runner.d,'read',side_effect=read):
            obj.pilot()
        trains=[v for v in calls if v[0]=='train']
        self.assertEqual(len(trains),38)  # 36 arm segments + 2 shared parents
        for i,call in enumerate(calls):
            if call[:2]==('task','query_bundle') and call[3]=='P3':
                self.assertEqual(calls[i+1][0:4],('task','query_bundle','--arm','P2'))

    def test_new_transport_hybrid_and_forced_audits(self):
        fixture=fixtures.DeploymentTests();fixture.setUp()
        try:
            fixture.manifest['routes']=fixture.manifest['routes'][:1]
            fixture.manifest['routes'][0]['start_decision']=4
            visitor=fixture.router();prefix={}
            for step in (4,5,6):
                value=fixture.sidecar(visitor,step)
                path=fixture.root/f'source_{step}.pkl';c.atomic_pickle(path,value)
                prefix[str(step)]=d.artifact(path)
            target=dict(decision_step=6,prefix=prefix,source_fingerprint=f.fingerprint(value),
                        visiting_index=2,forced_index=2,source_collection={'sha256':'source'},
                        sentinel=False,no_intervention=True)
            fixture.manifest['research_method']=x.METHOD
            fixture.manifest['routes'][0].update(feedback_target=target,continuation_model_key='fold0')
            for hybrid in (True,False):
                target['no_intervention']=hybrid;target['forced_index']=2 if hybrid else 7
                router=fixture.router()
                collection=dict(fixture.collection(router),research_method=x.METHOD,cohort='train',react_type='R')
                for step in (4,6,7):
                    sidecar=fixture.sidecar(router,step)
                    sidecar['selector_rollout_contract'].update(research_method=x.METHOD,react_type='R',
                        source_data_split='train',development_consumed=False,test_consumed=False,
                        legacy_exposed_benchmark=True,independent_unseen_test=False)
                    actual=validate_sidecar(sidecar,collection,router.manifest,'code')
                    self.assertEqual(actual['selected_index'],7 if step==6 and not hybrid else 2)
                    self.assertEqual(int(np.argmax(sidecar['current_logits'])),2)
                bad=copy.deepcopy(collection);bad['cohort']='development'
                with self.assertRaises(RuntimeError):validate_sidecar(sidecar,bad,router.manifest,'code')
        finally:fixture.tearDown()

    def test_label_extension_exact_requests_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); r=row();marker=root/'marker'
            c.atomic_json(marker,{})
            parent=d.artifact(marker)
            request=dict(arm='P2',seed=0,parent=parent,requests=[dict(row_id=r['row_id'],candidate_index=7,
                candidate_bank_sha=r['candidate_bank_sha'],source_fingerprint=r['source_fingerprint'],
                continuation_policy=r['continuation_policy'])])
            c.atomic_json(root/'queries/P2_seed0/g1/requests.json',request)
            with patch.object(data,'load_rows',return_value=([r],parent)):
                with self.assertRaises(RuntimeError):data.extend_cache(root,'P2',0,1,{'foreign':metric()},[])
                with self.assertRaises(RuntimeError):data.extend_cache(root,'P2',0,1,{},[])
                result=data.extend_cache(root,'P2',0,1,{r['row_id']:metric(.9)},[])
            saved=c.load_pickle(result['cache']['path'])['rows'][0]
            self.assertEqual(saved['candidate_indices'],[0,1,7])
            self.assertEqual(r['candidate_indices'],[0,1])
            self.assertEqual(len(saved['branch_outcomes']),3)

    def test_small_collection_launches_only_needed_workers_but_four_reserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);obj=BaseRunner.__new__(BaseRunner)
            obj.args=SimpleNamespace(gpus=4,source_worldengine_root=root)
            obj.alg=root/'alg';obj.sim=root/'sim';obj.config=root/'cfg';obj.code_sha='code'
            obj.checkpoint_manifest=root/'model.json'
            c.atomic_json(obj.checkpoint_manifest,dict(checkpoint='frozen.pt'))
            obj.env=dict(SIMENGINE_PYTHON='sim',ALGENGINE_PYTHON='alg',CUDA_VISIBLE_DEVICES='2,4,6,7')
            c.atomic_json(root/'run.json',dict(gpu_count=4))
            c.atomic_json(root/'routing.json',dict(routes=[]))
            c.atomic_json(root/'scenario.json',{})
            path=root/'collections/small/deployment_collection.json'
            c.atomic_json(path,dict(collection_id='small',research_method=x.METHOD,cohort='train',
                react_type='NR',worker_count=1,scenario=d.artifact(root/'scenario.json'),
                routing=d.artifact(root/'routing.json'),run_contract=d.artifact(root/'run.json')))
            obj.python=Mock();obj.execute=Mock()
            obj.collection(d.artifact(path))
            jobs=obj.execute.call_args.args[0]
            self.assertEqual(len(jobs),2)
            self.assertEqual(jobs[0][2]['CUDA_VISIBLE_DEVICES'],'2')
            self.assertEqual(jobs[1][2]['CUDA_VISIBLE_DEVICES'],'2')
            self.assertEqual(obj.execute.call_args.kwargs['gpus'],4)

    def test_diagnostic_stop_prevents_training(self):
        obj=runner.Runner.__new__(runner.Runner)
        with tempfile.TemporaryDirectory() as folder:
            obj.run=Path(folder);obj.code_sha='code';obj.args=SimpleNamespace(stage='all')
            marker=obj.run/'marker';c.atomic_json(marker,{})
            c.atomic_json(obj.run/'audit_report.json',dict(status='PASS',code_sha='code',
                **{k:d.artifact(marker) for k in ('inputs','cpu_preflight','source_screen','forecast')}))
            c.atomic_json(obj.run/'diagnose/report.json',dict(gate=dict(passed=False,decision='STOP_DIAGNOSIS_REVIEW')))
            obj.preflight=Mock();obj.pilot=Mock();obj.record=Mock()
            obj.dispatch()
            obj.preflight.assert_not_called();obj.pilot.assert_not_called()

    def test_segment_resume_model_optimizer_rng(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root=Path(folder); marker=root/'marker.json';c.atomic_json(marker,{})
            scalar=root/'source.json';c.atomic_json(scalar,dict(artifacts=dict(scalar_manifest=d.artifact(marker))))
            inputs=dict(artifacts=dict(source_inputs=d.artifact(scalar)))
            rows=[row(i,'NR' if i<96 else 'R',False) for i in range(192)]
            rows += [row(i+192,'NR' if i<32 else 'R') for i in range(64)]
            for r in rows:
                r.update(candidate_rewards=np.arange(20,dtype=np.float32)/20,
                         candidate_reward_valid_mask=np.ones(20,dtype=bool))
            def initial(_):
                model=fixtures.ToyRefit()
                with torch.no_grad():model.logits.zero_()
                return model,{},{}
            def score(model,rows,ids,device,mode):
                # exercise Torch RNG restoration, in addition to dense NumPy RNG
                return model.logits[None].expand(len(ids),-1)+(torch.rand(len(ids),20)*.001 if model.training else 0.)
            def load(path):
                payload=torch.load(path,map_location='cpu')
                model=fixtures.ToyRefit();model.load_state_dict(payload['scene_selector_state'])
                return model.eval(),payload
            interrupted=[False]
            class Dense:
                def __init__(self,run):self.calls=0
                def sample(self,rng):
                    self.calls+=1
                    if interrupted[0] and self.calls==51:raise RuntimeError('toy interruption')
                    rng.integers(100,size=16)
                    return rows[:16]
            for target,attribute,value in (
                (trainer,'require_pilot',lambda _:None),(data,'checked_inputs',lambda _:inputs),
                (data,'load_rows',lambda *args:(copy.deepcopy(rows),d.artifact(marker))),
                (trainer,'Dense',Dense),(trainer.m,'load_incumbent',initial),
                (trainer.m,'score',score),(trainer.m,'load_selector',load)):
                stack.enter_context(patch.object(target,attribute,value))
            stack.enter_context(patch('builtins.print'))
            for run in (root/'resume',root/'control'):
                c.atomic_json(run/'run_contract.json',dict(code_sha='toy'))
                c.atomic_json(run/'decision_inputs.json',dict(toy=True))
                trainer.train(run,'shared',1,500,'cpu')
                trainer.train(run,'P1',1,100,'cpu')
            interrupted[0]=True
            with self.assertRaisesRegex(RuntimeError,'toy interruption'):
                trainer.train(root/'resume','P1',1,300,'cpu')
            state=torch.load(root/'resume/train/P1_seed1/segment_300_resume.pt')
            self.assertEqual(state['step'],150)
            interrupted[0]=False
            for run in (root/'resume',root/'control'):
                trainer.train(run,'P1',1,300,'cpu')
                trainer.train(run,'P1',1,500,'cpu')
            a=torch.load(root/'resume/train/P1_seed1/segment_500_resume.pt')
            b=torch.load(root/'control/train/P1_seed1/segment_500_resume.pt')
            self.assertTrue(torch.equal(a['model']['logits'],b['model']['logits']))
            self.assertTrue(torch.equal(a['torch_rng'],b['torch_rng']))
            self.assertEqual(a['rng'],b['rng'])
            for k in a['optimizer']['state'][0]:
                self.assertTrue(torch.equal(a['optimizer']['state'][0][k],b['optimizer']['state'][0][k]))
            parent_a=trainer.parent_entry(root/'resume',1)
            p0=trainer.train(root/'resume','P0',1,100,'cpu')
            self.assertEqual(p0['provenance']['parent'],parent_a)


if __name__ == '__main__':
    unittest.main()
