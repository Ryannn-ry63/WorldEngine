import copy
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
import cfpi_common as c
import selector_feedback_common as f
import selector_cfpi_deployment_common as d
from selector_feedback_transport import apply_intervention,validate_feedback
from run_selector_cfpi_deployment import check_budget,charge


def metric(score=.7,success=1,ep=.7):
    return dict(score=score,success=success,no_at_fault_collisions=success,
                drivable_area_compliance=1.,ego_progress=ep,time_to_collision_within_bound=1.,
                comfort=1.,driving_direction_compliance=1.)


def effect(score=0.,success=0.):
    v = {k:0. for k in d.METRICS}
    v.update(score=score,success=success)
    return v


class FeedbackTests(unittest.TestCase):
    def test_pair_tie_no_preference(self):
        self.assertIsNone(f.pareto_pair(metric(),metric()))

    def test_pair_conflict_no_preference(self):
        self.assertIsNone(f.pareto_pair(metric(.8,0),metric(.7,1)))

    def test_pair_gap_boundary(self):
        self.assertEqual(f.pareto_pair(metric(.71),metric(.7))[:2],(0,1))
        self.assertIsNone(f.pareto_pair(metric(.709),metric(.7)))

    def test_pair_reverse(self):
        self.assertEqual(f.pareto_pair(metric(.1,0),metric(.8,1))[:2],(1,0))

    def test_pair_nan_rejected(self):
        with self.assertRaises(RuntimeError):
            f.pareto_pair(metric(float('nan')),metric())

    def test_pair_fallback_original20(self):
        a=np.zeros(20);a[2]=5
        gate=np.arange(20,dtype=float)
        self.assertEqual(f.candidate_pair(a,a,gate),[2,19])

    def test_candidate_tie_break_stable(self):
        self.assertEqual(f.candidate_pair(np.zeros(20),np.zeros(20),np.zeros(20)),[0,1])

    def test_target_strictly_pre_violation(self):
        v=metric(.1,0);v['first_violation_step']=6
        self.assertEqual(f.target_step('s',v,{i:None for i in range(4,12)},'n',True),5)
        v['first_violation_step']=4
        with self.assertRaises(RuntimeError):
            f.target_step('s',v,{i:None for i in range(4,12)},'n',True)

    def test_missing_timing_is_stop(self):
        with self.assertRaises(RuntimeError):
            f.target_step('s',metric(.1,0),{i:None for i in range(4,12)},'n',True)

    def test_uniform_control_excludes_already_failed_states(self):
        v=metric(.1,0);v['first_violation_step']=6
        for i in range(20):
            self.assertLess(f.target_step('scene'+str(i),v,{k:None for k in range(4,12)},'uniform'),6)

    def test_t_fallback_counts_and_disjoint(self):
        rows=[dict(scene_id=str(i),family='rare' if i<40 else 'common') for i in range(60)]
        base={r['scene_id']:metric() for r in rows};current=copy.deepcopy(base)
        current['0']=metric(.1,0)
        chosen,counts=f.second_selection(rows,base,current,'T','R')
        self.assertEqual(len(chosen),32)
        self.assertEqual(len({r['scene_id'] for r in chosen}),32)
        self.assertEqual(counts['rare'].get('regression'),1)
        self.assertGreater(counts['rare'].get('fallback_uniform',0),0)
        self.assertEqual(f.second_selection(rows,base,current,'T','R'),(chosen,counts))

    def test_s_u_matched_uniform_membership(self):
        rows=[dict(scene_id=str(i),family='rare' if i<40 else 'common') for i in range(60)]
        values={r['scene_id']:metric() for r in rows}
        self.assertEqual(f.second_selection(rows,values,values,'S','NR'),
                         f.second_selection(rows,values,values,'U','NR'))

    def test_co_primary_gate(self):
        effects={mode:dict(scalar=effect(.02),gate=effect(-.01),S=effect(.006),U=effect(.006)) for mode in f.MODES}
        self.assertTrue(f.shortlist_gate(effects)['passed'])
        effects['NR']['U']['score']=.004
        self.assertFalse(f.shortlist_gate(effects)['passed'])

    def test_final_all_seeds_and_safety(self):
        effects={mode:dict(scalar=[effect(.03)]*3,gate=[effect(0.,.02)]*3,common=[effect()]*3) for mode in f.MODES}
        self.assertTrue(f.final_gate(effects)['passed'])
        effects['R']['scalar']=[effect(.03,-.01)]*3
        self.assertFalse(f.final_gate(effects)['passed'])

    def test_final_missing_seed_rejected(self):
        effects={mode:dict(scalar=[effect(.03)]*2,gate=[effect(0.,.02)]*3,common=[effect()]*3) for mode in f.MODES}
        with self.assertRaises(RuntimeError):
            f.final_gate(effects)

    def test_budget_shared_buffer_once(self):
        ledger=dict(gpu_hours_used=0.,phase_gpu_hours={k:0. for k in f.LIMITS},phase_limits=f.LIMITS)
        charge(ledger,'feedback',28*3600/8,8)
        charge(ledger,'train',4*3600/8,8)
        check_budget(ledger,'eval',64.,30.)
        with self.assertRaises(RuntimeError):
            check_budget(ledger,'eval',64.,32.)

    def test_budget_explicit_scaling_preserves_default(self):
        self.assertEqual(f.budget_limits(64.),f.LIMITS)
        self.assertEqual(f.budget_limits(128.),dict(feedback=48.,train=8.,eval=60.,buffer=12.))
        self.assertEqual(sum(f.budget_limits(128.).values()),128.)

    def test_budget_invalid_allowance_rejected(self):
        for value in (0.,-1.,float('nan'),float('inf'),-float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                f.budget_limits(value)

    def test_expanded_budget_still_enforces_shared_buffer_and_total(self):
        limits=f.budget_limits(128.)
        ledger=dict(gpu_hours_used=0.,phase_gpu_hours={k:0. for k in limits},phase_limits=limits)
        check_budget(ledger,'feedback',128.,39.)
        charge(ledger,'feedback',56*3600/8,8)
        charge(ledger,'train',8*3600/8,8)
        check_budget(ledger,'eval',128.,60.)
        with self.assertRaises(RuntimeError):
            check_budget(ledger,'eval',128.,64.)
        with self.assertRaises(RuntimeError):
            check_budget(ledger,'feedback',128.,4.)
        ledger['gpu_hours_used']=127.
        ledger['phase_gpu_hours']['eval']=63.
        with self.assertRaises(RuntimeError):
            check_budget(ledger,'eval',128.,1.)

    def test_continuation_switch_only_after_target(self):
        manifest=dict(routes=[dict(scene_id='scene',origin_token='token',start_decision=4,
            model_key='visitor',continuation_model_key='continue',feedback_target=dict(decision_step=7))])
        self.assertEqual(d.route_for(manifest,'token',7)[0]['model_key'],'visitor')
        self.assertEqual(d.route_for(manifest,'token',8)[0]['model_key'],'continue')

    def test_forced_action_keeps_real_logits(self):
        import torch
        from audit_selector_cfpi_deployment import SHAPES
        with tempfile.TemporaryDirectory() as folder:
            context={k:np.zeros(shape,dtype=np.float32) for k,shape in SHAPES.items()}
            context['current_logits'][3]=5
            context['selected_index']=3
            context['candidate_trajectories_8'][7,:,0]=7
            path=Path(folder)/'context.pkl';c.atomic_pickle(path,context)
            target=dict(decision_step=4,prefix={'4':d.artifact(path)},source_fingerprint=f.fingerprint(context),
                        visiting_index=3,forced_index=7,source_collection=dict(sha256='source'),sentinel=False)
            result=dict(diffusiondrive_rollout_context=copy.deepcopy(context),trajectory=np.zeros((40,3)),chosen_ind=3)
            selected,meta=apply_intervention(result,dict(feedback_target=target),4,3,
                                            lambda x:torch.repeat_interleave(x,5,dim=1),'cpu')
            self.assertEqual(selected,7)
            self.assertEqual(int(result['diffusiondrive_rollout_context']['current_logits'].argmax()),3)
            self.assertEqual(meta['natural_selected_index'],3)
            self.assertTrue(np.all(result['trajectory'][:,0]==7))
            target['visiting_index']=2
            with self.assertRaises(RuntimeError):
                apply_intervention(result,dict(feedback_target=target),4,3,lambda x:x,'cpu')

    def test_wrong_mode_exposure_rejected(self):
        sidecar=dict(selector_rollout_contract=dict(research_method=f.METHOD,react_type='R',source_data_split='train',
                     development_consumed=False,test_consumed=False,legacy_exposed_benchmark=True,independent_unseen_test=False),
                     planner_step=4,selected_index=0,cfpi_deployment={})
        with self.assertRaises(RuntimeError):
            validate_feedback(sidecar,dict(react_type='NR',cohort='train'),{})

    def test_low_probability_pair_gradient(self):
        import torch
        from train_selector_feedback import pair_loss
        logits=torch.tensor([[-1000.,1000.]+[0.]*18],requires_grad=True)
        loss=pair_loss(logits,[dict(candidate_indices=[0,1],preference=(0,1,.5))])
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[0,0]),-.5)

    def test_no_pair_zero_gradient(self):
        import torch
        from train_selector_feedback import pair_loss
        logits=torch.zeros((1,20),requires_grad=True)
        pair_loss(logits,[dict(preference=None)]).backward()
        self.assertEqual(float(logits.grad.abs().sum()),0.)

    def test_fixed_evaluation_inventory(self):
        from report_selector_feedback import expected
        self.assertEqual(len(expected('screen')),10)
        self.assertEqual(len(expected('final')),18)
        with self.assertRaises(RuntimeError):
            expected('confirmation')

    def test_direct_extra_seed_requires_screen_gate(self):
        from unittest.mock import patch
        from train_selector_feedback import train
        with patch('report_selector_feedback.report',return_value=dict(decision='STOP_NO_MECHANISM_SIGNAL')):
            with self.assertRaisesRegex(RuntimeError,'Additional seeds'):
                train(Path('/unused'),'T',1,'cpu')


if __name__=='__main__':
    unittest.main()
