import csv
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_snapshot_common as s
import selector_snapshot_data as data
import selector_snapshot_audit as audit
import run_selector_snapshot as runner
import report_selector_snapshot as reporting
import selector_snapshot_tasks as tasks


def values(score=.8,success=1.):
    return dict(comfort=1.,drivable_area_compliance=1.,driving_direction_compliance=1.,
                ego_progress=.6,no_at_fault_collisions=success,score=score,success=success,
                time_to_collision_within_bound=1.)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def write_csv(self,rows):
        p=self.root/'metrics.csv'
        with p.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['token','valid',*d.METRICS])
            writer.writeheader();writer.writerows(rows)
        return p
    def test_real_denominator(self):
        rows=[dict(token=f's{i}',valid='True',**values(success=float(i<266))) for i in range(288)]
        rows.append(dict(token='overall_average',valid='True',**values(success=.9)))
        m=s.metrics(self.write_csv(rows))
        self.assertEqual(len(m),288);self.assertAlmostEqual(s.mean(m)['success'],266/288)
    def test_official_aggregate_excluded(self):
        p=self.write_csv([dict(token='a',valid='True',**values()),dict(token='average',valid='True',**values())])
        self.assertEqual(set(s.metrics(p,{'a'},valid=True)),{'a'})
    def test_open_audit_reuse_rejects_condition_and_status_drift(self):
        pdm=self.write_csv([dict(token='a',valid='True',**values())])
        c.atomic_json(self.root/'run_contract.json',{})
        inp=dict(models={'P1_seed2':{'sha256':'model'}},open_tokens={'open_rare':['a']})
        good=dict(status='PASS',policy='P1_seed2',block='open_rare',count=1,
                  checkpoint=inp['models']['P1_seed2'],noise=s.FORMAL_NOISE,
                  contract=d.artifact(self.root/'run_contract.json'),artifacts=[d.artifact(pdm)],
                  pdm=d.artifact(pdm),metrics=s.metrics(pdm))
        out=self.root/'openloop/P1_seed2/open_rare/audit.json'
        with patch.object(s,'checked_inputs',return_value=inp):
            c.atomic_json(out,good)
            self.assertEqual(tasks.checked_open(self.root,'P1_seed2','open_rare'),good)
            for key,value in [('status','FAIL'),('policy','scalar_v3'),('block','open_navtest'),('count',2)]:
                c.atomic_json(out,dict(good,**{key:value}))
                with self.assertRaises(RuntimeError):tasks.checked_open(self.root,'P1_seed2','open_rare')
    def test_missing_duplicate_invalid_nonfinite(self):
        good=dict(token='a',valid='True',**values())
        for rows,expected in (([good],{'a','b'}),([good,good],None),
                              ([dict(good,valid='False')],None),([dict(good,score=float('nan'))],None)):
            with self.assertRaises(RuntimeError):s.metrics(self.write_csv(rows),expected,valid=True)
    def test_archive_collision(self):
        p=self.root/'source';p.write_text('first')
        out=self.root/'archive';s.preserved_copy(p,out);s.preserved_copy(p,out)
        p.write_text('changed')
        with self.assertRaises(RuntimeError):s.preserved_copy(p,out)
        self.assertEqual(out.read_text(),'first')
    def export_inputs(self):
        frozen={f'base{i}':torch.tensor([float(i)]) for i in range(963)}
        frozen.update({s.PREFIX+f'w{i}':torch.zeros(1) for i in range(54)})
        scalar=self.root/'scalar.pt';torch.save(dict(state_dict=frozen,meta={}),scalar)
        payload=dict(score_mode='residual',inference_uses_reward_or_q=False,
            provenance=dict(policy='P1_seed2'),scene_selector_config={},
            scene_selector_state={f'w{i}':torch.ones(1) for i in range(54)})
        selector=self.root/'selector.pt';torch.save(payload,selector)
        return scalar,selector,payload
    def test_export_and_repeat(self):
        scalar,selector,_=self.export_inputs();out=self.root/'native.pt'
        a=data.export_model(scalar,selector,out,{'source_selector_sha':'s'})
        b=data.export_model(scalar,selector,out,{'source_selector_sha':'s'})
        self.assertEqual(a,b);self.assertEqual(a['unchanged_nonselector_tensors'],963)
        w=torch.load(out)['state_dict'];self.assertTrue(torch.equal(w['base9'],torch.tensor([9.])))
        self.assertTrue(torch.equal(w[s.PREFIX+'w0'],torch.ones(1)))
    def test_export_wrong_policy_shape_or_frozen_drift(self):
        scalar,selector,payload=self.export_inputs()
        payload['provenance']['policy']='P3_seed2';torch.save(payload,selector)
        with self.assertRaises(RuntimeError):data.export_model(scalar,selector,self.root/'n.pt',{})
    def native_fixture(self):
        from audit_selector_cfpi_deployment import SHAPES
        co=dict(checkpoint={'sha256':'model'},noise='noise',id='condition',mode='R',prefixes={'token':'scene'})
        sc={k:np.zeros(shape) for k,shape in SHAPES.items()}
        sc.update(checkpoint_sha256='model',candidate_noise_namespace='noise',code_sha='code',
            deployed_trajectory=np.zeros((40,3)),
            scene_prefix='token',planner_step=4,selected_index=0,selected_indices=np.array(0),
            selector_rollout_contract=dict(experiment=s.METHOD,condition='condition',react_type='R',
                                          inference_uses_reward_or_q=False))
        return sc,co
    def test_native_validation(self):
        sc,co=self.native_fixture();self.assertEqual(s.validate_native(sc,co,'code')['scene_id'],'scene')
        sc['planner_step']=12
        with self.assertRaises(RuntimeError):s.validate_native(sc,co,'code')
        s.validate_native(sc,co,'code',terminal=True)
    def test_native_collection_actual_actions_and_coverage(self):
        root=self.root/'collection';root.mkdir()
        c.atomic_json(self.root/'contract.json',dict(code_sha='code'))
        c.atomic_json(self.root/'inputs.json',{})
        cp=self.root/'checkpoint';cp.write_text('model')
        scenario=self.root/'scenario';scenario.write_text('scenes')
        co=dict(id='condition',mode='R',noise='noise',workers=4,
                checkpoint=d.artifact(cp),scenario=d.artifact(scenario),
                inputs=d.artifact(self.root/'inputs.json'),contract=d.artifact(self.root/'contract.json'),
                prefixes={f't{i}':f's{i}' for i in range(4)},rows=[dict(scene_id=f's{i}') for i in range(4)])
        path=root/'collection.json';c.atomic_json(path,co)
        from audit_selector_cfpi_deployment import ACTION_FIELDS
        action={k:([[0.,0.]] if k=='waypoints' else None) for k in ACTION_FIELDS}
        for i in range(4):
            worker=root/f'split_{i}'
            for sub in ('plan_traj','diffusiondrive_candidate_sidecars','WE_output/openscene_format/snapshot_records'):
                (worker/sub).mkdir(parents=True,exist_ok=True)
            indices=[]
            for step in range(4,13):
                sc,_=self.native_fixture()
                sc.update(scene_prefix=f't{i}',planner_step=step,checkpoint_sha256=co['checkpoint']['sha256'])
                side=worker/'diffusiondrive_candidate_sidecars'/f't{i}_{step}.pkl'
                c.atomic_pickle(side,sc)
                plan=worker/'plan_traj'/f't{i}_{step}.npy';np.save(plan,sc['deployed_trajectory'])
                indices.append(dict(prefix=f't{i}',step=step,plan_idx=0))
                if step<12:
                    rec=dict(collection=d.artifact(path),validation=s.validate_native(sc,co,'code'),
                             state_step=step-1,sidecar=d.artifact(side),plan=d.artifact(plan),
                             actual_action=action,expected_action=action,candidate_trajectory_max_abs=0.)
                    c.atomic_json(worker/'WE_output/openscene_format/snapshot_records'/f't{i}_{step}.json',rec)
            with (worker/'plan_traj/plan_idx.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=['prefix','step','plan_idx'])
                writer.writeheader();writer.writerows(indices)
            with (worker/'WE_output/openscene_format/all_scenes_pdm_averages_R.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=['token',*d.METRICS])
                writer.writeheader();writer.writerow(dict(token=f's{i}',**values()))
        with patch.object(audit,'load_current_reports',return_value=([],{f's{i}':[True] for i in range(4)})),\
             patch.object(audit,'completed_scenes',return_value=([],{f's{i}' for i in range(4)})):
            result=audit.audit(path)
            self.assertEqual(result['scenes'],4);self.assertEqual(result['terminal_publications'],4)
            self.assertEqual(sum(len(v) for v in result['records'].values()),32)
            # Corrupt a published index: unchanged sidecars must not hide it.
            idx=root/'split_0/plan_traj/plan_idx.csv'
            idx.write_text(idx.read_text().replace('t0,4,0','t0,4,1'))
            with self.assertRaisesRegex(RuntimeError,'planner CSV'):
                audit.audit(path)
    def test_native_rejects_route_noise_and_selection_drift(self):
        for key,value in [('candidate_noise_namespace','wrong'),('checkpoint_sha256','wrong'),
                          ('cfpi_deployment',{}),('selected_index',1),('scene_prefix','other')]:
            sc,co=self.native_fixture();sc[key]=value
            with self.assertRaises(RuntimeError):s.validate_native(sc,co,'code')
    def test_fixed_scope(self):
        self.assertEqual(len(s.conditions()),12)
        self.assertEqual({p for p,b in s.conditions()},{'scalar_v3','P1_seed2'})
        self.assertFalse(s.EXPOSURE['training_authorized'])
    def test_no_performance_gate_in_evaluation(self):
        r=runner.Runner.__new__(runner.Runner);r.run=self.root;calls=[]
        c.atomic_json(self.root/'inputs.json',{})
        c.atomic_json(self.root/'bridge_report.json',dict(status='PASS',inputs=d.artifact(self.root/'inputs.json'),conditions=[]))
        r.helper=lambda *a,**kw:calls.append(('helper',a))
        r.collection=lambda *a:calls.append(('collection',a))
        r.record=lambda *a:None
        r.evaluate()
        self.assertEqual(sum(x[0]=='collection' for x in calls),8)
        self.assertEqual(sum(x[0]=='helper' and x[1][0]=='open' for x in calls),4)
        self.assertEqual(calls[-1][1][0],'report')
    def test_bridge_required(self):
        r=runner.Runner.__new__(runner.Runner);r.run=self.root
        c.atomic_json(self.root/'bridge_report.json',dict(status='FAIL'))
        with self.assertRaises(RuntimeError):r.evaluate()
    def test_comparison_retains_negative_effects(self):
        rows=[dict(scene_id='s0',origin_log='l0'),dict(scene_id='s1',origin_log='l1')]
        ref={r['scene_id']:values() for r in rows}
        cur={r['scene_id']:values(score=.5,success=0) for r in rows}
        v=reporting.comparison(cur,ref,rows)
        self.assertEqual(v['broken'],2);self.assertLess(v['delta']['score'],0)
        self.assertFalse(v['intervals']['score']['seed_average_first'])
    def test_report_incomplete_is_not_success(self):
        inp=dict(artifacts={},models={})
        with patch.object(s,'checked_inputs',return_value=inp):
            q=reporting.report(self.root)
        self.assertEqual(q['status'],'INCOMPLETE');self.assertEqual(len(q['missing']),12)
    def test_four_allocated_ids_and_native_flags(self):
        r=runner.Runner.__new__(runner.Runner);r.run=self.root
        r.devices=['3','5','7','9'];r.alg=ROOT/'projects/AlgEngine';r.sim=ROOT/'projects/SimEngine'
        r.env=dict(ALGENGINE_PYTHON='alg-python',SIMENGINE_PYTHON='sim-python',CUDA_VISIBLE_DEVICES='3,5,7,9')
        r.config=r.alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_p1_snapshot.py'
        root=self.root/'collections/full_NR_P1_seed2';root.mkdir(parents=True)
        cp=self.root/'checkpoint.pth';cp.write_text('weights')
        scenario=self.root/'scenarios.pkl';scenario.write_text('scene')
        c.atomic_json(self.root/'run_contract.json',{})
        co=dict(id=root.name,contract=d.artifact(self.root/'run_contract.json'),noise=s.FORMAL_NOISE,
                mode='NR',scenario=d.artifact(scenario),asset_folder='/assets',checkpoint=d.artifact(cp))
        c.atomic_json(root/'collection.json',co)
        captured=[]
        r.execute=lambda jobs,*args,**kwargs:captured.extend(jobs)
        r.helper=lambda *args,**kwargs:None
        r.collection(root/'collection.json')
        self.assertEqual(len(captured),5)
        self.assertEqual([v[2]['CUDA_VISIBLE_DEVICES'] for v in captured[1:]],r.devices)
        self.assertIn('with_dense_reward_manager=false',captured[0][0])
        self.assertIn('selector_snapshot=true',captured[0][0])
        self.assertNotIn('diffusiondrive_cfpi_deployment=true',captured[0][0])
    def test_complete_negative_report_and_repeat(self):
        full=[dict(scene_id=f'l{i}-s{i}',origin_log=f'l{i}') for i in range(4)]
        common=[dict(scene_id=f'c{i}-s{i}',origin_log=f'c{i}') for i in range(4)]
        inputs=dict(artifacts={},models={},cohorts=dict(full=full,development=full[:2],common=common),
                    source_decision='STOP_EFFECT_FAIL')
        c.atomic_json(self.root/'inputs.json',inputs);c.atomic_json(self.root/'bridge_report.json',{'status':'PASS'})
        c.atomic_json(self.root/'run_contract.json',{});c.atomic_json(self.root/'audit_report.json',{})
        hs=self.root/'history.json'
        c.atomic_json(hs,dict(metrics=dict(closedloop_nonreactive=values(),closedloop_reactive=values())))
        src=self.root/'source.json'
        c.atomic_json(src,dict(reference_means={f'development_{mode}_scalar_v3':values() for mode in ('NR','R')}))
        inputs['artifacts'].update(historical_summary=d.artifact(hs),source_report=d.artifact(src))
        historical=self.write_csv([dict(token=r['scene_id'],valid='True',**values()) for r in full])
        for mode in ('nr','r'):
            inputs['artifacts']['historical_closedloop_'+mode]=d.artifact(historical)
        table={}
        for policy,block in s.conditions():
            rows=common if block.startswith('common') else full
            metrics={r['scene_id']:values(score=.4 if policy=='P1_seed2' else .8,
                                         success=0. if policy=='P1_seed2' else 1.) for r in rows}
            obj=dict(status='PASS',metrics=metrics,ade_fde=dict(ade_4s=1.,fde_4s=2.))
            path=(self.root/'openloop'/policy/block if block.startswith('open')
                  else self.root/'collections'/f'{block}_{policy}')/'audit.json'
            c.atomic_json(path,obj);table[(policy,block)]=obj
        def closed(path,*args,**kwargs):
            name=Path(path).name
            return next(v for (p,b),v in table.items() if name==b+'_'+p)
        with patch.object(s,'checked_inputs',return_value=inputs),patch.object(reporting,'checked',side_effect=closed),\
             patch.object(reporting,'checked_open',side_effect=lambda run,p,b:table[(p,b)]):
            a=reporting.report(self.root)
            b=reporting.report(self.root)
        self.assertEqual(a,b);self.assertEqual(a['decision'],'COMPLETE_OBSERVED_BEST_SNAPSHOT')
        self.assertLess(a['comparisons']['full_NR']['delta']['score'],0)
        self.assertFalse(a['performance_gate_applied'])
        self.assertEqual(len((self.root/'summary.tsv').read_text().splitlines()[0].split('\t')),29)
        self.assertTrue((self.root/'archive/release_manifest.json').exists())
    def test_partial_retry_preserves_outputs(self):
        root=self.root/'condition';(root/'split_0').mkdir(parents=True)
        (root/'split_0/result.txt').write_text('partial')
        c.atomic_json(root/'collection.json',{'immutable':True})
        destination=runner.archive_incomplete_collection(root)
        self.assertIsNotNone(destination)
        self.assertTrue(list((root/'attempt_archive').glob('*/split_0/result.txt')))
        self.assertTrue((root/'collection.json').exists())
    def test_cpu_stage_is_not_charged(self):
        r=runner.Runner.__new__(runner.Runner);r.account_idle=False
        r.charge_idle()  # no ledger required or mutated for offline stages


if __name__=='__main__':
    unittest.main()
