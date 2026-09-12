"""CPU-only transport/optimizer integration fixtures. No real experiment is trained."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
sys.path.insert(0,str(ROOT/'research/cfpi/tests'))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import train_selector_feedback as trainer
import selector_cfpi_deployment_router as router_module
from audit_selector_cfpi_deployment import validate_sidecar
import test_continuous_deployment as prior_tests


class IntegrationTests(unittest.TestCase):
    def test_refreshed_visitor_forced_action_and_auditor(self):
        fixture=prior_tests.DeploymentTests()
        fixture.setUp()
        try:
            fixture.manifest['routes']=fixture.manifest['routes'][:1]
            fixture.manifest['routes'][0]['start_decision']=4
            visitor=fixture.router()
            prefix={}
            for step in (4,5,6):
                value=fixture.sidecar(visitor,step)
                path=fixture.root/f'source_{step}.pkl';c.atomic_pickle(path,value)
                prefix[str(step)]=d.artifact(path)
            target=dict(decision_step=6,prefix=prefix,source_fingerprint=f.fingerprint(value),
                        visiting_index=2,forced_index=7,source_collection=dict(sha256='source'),sentinel=False)
            fixture.manifest['research_method']=f.METHOD
            fixture.manifest['routes'][0].update(feedback_target=target,continuation_model_key='fold0')
            router=fixture.router()
            collection=dict(fixture.collection(router),research_method=f.METHOD,cohort='train',react_type='NR')
            for step in (4,5,6,7,8):
                sidecar=fixture.sidecar(router,step)
                sidecar['selector_rollout_contract'].update(
                    research_method=f.METHOD,react_type='NR',source_data_split='train',
                    development_consumed=False,test_consumed=False,
                    legacy_exposed_benchmark=True,independent_unseen_test=False)
                result=validate_sidecar(sidecar,collection,router.manifest,'code')
                self.assertEqual(result['selected_index'],7 if step==6 else 2)
                self.assertEqual(int(np.asarray(sidecar['current_logits']).argmax()),2)
                if step==6:
                    del sidecar['cfpi_deployment']['feedback_intervention']
                    with self.assertRaises(RuntimeError):
                        validate_sidecar(sidecar,collection,router.manifest,'code')
        finally:
            fixture.tearDown()

    def test_source_cap_exclusion_and_strata(self):
        pools=dict(rare_union=set(),matched_common=set());metrics={}
        from test_feedback import metric
        for prefix,n,family,value in (
            ('failed',25,'rare_union',metric(.1,0)),('slow',50,'rare_union',metric(.7,1,.3)),
            ('other',70,'rare_union',metric()),('common',50,'matched_common',metric())):
            for i in range(n):
                scene=f'{prefix}_{i//2}-'+format(i,'016x')
                pools[family].add(scene);metrics[scene]=copy.deepcopy(value)
        rows=f.source_selection(pools,metrics,{'failed_0','common_0'})
        self.assertEqual(len(rows),128)
        self.assertFalse({r['origin_log'] for r in rows}&{'failed_0','common_0'})
        from collections import Counter
        self.assertLessEqual(max(Counter(r['origin_log'] for r in rows).values()),2)
        self.assertEqual(dict(Counter(r['stratum'] for r in rows)),dict(f.STRATA))
        self.assertEqual(rows,f.source_selection(pools,metrics,{'common_0','failed_0'}))

    def test_real_training_loop_resume_and_shared_parent(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            marker=root/'marker.json';c.atomic_json(marker,{'toy':True})
            c.atomic_json(root/'feedback_inputs.json',dict(artifacts=dict(scalar_manifest=d.artifact(marker)),excluded_logs=[]))
            c.atomic_json(root/'run_contract.json',dict(code_sha='toy'))
            c.atomic_json(root/'dense/manifest.json',dict(toy=True))
            def load_incumbent(_):
                model=prior_tests.ToyRefit()
                with torch.no_grad():model.logits.zero_()
                return model,{},{}
            def score(model,rows,ids,device,mode):
                return model.logits[None].expand(len(ids),-1)
            def load_selector(path):
                payload=torch.load(path,map_location='cpu')
                model=prior_tests.ToyRefit();model.load_state_dict(payload['scene_selector_state'])
                return model.eval(),payload
            def feedback_rows(run,arm):
                rows=[dict(mode=mode,candidate_indices=[0,1],preference=(1,0,.5),
                           candidate_rewards=np.arange(20,dtype=np.float32)/20,
                           candidate_reward_valid_mask=np.ones(20,dtype=bool)) for mode in f.MODES for _ in range(96)]
                return rows,[d.artifact(marker)]
            class ToyDense:
                def __init__(self,run):self.calls=0
                def sample(self,rng):
                    self.calls+=1
                    if interrupt[0] and self.calls==251:
                        raise RuntimeError('intentional toy interruption')
                    rng.integers(100,size=16)
                    return [feedback_rows(None,None)[0][0]]*16
            interrupt=[True]
            with patch.object(trainer,'Dense',ToyDense), patch.object(trainer,'feedback_rows',side_effect=feedback_rows), patch.object(trainer.m,'load_incumbent',side_effect=load_incumbent), patch.object(trainer.m,'score',side_effect=score), patch.object(trainer.m,'load_selector',side_effect=load_selector), patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError,'intentional'):
                    trainer.train(root,'shared',0,'cpu')
                checkpoint=torch.load(root/'train/shared_seed0/resume.pt',map_location='cpu')
                self.assertEqual(checkpoint['step'],250)
                # Preserve interrupted artifacts in the test directory; resume same parent.
                interrupt[0]=False
                result=trainer.train(root,'shared',0,'cpu')
                resumed=torch.load(result['selector']['path'])['scene_selector_state']
                # Second seed0 run has identical data/RNG but no interruption.
                control=root/'control'
                c.atomic_json(control/'feedback_inputs.json',d.read(root/'feedback_inputs.json'))
                c.atomic_json(control/'run_contract.json',d.read(root/'run_contract.json'))
                c.atomic_json(control/'dense/manifest.json',dict(toy=True))
                uninterrupted=trainer.train(control,'shared',0,'cpu')
                expected=torch.load(uninterrupted['selector']['path'])['scene_selector_state']
                for key in expected:self.assertTrue(torch.equal(resumed[key],expected[key]))
                s=trainer.train(root,'S',0,'cpu')
                u=trainer.train(root,'U',0,'cpu')
                self.assertEqual(s['provenance']['parent'],u['provenance']['parent'])
                state_s=torch.load(s['selector']['path'])['scene_selector_state']
                state_u=torch.load(u['selector']['path'])['scene_selector_state']
                for key in state_s:self.assertTrue(torch.equal(state_s[key],state_u[key]))
                self.assertEqual(s['provenance']['total_optimizer_steps'],1000)


if __name__=='__main__':
    unittest.main()
