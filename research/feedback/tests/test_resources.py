"""CPU-only resource orchestration checks; no Ray cluster or CUDA work launched."""
import ast
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import run_selector_feedback as runner
from run_selector_cfpi_deployment import Runner as DeploymentRunner
from run_selector_rare import forecast


class ResourceTests(unittest.TestCase):
    def test_device_maps_four_eight_and_scheduler_ids(self):
        self.assertEqual(f.gpu_devices({},4),['0','1','2','3'])
        self.assertEqual(f.gpu_devices({},8),list(map(str,range(8))))
        self.assertEqual(f.gpu_devices({'CUDA_VISIBLE_DEVICES':'2, 4,6,7'},4),['2','4','6','7'])
        ids=['GPU-one','GPU-two','GPU-three','GPU-four']
        self.assertEqual(f.gpu_devices({'CUDA_VISIBLE_DEVICES':','.join(ids)},4),ids)

    def test_bad_device_maps_do_not_silently_truncate(self):
        for value in ('','-1','0,0,1,2','0,1,2,','0,1,2,3,4,5,6,7'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                f.gpu_devices({'CUDA_VISIBLE_DEVICES':value},4)
        with self.assertRaises(ValueError):
            f.gpu_devices({},2)

    def test_four_gpu_runtime_not_mistaken_for_eight_gpu_cost(self):
        old=[dict(scenes=64,seconds=1000.)]
        live=[dict(scenes=64,seconds=2000.,gpu_count=4)]
        self.assertEqual(f.normalized_runtime_samples(old)[0]['seconds'],1000.)
        self.assertEqual(f.normalized_runtime_samples(live)[0]['seconds'],1000.)
        self.assertEqual(forecast(f.normalized_runtime_samples(old),64),
                         forecast(f.normalized_runtime_samples(live),64))
        self.assertEqual(live[0]['seconds'],2000.)  # Do not mutate ledger history.

    def test_invalid_runtime_samples_rejected(self):
        for row in (dict(scenes=8,seconds=float('nan')),dict(scenes=0,seconds=2.),
                    dict(scenes=8,seconds=-1.),dict(scenes=8,seconds=2.,gpu_count=2)):
            with self.subTest(row=row),self.assertRaises(ValueError):
                f.normalized_runtime_samples([row])

    def test_idle_accounting_uses_reserved_gpu_count(self):
        for gpus in (4,8):
            with self.subTest(gpus=gpus),tempfile.TemporaryDirectory() as folder:
                obj=runner.Runner.__new__(runner.Runner)
                obj.account_idle=True
                obj.args=SimpleNamespace(gpus=gpus,gpu_hours=128.)
                obj.phase='feedback';obj.idle_since=10.
                obj.ledger_path=Path(folder)/'ledger.json'
                limits=f.budget_limits(128.)
                obj.ledger=dict(gpu_hours_used=0.,phase_gpu_hours={k:0. for k in limits},phase_limits=limits)
                with patch.object(runner.time,'monotonic',return_value=3610.):
                    obj.charge_idle()
                self.assertEqual(obj.ledger['gpu_hours_used'],float(gpus))
                self.assertEqual(d.read(obj.ledger_path)['phase_gpu_hours']['feedback'],float(gpus))

    def test_four_and_eight_gpu_cli_stages(self):
        for gpus in (4,8):
            for stage in ('audit','preflight','all','report'):
                with self.subTest(gpus=gpus,stage=stage),patch.object(sys,'argv',[
                    'run_selector_feedback.py',stage,'resource_test','--gpus',str(gpus),'--gpu-hours','128'
                ]),patch.object(runner,'Runner') as fake:
                    runner.main()
                    self.assertEqual(fake.call_args.args[0].gpus,gpus)
                    fake.return_value.dispatch.assert_called_once()

    def test_real_runner_freezes_four_gpu_contract_and_resumes_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);prior=root/'prior'
            for name in ('run_diffusiondrive_selector_feedback.sh','run_diffusiondrive_selector_feedback_8h100.sh'):
                c.atomic_json(root/name,dict(test_inventory_only=True))
            c.atomic_json(prior/'rare_inputs.json',dict(artifacts=dict(old_inputs=dict(path=str(root/'old/inputs.json')))))
            args=SimpleNamespace(run_id='four_gpu_contract',gpus=4,gpu_hours=128.,stage='audit',
                                 source_worldengine_root=root/'canonical',prior_run=prior)
            with patch.object(runner,'ROOT',root),patch.object(runner,'environment',side_effect=lambda *_:dict(
                    ALGENGINE_ROOT=str(root/'AlgEngine'),SIMENGINE_ROOT=str(root/'SimEngine'))),patch.object(
                    runner,'implementation_inventory',return_value={}),patch.object(
                    runner,'duration_samples',return_value=[dict(scenes=64,seconds=1000.)]),patch.object(runner.signal,'signal'):
                obj=runner.Runner(args)
                contract=d.read(obj.run/'run_contract.json')
                self.assertEqual(contract['gpu_count'],4)
                self.assertEqual(contract['collection_timeout_seconds'],43200)
                self.assertEqual(obj.env['CUDA_VISIBLE_DEVICES'],'0,1,2,3')
                self.assertEqual(obj.env['WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP'],'0,1,2,3')
                obj.ledger['gpu_hours_used']=3.;obj.ledger['phase_gpu_hours']['feedback']=3.
                c.atomic_json(obj.ledger_path,obj.ledger)
                obj.lock.close()
                resumed=runner.Runner(args)
                self.assertEqual(resumed.ledger['gpu_hours_used'],3.)
                resumed.lock.close()
                args.gpus=8
                with self.assertRaises(RuntimeError):
                    runner.Runner(args)
                self.assertEqual(d.read(obj.run/'run_contract.json'),contract)
                self.assertEqual(d.read(obj.ledger_path)['gpu_hours_used'],3.)

    def test_collection_launches_matching_planners_and_environment(self):
        for gpus in (4,8):
            for mode in f.MODES:
                with self.subTest(gpus=gpus,mode=mode),tempfile.TemporaryDirectory() as folder:
                    root=Path(folder)
                    obj=DeploymentRunner.__new__(DeploymentRunner)
                    obj.args=SimpleNamespace(gpus=gpus,source_worldengine_root=root)
                    obj.sim=root/'SimEngine';obj.alg=root/'AlgEngine';obj.config=root/'config.py'
                    obj.code_sha='test-code';obj.collection_timeout=21600*8//gpus
                    obj.checkpoint_manifest=root/'model.json'
                    c.atomic_json(obj.checkpoint_manifest,dict(checkpoint='frozen.pt'))
                    ids=['2','4','6','7'] if gpus==4 else list(map(str,range(8)))
                    obj.env=dict(SIMENGINE_PYTHON='sim-python',ALGENGINE_PYTHON='alg-python',
                                 CUDA_VISIBLE_DEVICES=','.join(ids),WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP=','.join(ids))
                    c.atomic_json(root/'run.json',dict(gpu_count=gpus))
                    c.atomic_json(root/'routing.json',dict(routes=[]))
                    c.atomic_json(root/'scenario.json',dict(toy=True))
                    path=root/'collections'/mode/'deployment_collection.json'
                    c.atomic_json(path,dict(collection_id=mode,research_method=f.METHOD,cohort='train',react_type=mode,
                        scenario=d.artifact(root/'scenario.json'),routing=d.artifact(root/'routing.json'),
                        run_contract=d.artifact(root/'run.json')))
                    obj.python=Mock();obj.execute=Mock()
                    obj.collection(d.artifact(path))
                    jobs=obj.execute.call_args.args[0]
                    self.assertEqual(len(jobs),gpus+1)
                    self.assertEqual(obj.execute.call_args.kwargs['timeout'],21600*8//gpus)
                    self.assertEqual(obj.execute.call_args.kwargs['gpus'],gpus)
                    self.assertEqual(jobs[0][2]['CUDA_VISIBLE_DEVICES'],','.join(ids))
                    self.assertIn('agent_policy='+('idm_policy' if mode=='R' else 'trajectory_policy'),jobs[0][0])
                    for i,job in enumerate(jobs[1:]):
                        self.assertEqual(job[2]['CUDA_VISIBLE_DEVICES'],ids[i])
                        self.assertIn(f'sim.monitored_folder={path.parent}/split_{i}/frames',job[0])
                    c.atomic_json(root/'wrong_run.json',dict(gpu_count=8 if gpus==4 else 4))
                    bad=d.read(path);bad['run_contract']=d.artifact(root/'wrong_run.json')
                    bad_path=path.parent/'wrong_collection.json';c.atomic_json(bad_path,bad)
                    obj.execute.reset_mock();obj.python.reset_mock()
                    with self.assertRaisesRegex(RuntimeError,'GPU count'):
                        obj.collection(d.artifact(bad_path))
                    obj.execute.assert_not_called()
                    obj.python.assert_not_called()

    def process_module(self):
        path=ROOT/'projects/SimEngine/worldengine/utils/multithreading/process_utils.py'
        spec=importlib.util.spec_from_file_location('feedback_process_utils_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_ray_physical_ids_map_to_logical_planner_folders(self):
        module=self.process_module()
        for ids in (['2','4','6','7'],['GPU-one','GPU-two','GPU-three','GPU-four']):
            for i,physical in enumerate(ids):
                ray=SimpleNamespace(is_initialized=lambda:True,get_gpu_ids=lambda:[physical])
                with patch.dict(sys.modules,ray=ray),patch.dict(os.environ,WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP=','.join(ids)):
                    self.assertEqual(module.get_process_id(),str(i))
                    from omegaconf import OmegaConf
                    value=module.resolve_worker_placeholders(OmegaConf.create(dict(worker_id_prefix='split_',path='run/__WORKER_ID__/frames')))
                    self.assertEqual(value.path,f'run/split_{i}/frames')

    def test_ray_map_rejects_wrong_assignment_preserves_legacy(self):
        module=self.process_module()
        for ids in ([],['9'],['2','4']):
            ray=SimpleNamespace(is_initialized=lambda:True,get_gpu_ids=lambda:ids)
            with patch.dict(sys.modules,ray=ray),patch.dict(os.environ,WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP='2,4,6,7'):
                with self.assertRaises(RuntimeError):
                    module.get_process_id()
        ray=SimpleNamespace(is_initialized=lambda:True,get_gpu_ids=lambda:[6])
        with patch.dict(sys.modules,ray=ray),patch.dict(os.environ,{},clear=True):
            self.assertEqual(module.get_process_id(),'6')

    def test_actual_scene_splitter_keeps_all_scenes_on_four_workers(self):
        # Execute the repository's pure splitter without importing the simulator.
        path=ROOT/'projects/SimEngine/worldengine/runner/builders/env_builder.py'
        tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='distribute_scenes')
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
        scope={};exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
        cfg=SimpleNamespace(distributed_mode='SCENARIO_BASED',worker_id_prefix='split_')
        worker=SimpleNamespace(config=SimpleNamespace(number_of_gpus_per_node=4))
        for n in (8,32,58,64,96,128):
            scenes={str(i):{} for i in range(n)}
            parts=scope['distribute_scenes'](cfg,scenes,worker)
            self.assertEqual(set(parts),{f'split_{i}' for i in range(4)})
            self.assertEqual([s for part in parts.values() for s in part],list(scenes))
            self.assertLessEqual(max(map(len,parts.values()))-min(map(len,parts.values())),1)


if __name__=='__main__':
    unittest.main()
