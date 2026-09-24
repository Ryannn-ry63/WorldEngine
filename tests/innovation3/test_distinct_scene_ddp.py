"""CPU acceptance for scene assignments and real mixed-signal collectives."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
from innovation3.scene_manifest import load_assignments, origin_log, validate_scene_reports


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManifestTest(unittest.TestCase):
    def fixture(self, root):
        settings = root/'settings.json'
        source = root/'scenarios/original/navtrain_failures_per1/all_scenarios.pkl'
        source.parent.mkdir(parents=True); source.write_bytes(b'registered fixture')
        settings.write_text(json.dumps(dict(scenario_root=str(root/'scenarios'), asset_root=str(root/'assets'))))
        rows = []
        for index in range(2):
            log = '2021.06.09.11.54.15_veh-'+str(12+index)
            sid = log+'_00015_00259-1234567890abcdef'
            asset = root/'assets'/sid/'background'/(sid+'.ckpt')
            asset.parent.mkdir(parents=True); asset.write_bytes(b'asset fixture')
            scene = root/(sid+'.pkl'); scene.write_bytes(b'scene fixture')
            rows.append(dict(scene_id=sid,origin_log=log,candidate_seed=200+index,scene_seed=100+index,
                             scene_path=str(scene),scene_sha256=_hash(scene),asset_sha256=_hash(asset)))
        manifest = dict(schema=1,purpose='distinct_scene_engineering_pilot',formal_ready=False,reaction='R',steps=8,
                        settings_sha256=_hash(settings),scenario_source_sha256=_hash(source),scenes=rows)
        path = root/'manifest.json'; path.write_text(json.dumps(manifest))
        return path,settings,manifest

    def test_distinct_assignments_and_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            p,c,m = self.fixture(Path(d)); a,rows=load_assignments(p,c,2)
            self.assertEqual(rows,m['scenes'])
            Path(rows[0]['scene_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'scene checksum'):load_assignments(p,c,2)

    def test_initialized_manifest_binds_mode_and_settings_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p, c, m = self.fixture(Path(d))
            cfg = json.loads(c.read_text())
            cfg['online_parameterization'] = 'v3_initialized_selector_finetune'
            c.write_text(json.dumps(cfg))
            m['settings_sha256'] = _hash(c)
            p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'Legacy'): load_assignments(p, c, 2)
            m.update(purpose='distinct_scene_engineering_pilot_v3_initialized_selector',
                     parameterization='v3_initialized_selector_finetune')
            p.write_text(json.dumps(m))
            self.assertEqual(len(load_assignments(p, c, 2)[1]), 2)
            c.write_text(c.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'settings SHA'): load_assignments(p, c, 2)

    def test_same_original_log_different_clips_rejected(self):
        self.assertEqual(origin_log('2021.06.09.11.54.15_veh-12_00015_00259'),
                         origin_log('2021.06.09.11.54.15_veh-12_00300_00800'))
        with tempfile.TemporaryDirectory() as d:
            p,c,m=self.fixture(Path(d)); row=m['scenes'][1]
            row['scene_id']='2021.06.09.11.54.15_veh-12_00300_00800-abcdef1234567890'
            row['origin_log']='2021.06.09.11.54.15_veh-12'
            p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'Duplicate'):load_assignments(p,c,2)

    def test_settings_and_wrong_worker_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,c,m=self.fixture(Path(d)); c.write_text(c.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'settings SHA256'):load_assignments(p,c,2)
            reports=[dict(worker=dict(scene=r['scene_id'],asset=dict(sha256=r['asset_sha256']),
                         source=dict(sha256=m['scenario_source_sha256'])),seed_namespaces=dict(
                         candidate_seed=r['candidate_seed'],scene_seed=r['scene_seed'])) for r in m['scenes']]
            self.assertTrue(validate_scene_reports(reports,m['scenes'],m))
            reports[1]['worker']['scene']=reports[0]['worker']['scene']
            with self.assertRaisesRegex(RuntimeError,'different scene'):validate_scene_reports(reports,m['scenes'],m)


def _mixed_signal_rank(rank, rendezvous, directory, mode="frozen_v3_plus_zero_residual"):
    import torch
    import torch.distributed as dist
    from innovation3 import ddp_online_probe as ddp
    from innovation3.learner import OnlineV3Learner
    from innovation3.live_learning import state_digest
    from test_online_learner import model, context
    torch.set_num_threads(1); torch.manual_seed(19)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    ddp.torch,ddp.dist,ddp.state_digest=torch,dist,state_digest
    try:
        cls=ddp.distributed_learner_class(OnlineV3Learner)
        trained = model().eval()
        if mode != 'mixed':
            with torch.no_grad(): trained.delta_head[-1].weight.normal_(0, .02)
        # In the mixed case both state dicts are identical zero-output V3;
        # mode metadata itself must trigger the rejection.
        selected_mode = ('v3_initialized_selector_finetune' if rank else
                         'frozen_v3_plus_zero_residual') if mode == 'mixed' else mode
        try:
            learner=cls(trained, seed=rank, parameterization=selected_mode,
                        initialization_source=dict(kind='selector_state_fingerprint'))
        except RuntimeError as error:
            if mode != 'mixed' or 'initialized' not in str(error): raise
            Path(directory, str(rank)+'.json').write_text(json.dumps(dict(rejected=True)))
            return
        if mode == 'mixed': raise AssertionError('Mixed DDP modes were accepted')
        torch.manual_seed(40+rank); x=context(); base=torch.randn(1,20)
        traces=[]
        for step in range(3):
            learner.choose(x,base,('scene'+str(rank),step))
            before=state_digest(dict(selector=learner.selector.state_dict(),optimizer=learner.optimizer.state_dict()))
            # Each rank supplies signal on a different step, then both tie.
            rewards=[i/19 for i in range(20)] if step==rank else [1.]*20
            report=learner.update([rewards],('scene'+str(rank),step))
            after=state_digest(dict(selector=learner.selector.state_dict(),optimizer=learner.optimizer.state_dict()))
            assert report['local_group_signal']==(step==rank)
            assert report['optimized']==(step<2)
            if step==2: assert before==after
            traces.append(report)
        assert learner.version==2 and learner.attempts==3
        Path(directory, str(rank)+'.json').write_text(json.dumps(traces))
    finally:dist.destroy_process_group()


class MixedSignalTest(unittest.TestCase):
    def test_initialized_gloo_and_mixed_modes_rejected(self):
        import torch.multiprocessing as mp
        for mode in ('v3_initialized_selector_finetune', 'mixed'):
            with tempfile.TemporaryDirectory() as d:
                mp.spawn(_mixed_signal_rank, args=(str(Path(d)/'rendezvous'), d, mode),
                         nprocs=2, join=True)
                rows=[json.loads(Path(d,str(i)+'.json').read_text()) for i in range(2)]
                if mode == 'mixed': self.assertTrue(all(x['rejected'] for x in rows))
                else:
                    self.assertEqual([r['policy_version'] for r in rows[0]], [1, 2, 2])
                    self.assertTrue(all(r['ddp_state_parity'] for rs in rows for r in rs))

    def test_real_gloo_mixed_signal_updates_and_global_skip(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(_mixed_signal_rank,args=(str(Path(d)/'rendezvous'),d),nprocs=2,join=True)
            rows=[json.loads(Path(d,str(i)+'.json').read_text()) for i in range(2)]
            self.assertEqual([r['policy_version'] for r in rows[0]],[1,2,2])
            self.assertTrue(all(r['ddp_state_parity'] for rs in rows for r in rs))


if __name__=='__main__':unittest.main()
