import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'projects/AlgEngine/scripts/diffusiondrive'))
from grpo_selector_v3_cached_common import SceneConditionedTrajectorySetSelector
from innovation3.export_online_selector_v3 import main
from innovation3.learner import V3_INITIALIZED_SELECTOR_FINETUNE, OnlineV3Learner
from innovation3.paths import sha256_file


class ExportTest(unittest.TestCase):
    def test_initialized_selector_export_is_deployable_and_disposable(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        config = dict(feature_dim=16, model_dim=16, route_bev_dim=16, context_dim=16,
                      geometry_hidden_dim=8, num_heads=4, feedforward_dim=32,
                      num_set_layers=1)
        model = SceneConditionedTrajectorySetSelector(**config).float().eval()
        with torch.no_grad():
            model.delta_head[-1].weight.normal_(0, .02)
            model.delta_head[-1].bias.fill_(.3)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offline_path = root / 'offline.pt'
            offline_payload = dict(schema_version=3, method='scene_conditioned_exact_group_grpo',
                                   scene_selector_config=config,
                                   scene_selector_state={k: v.detach().cpu() for k, v in model.state_dict().items()})
            torch.save(offline_payload, offline_path)
            source = dict(kind='offline_selector_file', path=str(offline_path),
                          sha256=sha256_file(offline_path), payload_schema=3,
                          method=offline_payload['method'], scene_selector_config=config)
            learner = OnlineV3Learner(model, parameterization=V3_INITIALIZED_SELECTOR_FINETUNE,
                                      initialization_source=source)
            from test_online_learner import context
            x, base = context(), torch.randn(1, 20)
            learner.choose(x, base, 'one')
            learner.update(torch.arange(20)[None, :], 'one')
            with torch.no_grad(): expected = learner.current_logits(base, x)
            checkpoint_path = root / 'learner.pt'
            torch.save(dict(kind='DISPOSABLE_STRICT_ONLINE_PROBE_NOT_FORMAL_TRAINING',
                            learner=learner.state_dict()), checkpoint_path)
            output = root / 'exported.pt'
            manifest = root / 'export_manifest.json'
            argv = ['export_online_selector_v3', '--checkpoint', str(checkpoint_path),
                    '--offline-selector', str(offline_path), '--output', str(output),
                    '--manifest', str(manifest)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(main())
            exported = torch.load(output, map_location='cpu')
            self.assertEqual(exported['schema_version'], 3)
            self.assertEqual(exported['method'], 'scene_conditioned_exact_group_grpo')
            self.assertEqual(exported['online_export']['parameterization'],
                             V3_INITIALIZED_SELECTOR_FINETUNE)
            self.assertTrue(exported['online_export']['disposable'])
            self.assertEqual(set(exported), {
                'schema_version', 'method', 'ablation', 'temperature', 'learning_rate',
                'kl_weight', 'train_seed', 'epoch', 'scene_selector_config',
                'scene_selector_state', 'online_export'})
            self.assertEqual(set(exported['scene_selector_state']), set(learner.selector.state_dict()))
            for key, value in learner.selector.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), exported['scene_selector_state'][key]))
            deployed = SceneConditionedTrajectorySetSelector(**exported['scene_selector_config']).eval()
            deployed.load_state_dict(exported['scene_selector_state'], strict=True)
            with torch.no_grad(): actual = base + deployed(**x)
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(int(expected.argmax()), int(actual.argmax()))
            self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in original.items()))
            self.assertNotIn('reference', exported)
            report = json.loads(manifest.read_text())
            self.assertEqual(report['status'], 'PASS_DISPOSABLE_ONLINE_SELECTOR_EXPORT')
            self.assertTrue(report['avoids_legacy_residual_double_add'])
            # Exercise normal materialization and its frozen-baseline audit.
            import materialize_grpo_selector_v3 as materialize
            import audit_grpo_selector_v3_checkpoint as audit
            baseline = root / 'baseline.pt'
            torch.save({'state_dict': {'backbone.weight': torch.randn(3, 3),
                                      'planning_head.original_classifier': torch.randn(2, 3)}}, baseline)
            final, materialized_report = root/'deployed.pt', root/'materialized.json'
            command = ['materialize', '--baseline', str(baseline), '--scene-selector-state', str(output),
                       '--expected-selector-sha256', sha256_file(output), '--output', str(final),
                       '--manifest', str(materialized_report)]
            with patch.object(sys, 'argv', command), patch.object(materialize, 'BASELINE_SHA256', sha256_file(baseline)), contextlib.redirect_stdout(io.StringIO()):
                materialize.main()
            self.assertTrue(json.loads(materialized_report.read_text())['disposable'])
            full = torch.load(final, map_location='cpu')
            self.assertTrue(full['meta']['diffusiondrive_grpo_selector_v3']['online_export']['disposable'])
            audit_path = root/'audit.json'
            with patch.object(sys, 'argv', ['audit', '--baseline', str(baseline), '--checkpoint', str(final),
                                            '--output', str(audit_path)]), contextlib.redirect_stdout(io.StringIO()):
                audit.main()
            self.assertEqual(json.loads(audit_path.read_text())['changed_baseline_tensor_count'], 0)
            deployed.load_state_dict({k.removeprefix(materialize.PREFIX): v for k, v in full['state_dict'].items()
                                     if k.startswith(materialize.PREFIX)}, strict=True)
            with torch.no_grad(): self.assertTrue(torch.equal(base + deployed(**x), expected))
            # A foreign source is rejected without creating an output.
            offline_payload['scene_selector_state']['delta_head.3.bias'] += 1
            torch.save(offline_payload, offline_path)
            rejected_output = root/'rejected.pt'
            rejected_manifest = root/'rejected.json'
            with patch.object(sys, 'argv', ['export', '--checkpoint', str(checkpoint_path),
                    '--offline-selector', str(offline_path), '--output', str(rejected_output),
                    '--manifest', str(rejected_manifest)]):
                with self.assertRaisesRegex(ValueError, 'SHA256'): main()
            self.assertFalse(rejected_output.exists())
            self.assertFalse(rejected_manifest.exists())


if __name__ == '__main__':
    unittest.main()
