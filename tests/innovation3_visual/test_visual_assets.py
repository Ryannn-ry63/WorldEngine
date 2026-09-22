import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'projects/AlgEngine/scripts/diffusiondrive'))
from innovation3.visual_assets import visual_asset


class VisualAssetTest(unittest.TestCase):
    def test_requires_exact_scene_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_id = 'scene-a'
            asset = root / scene_id / 'background' / (scene_id + '.ckpt')
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b'checkpoint')
            cfg = {'asset_root': str(root), 'visual_scene_id': scene_id}
            scenes = {scene_id: {'metadata': {}}}
            selected, checked_root, checked_asset = visual_asset(cfg, scenes)
            self.assertEqual(selected, scene_id)
            self.assertEqual(checked_root, root.resolve())
            self.assertEqual(checked_asset, asset.resolve())

    def test_does_not_substitute_missing_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {'asset_root': directory, 'visual_scene_id': 'scene-a'}
            scenes = {'scene-a': {'metadata': {}}, 'scene-b': {'metadata': {}}}
            with self.assertRaisesRegex(FileNotFoundError, 'no automatic substitution'):
                visual_asset(cfg, scenes)


if __name__ == '__main__':
    unittest.main()
