import json
import tempfile
import unittest
from pathlib import Path

from innovation3.online_session_probe import _episode_settings


class SessionSettingsTest(unittest.TestCase):
    def test_episode_settings_keep_assignment_and_advance_rng_namespace(self):
        base = {'scenario_root': '/scenarios', 'online_candidate_seed': 9}
        row = dict(scene_id='scene-a', scene_path='/scenes/a.pkl',
                   candidate_seed=11, scene_seed=17)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.json'
            _episode_settings(base, row, path, 2)
            cfg = json.loads(path.read_text())
        stride = 1000003
        self.assertEqual(cfg['visual_scene_id'], 'scene-a')
        self.assertEqual(cfg['visual_scene_path'], '/scenes/a.pkl')
        self.assertEqual(cfg['online_candidate_seed'], 11 + 2 * stride)
        self.assertEqual(cfg['online_scene_seed'], 17 + 2 * stride)
        self.assertEqual(cfg['online_action_seed'], 11 + 2 * stride + 17)
        self.assertEqual(cfg['scenario_root'], '/scenarios')


if __name__ == '__main__':
    unittest.main()
