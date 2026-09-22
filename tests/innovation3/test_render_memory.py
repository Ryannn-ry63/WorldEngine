from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'projects/SimEngine'))
from omegaconf import OmegaConf
from worldengine.engine.engine_utils import initialize_engine, close_engine
from worldengine.manager.render_manager import RenderManager


class RenderMemoryTest(unittest.TestCase):
    def setUp(self):
        self.engine = initialize_engine(OmegaConf.create({}))
        self.engine.register_manager('data_manager', Mock(PRIORITY=1))
        self.render = RenderManager.__new__(RenderManager)
        self.render.current_scene_id = 'test'
        self.render.rendering_results = {}
        self.images = {'camera': object()}
        self.render.render = Mock(return_value=self.images)

    def tearDown(self):
        close_engine()

    def test_default_preserves_file_bridge(self):
        self.assertIs(self.render.get_observations(), self.images)
        self.engine.data_manager.save_current_frame_data.assert_called_once()
        self.assertIs(self.render.rendering_results[0], self.images)

    def test_memory_mode_neither_saves_nor_retains_frames(self):
        self.assertIs(self.render.get_observations(persist=False, cache=False), self.images)
        self.engine.data_manager.save_current_frame_data.assert_not_called()
        self.assertEqual(self.render.rendering_results, {})


if __name__ == '__main__':
    unittest.main()
