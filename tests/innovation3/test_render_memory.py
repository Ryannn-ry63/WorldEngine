from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'projects/SimEngine'))
from omegaconf import OmegaConf
from worldengine.engine.engine_utils import initialize_engine, close_engine
from worldengine.manager.render_manager import RenderManager
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD


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

    def test_render_does_not_translate_agent_owned_bounding_box(self):
        class Policy:
            is_current_step_valid = True
        class Agent:
            policy = Policy()
            bounding_box = np.array([10., 20., 0., 4., 2., .0])
        agent = Agent()
        scene = {SD.ID: 'test', SD.BASE_TIMESTAMP: 0,
                 SD.METADATA: {SD.OLD_ORIGIN_IN_CURRENT_COORDINATE: np.array([3., 5.])},
                 SD.CAMERAS: {}, SD.LIDAR: {}}
        engine = Mock()
        engine.managers = {'agent_manager': Mock(get_dynamic_agents={'car': agent})}
        engine.episode_step = 0
        engine.sim_time_interval = .05
        engine.current_scene = scene
        self.render.local2global_translation_xy = np.array([-3., -5.])
        self.render.base_timestamp = 0
        self.render.current_scene = scene
        self.render.renderer = Mock()
        self.render.renderer.render.return_value = self.images
        before = agent.bounding_box.copy()
        with patch('worldengine.manager.base_manager.get_engine', return_value=engine):
            RenderManager.render(self.render)
        np.testing.assert_array_equal(agent.bounding_box, before)
        np.testing.assert_array_equal(self.render.renderer.render.call_args.args[0]['agent_state']['car'][:2], [7., 15.])


if __name__ == '__main__':
    unittest.main()
