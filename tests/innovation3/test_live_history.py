import copy
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'projects/SimEngine'))
from worldengine.online.observations import FrameHistory, FRAME_FIELDS
from worldengine.components.agents.client.navformer_client import NAVFormerClient
from visual_fixture import raw_frame


class HistoryTest(unittest.TestCase):
    def test_matches_actual_file_client_after_window_rolls(self):
        history=FrameHistory()
        client=NAVFormerClient.__new__(NAVFormerClient)
        client.episode_data_processed={}; client.seq_index=[]
        with tempfile.TemporaryDirectory() as directory:
            client.data_folder=Path(directory)
            for step in range(7):
                raw=raw_frame(step)
                original=pickle.dumps(raw)
                actual=history.append(raw)
                self.assertEqual(original,pickle.dumps(raw))
                client.process_frame(copy.deepcopy(raw),step)
                with (Path(directory)/f'scene_{step}.pkl').open('rb') as f: expected=pickle.load(f)
                for key in FRAME_FIELDS:
                    if isinstance(actual[key],np.ndarray): np.testing.assert_array_equal(actual[key],expected[key])
                self.assertFalse(actual['gt_fut_bbox_sdc_mask'].any())
                self.assertFalse(actual['gt_fut_bbox_sdc_lidar'].any())
                self.assertLessEqual(len(history.raw),4)
                self.assertLessEqual(len(history.frames),4)
            with self.assertRaises(ValueError): history.append(raw_frame(6))
            with self.assertRaises(ValueError): history.append(raw_frame(8))
            with self.assertRaises(ValueError): history.append(raw_frame(7,'other'))
