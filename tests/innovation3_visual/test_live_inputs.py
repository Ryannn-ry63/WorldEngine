import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'),str(ROOT/'tests/innovation3')]
from innovation3.visual_model import configuration
from innovation3.live_inputs import LiveInputs
from innovation3.visual_parity import assert_same, file_oracle
from innovation3.image_transport import metadata_to_wire,metadata_from_wire
from innovation3.transport import encode
from worldengine.online.observations import FrameHistory
from visual_fixture import raw_frame


class LiveInputsTest(unittest.TestCase):
    def test_full_original_pipeline_parity_and_no_live_file_reads(self):
        cfg=json.loads((ROOT.parent/'registry/online_runtime.json').read_text())
        config=configuration(cfg,0)
        live=LiveInputs(config.data.test)
        history=FrameHistory(); frames=[]; cameras=[]
        rng=np.random.RandomState(5)
        dataset_id=id(live.dataset)
        for step in range(6):
            frame=history.append(raw_frame(step))
            # JSON sorts keys; wire format must retain original camera order.
            frame=metadata_from_wire(json.loads(encode(metadata_to_wire(frame))))
            views={name:cv2.imencode('.jpg',rng.randint(0,256,(48,64,3),dtype=np.uint8))[1].tobytes()
                   for name in frame['cams']}
            frames.append(frame);cameras.append(views)
            if step<3:continue
            frames,cameras=frames[-4:],cameras[-4:]
            expected=file_oracle(config.data.test,frames,cameras)
            with patch('builtins.open',side_effect=AssertionError('Live path tried file IO')):
                actual=live.prepare(frames,cameras)
            assert_same(actual,expected)
            self.assertEqual(id(live.dataset),dataset_id)
        bad=copy.deepcopy(frames);bad[-1]['scene_token']='other'
        with self.assertRaises(ValueError):live.prepare(bad,cameras)
        badviews=copy.deepcopy(cameras);badviews[-1]=dict(reversed(list(badviews[-1].items())))
        with self.assertRaises(ValueError):live.prepare(frames,badviews)
        with self.assertRaises(ValueError):live.prepare(frames[:-1],cameras[:-1])
