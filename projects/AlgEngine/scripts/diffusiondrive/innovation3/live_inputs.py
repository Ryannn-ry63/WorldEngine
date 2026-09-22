"""Resident original four-frame pipeline with in-memory JPEG reads only."""
import copy
import pickle
import tempfile
import cv2
import numpy as np
from mmcv.parallel import collate
from mmdet3d_plugin.datasets.navsim_openscene_closed_loop import NavSimOpenSceneE2EClosedLoop
from mmdet3d_plugin.datasets.pipelines.loading import LoadMultiViewImageFromFilesWithDownsample


class MemoryJpegLoader(LoadMultiViewImageFromFilesWithDownsample):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.images = {}

    def imread(self, path):
        if str(path) not in self.images:
            raise ValueError('Camera missing from live window: ' + str(path))
        result = cv2.imdecode(np.frombuffer(self.images[str(path)], np.uint8), self.flag)
        if result is None:
            raise ValueError('Invalid live JPEG payload')
        return result


class ResidentDataset(NavSimOpenSceneE2EClosedLoop):
    def load_annotations(self, ann_file):
        self.index_map = []
        return []


class LiveInputs:
    def __init__(self, dataset_cfg):
        cfg = copy.deepcopy(dict(dataset_cfg))
        cfg.pop('type', None)
        cfg.update(metric_cache_path=None, test_mode=True)
        self._metadata_file = tempfile.NamedTemporaryFile(prefix='innovation3-live-', suffix='.pkl', delete=False)
        pickle.dump({'infos': []}, self._metadata_file)
        self._metadata_file.close()
        cfg['ann_file'] = self._metadata_file.name
        loader = cfg['pipeline'][0]
        if loader['type'] != 'LoadMultiViewImageFromFilesWithDownsample':
            raise ValueError('Unreviewed live image loader')
        options = dict(loader); options.pop('type'); options['img_root'] = ''
        self.loader = MemoryJpegLoader(**options)
        cfg['pipeline'][0] = self.loader
        self.dataset = ResidentDataset(**cfg)
        if self.dataset.queue_length != 4:
            raise ValueError('Only the original four-frame model is supported')
        self.build_count = 1

    def close(self):
        import os
        try: os.unlink(self._metadata_file.name)
        except FileNotFoundError: pass

    def prepare(self, frames, camera_frames):
        if len(frames) != 4 or len(camera_frames) != 4:
            raise ValueError('A complete four-frame live window is required')
        for a, b in zip(frames, frames[1:]):
            if (a['scene_token'] != b['scene_token'] or b['frame_idx'] != a['frame_idx']+1 or
                    b['timestamp'] <= a['timestamp']):
                raise ValueError('Mixed or nonconsecutive live window')
        images = {}
        for frame, cameras in zip(frames, camera_frames):
            if list(cameras) != list(frame['cams']):
                raise ValueError('Camera order/name mismatch')
            for name, camera in frame['cams'].items():
                path = camera['data_path']
                if path in images:
                    raise ValueError('Repeated image identity in live window')
                images[path] = cameras[name]
        self.loader.images = images
        self.dataset.data_infos = copy.deepcopy(list(frames))
        self.dataset.index_map = [3]
        sample = self.dataset[0]
        if sample is None:
            raise ValueError('Original input pipeline rejected live frame')
        return collate([sample], samples_per_gpu=1)
