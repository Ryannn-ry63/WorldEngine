"""Read-only canonical rendering and bounded, current/past-only frame metadata.

Renderer and history are explicit outside the dynamics snapshot, are never
advanced by branches, and are NOT covered by the v1 restore guarantee.
"""
import copy
import json
import random
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from omegaconf import OmegaConf
from worldengine.manager.render_manager import RenderManager
from worldengine.manager.data_manager import DataManager
from worldengine.components.agents.client.navformer_client import NAVFormerClient

FRAME_FIELDS = (
    'token', 'frame_idx', 'timestamp', 'log_name', 'log_token', 'scene_name',
    'scene_token', 'lidar_path', 'sample_prev', 'sample_next', 'lidar2global',
    'lidar2ego', 'ego2global', 'ego2global_translation', 'ego2global_rotation',
    'cams', 'can_bus', 'driving_command', 'gt_pre_bbox_sdc_lidar',
    'gt_fut_bbox_sdc_lidar', 'gt_pre_bbox_sdc_global', 'gt_fut_bbox_sdc_global',
    'gt_pre_bbox_sdc_mask', 'gt_fut_bbox_sdc_mask', 'gt_pre_command_sdc',
)


class FrameHistory:
    def __init__(self):
        self.raw = deque(maxlen=4)
        self.frames = deque(maxlen=4)
        # Only pure parse methods are used; no file client is initialized.
        self.parser = NAVFormerClient.__new__(NAVFormerClient)

    def append(self, raw):
        if self.raw:
            previous = self.raw[-1]
            if (raw['scene_token'] != previous['scene_token'] or
                    raw['frame_idx'] != previous['frame_idx'] + 1 or
                    raw['timestamp'] <= previous['timestamp']):
                raise ValueError('History requires consecutive frames in one episode')
        prepared = self.parser._postprocess_frame_data(copy.deepcopy(raw))
        self.raw.append(prepared)
        sequence = {v['frame_idx']: copy.deepcopy(v) for v in self.raw}
        parsed = self.parser.parse_clip(sequence, list(sequence))
        current = parsed[prepared['frame_idx']]
        if current is None:
            raise ValueError('Invalid live history')
        # Original parser constructs zero, invalid future labels. They only
        # satisfy the existing inference signature and cannot be reward data.
        if np.any(current['gt_fut_bbox_sdc_mask']) or np.any(current['gt_fut_bbox_sdc_lidar']):
            raise ValueError('Future labels must be unavailable in a live observation')
        frame = {k: copy.deepcopy(current[k]) for k in FRAME_FIELDS}
        self.frames.append(frame)
        return copy.deepcopy(frame)


class CanonicalObserver:
    def __init__(self, sim, asset_root):
        self.sim = sim
        # Configure before the first observation snapshot; do not register GPU
        # managers in the physics engine or weaken SnapshotCodec validation.
        cfg = sim.engine.global_config
        cfg.asset_folder_path = str(asset_root)
        path = Path(__file__).resolve().parents[1]/'configs/common/renderer/mtgs.yaml'
        cfg.renderer = OmegaConf.load(path)
        self.render = RenderManager()
        self.render.reset()
        self.data = DataManager()
        self.history = FrameHistory()

    def observe(self):
        import torch
        before = self.sim.snapshot()
        # Renderer/preprocessing is an observation side effect. Isolate the
        # process-global Python RNG so it cannot alter future dynamics/spawns.
        python_rng = random.getstate()
        try:
            with torch.no_grad():
                rendered = self.render.get_observations(persist=False, cache=False)
            raw = self.data._get_current_frame_data(render_results=rendered, persist_images=False)
            frame = self.history.append(raw)
            images = {}
            for name, camera in raw['cams'].items():
                image = rendered['cameras'][name]['image']
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError('Expected rendered uint8 BGR camera')
                ok, encoded = cv2.imencode('.jpg', image)  # matches legacy imwrite default
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                images[name] = encoded.tobytes()
        finally:
            random.setstate(python_rng)
        after = self.sim.snapshot()
        if after.state_hash != before.state_hash:
            # Keep the strict gate, but expose the first changed dynamics
            # component so renderer/data-manager side effects are actionable.
            mismatch = self.sim.codec.mismatch(
                'Rendering/history changed canonical dynamics state',
                before, before, after, None)
            raise RuntimeError('Rendering/history changed canonical dynamics state: ' +
                               json.dumps(mismatch.details, sort_keys=True, default=str))
        self.sim.codec.audit_static()
        return frame, images, before.state_hash
