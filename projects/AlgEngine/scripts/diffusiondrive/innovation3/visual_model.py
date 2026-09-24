"""Single resident, strictly frozen epoch100 + trained V3 for visual probes."""
from pathlib import Path
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model
from mmcv.utils import import_modules_from_strings
from mmdet3d.models import build_model
from .paths import checked_path, sha256_file


def configuration(cfg, seed):
    code = Path(__file__).resolve().parents[5]
    path = code/'projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py'
    config = Config.fromfile(str(path))
    config.model.planning_head.online_reward = None
    config.model.planning_head.export_rollout_context = True
    config.model.planning_head.candidate_noise_namespace = 'innovation3_visual_probe_seed' + str(seed)
    config.model.planning_head.reference_checkpoint = cfg['baseline']
    config.model.planning_head.reference_checkpoint_sha256 = cfg['expected_sha256']['baseline']
    config.model.pretrained = None
    config.model.train_cfg = None
    if hasattr(config.model, 'img_backbone'):
        config.model.img_backbone.norm_cfg = dict(type='BN', requires_grad=True)
    config.data.test.type = 'NavSimOpenSceneE2EClosedLoop'
    config.data.test.metric_cache_path = None
    config.data.test.test_mode = True
    config.data.test.pipeline[0].img_root = ''
    import_modules_from_strings(**config.custom_imports)
    return config


def load_frozen(config, cfg, device_id=None):
    for key in ('baseline', 'selector_state', 'anchors'):
        checked_path(cfg[key])
        if sha256_file(cfg[key]) != cfg['expected_sha256'][key]:
            raise ValueError('Registered checkpoint/input SHA mismatch: ' + key)
    payload = torch.load(cfg['selector_state'], map_location='cpu')
    if payload.get('schema_version') != 3 or payload.get('method') != 'scene_conditioned_exact_group_grpo':
        raise ValueError('Unexpected V3 checkpoint contract')
    config.model.planning_head.scene_selector = payload['scene_selector_config']
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    checkpoint = torch.load(cfg['baseline'], map_location='cpu')
    source = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    source = {k.removeprefix('module.'): v for k, v in source.items()}
    if any(k.startswith('planning_head.scene_selector.') for k in source):
        raise ValueError('Expected immutable original baseline without V3')
    result = model.load_state_dict(source, strict=False)
    allowed = ('planning_head.reference_selector.', 'planning_head.scene_selector.')
    missing = [k for k in result.missing_keys if not k.startswith(allowed)]
    if missing or result.unexpected_keys:
        raise ValueError('Unreviewed generator checkpoint keys: ' + str((missing, result.unexpected_keys)))
    model.planning_head.scene_selector.load_state_dict(payload['scene_selector_state'], strict=True)
    model.planning_head.initialize_reference_selector()
    if config.get('fp16') is not None:
        wrap_fp16_model(model)
    model.eval().requires_grad_(False)
    if not model.skip_tracking:
        raise ValueError('Visual bridge currently supports the configured frozen skip_tracking path only')
    # ``device_ids=[0]`` is only correct when the process exposes one GPU.
    # Real DDP workers keep the full explicit CUDA_VISIBLE_DEVICES list and
    # select their local rank before entering NCCL.  Use that selected device
    # both for the model and for MMDataParallel so ranks cannot alias GPU 0.
    if device_id is None:
        device_id = torch.cuda.current_device()
    device_id = int(device_id)
    model = MMDataParallel(model.cuda(device_id), device_ids=[device_id])
    return model


def reset_temporal_state(model):
    """Clear per-scene inference state while retaining frozen parameters."""
    module = getattr(model, 'module', model)
    module.prev_frame_info = dict(prev_bev=None, scene_token=None, prev_pos=0, prev_angle=0)
    for name in ('prev_bev', 'scene_token', 'timestamp', 'test_track_instances',
                 'l2g_r_mat', 'l2g_t'):
        if hasattr(module, name):
            setattr(module, name, None)
