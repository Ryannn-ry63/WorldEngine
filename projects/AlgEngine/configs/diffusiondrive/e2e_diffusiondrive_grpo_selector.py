"""Formal selector-only GRPO post-training for DiffusionDrive.

This is the single canonical training config for the paper experiment. The
official epoch-100 DiffusionDrive checkpoint initializes both pi_current and
the immutable pi_ref == pi_old. The perception stack, DiT denoiser, trajectory
regression branches, anchors, and every selector except the final 20-way score
branch stay frozen. NAVSIM-v1 PDM rewards are computed online for the exact 20
trajectories produced in the same forward pass.

There is deliberately no imitation, entropy, ranking, fixed-vocabulary, or
generator loss. Formal runs use eight complete epochs; the old
iteration-limited smoke/pilot runs are not paper results.
"""

import os


_base_ = ["./e2e_diffusiondrive.py"]

custom_imports = dict(
    imports=[
        "mmdet3d_plugin",
        "mmdet3d_plugin.navformer.dense_heads.diffusion_grpo_online_planning_head",
    ]
)

WORLDENGINE_ROOT = os.getenv("WORLDENGINE_ROOT", os.path.abspath("."))
DEFAULT_DIFFUSIONDRIVE_CHECKPOINT = (
    "/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/"
    "repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth"
)
DEFAULT_DIFFUSIONDRIVE_SHA256 = (
    "1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
)
BASELINE_CHECKPOINT = os.getenv(
    "DIFFUSIONDRIVE_SELECTOR_INIT_CHECKPOINT", DEFAULT_DIFFUSIONDRIVE_CHECKPOINT
)
BASELINE_SHA256 = os.getenv(
    "DIFFUSIONDRIVE_SELECTOR_INIT_SHA256", DEFAULT_DIFFUSIONDRIVE_SHA256
)

NAVSIM_EXP_ROOT = os.getenv(
    "NAVSIM_EXP_ROOT", os.path.join(os.path.dirname(WORLDENGINE_ROOT), "exp")
)
METRIC_CACHE_TRAIN = os.getenv(
    "NAVSIM_METRIC_CACHE_PATH_TRAIN",
    os.getenv(
        "NAVSIM_METRIC_CACHE_PATH",
        os.path.join(NAVSIM_EXP_ROOT, "metric_cache_trainval"),
    ),
)
METRIC_CACHE_EVAL = os.getenv(
    "NAVSIM_METRIC_CACHE_PATH_EVAL",
    os.path.join(NAVSIM_EXP_ROOT, "metric_cache_navtest"),
)
DEFAULT_SPLIT_ROOT = os.path.join(
    WORLDENGINE_ROOT,
    "experiments/grpo_sources/diffusiondrive_selector_navtrain_split_v2",
)
NAVTRAIN_FILTER = os.getenv(
    "DIFFUSIONDRIVE_GRPO_NAV_FILTER_PATH",
    os.path.join(DEFAULT_SPLIT_ROOT, "navtrain_grpo_train.yaml"),
)

load_from = BASELINE_CHECKPOINT

model = dict(
    process_perception=False,
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_bn=True,
    freeze_bev_encoder=True,
    planning_head=dict(
        type="DiffusionGRPOOnlineSelectorPlanningHead",
        selector_layer=-1,
        reward_key="score",
        policy_loss_weight=1.0,
        clip_epsilon=float(os.getenv("DIFFUSIONDRIVE_GRPO_CLIP_EPS", "0.2")),
        kl_weight=float(os.getenv("DIFFUSIONDRIVE_GRPO_KL_WEIGHT", "0.001")),
        advantage_epsilon=1e-6,
        minimum_reward_std=1e-6,
        reference_checkpoint=BASELINE_CHECKPOINT,
        candidate_noise_namespace=os.getenv(
            "DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE"
        ),
        reference_checkpoint_sha256=BASELINE_SHA256,
        online_reward=dict(
            metric_cache_path=METRIC_CACHE_TRAIN,
            num_candidates=20,
            metric_cache_lru_size=int(
                os.getenv("DIFFUSIONDRIVE_PDM_CACHE_LRU_SIZE", "128")
            ),
            fail_on_missing_cache=True,
            use_cuda_simulator=True,
        ),
    ),
)

custom_hooks = [
    dict(type="DiffusionGRPOOnlineReferenceSelectorHook", priority="VERY_HIGH")
]

optimizer = dict(
    _delete_=True,
    type="AdamW",
    constructor="DiffusionGRPOOnlineSelectorOptimizerConstructor",
    lr=float(os.getenv("DIFFUSIONDRIVE_GRPO_LR", "3e-6")),
    weight_decay=1e-4,
)
lr_config = dict(_delete_=True, policy="Fixed")

# Online rewards correspond to exact candidates generated after image loading,
# therefore fixed 8192-vocabulary PDM arrays are absent.
img_root_train = os.path.join(
    WORLDENGINE_ROOT, "data/raw/openscene-v1.1/sensor_blobs/trainval"
)
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False,
)
class_names = [
    "vehicle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
    "czone_sign",
    "generic_object",
]
online_train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFilesWithDownsample",
        to_float32=True,
        img_root=img_root_train,
        downsample_factor=2,
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "img",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
            "sdc_planning",
            "sdc_planning_mask",
            "command",
            "sdc_planning_world",
            "sdc_planning_past",
            "sdc_planning_mask_past",
            "gt_pre_command_sdc",
            "sdc_status",
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=int(os.getenv("DIFFUSIONDRIVE_GRPO_WORKERS_PER_GPU", "2")),
    train=dict(
        pipeline=online_train_pipeline,
        nav_filter_path=NAVTRAIN_FILTER,
        metric_cache_path=METRIC_CACHE_TRAIN,
        diffusiondrive_data_mode=True,
        online_candidate_reward=True,
    ),
    # These remain the official base-config definitions but are never used for
    # model selection. Calibration is a separate navtrain-held-out job.
    val=dict(metric_cache_path=METRIC_CACHE_EVAL),
    test=dict(metric_cache_path=METRIC_CACHE_EVAL),
)

total_epochs = 8
runner = dict(_delete_=True, type="EpochBasedRunner", max_epochs=total_epochs)
# Formal launchers always pass --no-validate. Keep the interval beyond the run
# as a second guard against accidental navtest model selection.
evaluation = dict(_delete_=True, interval=total_epochs + 1)
checkpoint_config = dict(
    _delete_=True,
    by_epoch=True,
    interval=1,
    max_keep_ckpts=total_epochs,
)
find_unused_parameters = True

selector_reward_contract = dict(
    source="online_navsim_v1_pdm",
    num_dynamic_candidates=20,
    candidate_shape="(B, 20, 8, 3)",
    reward_shape="(B, 20)",
    pi_old_equals_pi_ref=True,
    generator_frozen=True,
    trainable_selector_tensors=10,
    imitation_loss=False,
    entropy_loss=False,
    ranking_loss=False,
    fixed_vocabulary_size=None,
    formal_epochs=8,
    model_selection_split="navtrain_scene_disjoint_calibration",
)
