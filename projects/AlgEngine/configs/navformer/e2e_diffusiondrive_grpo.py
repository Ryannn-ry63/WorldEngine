"""Generation-only GRPO fine-tuning for the WorldEngine DiffusionDrive port."""

import os
import pickle

_base_ = ["./e2e_diffusiondrive.py"]

WORLDENGINE_ROOT = os.getenv("WORLDENGINE_ROOT", os.path.abspath("."))
DEFAULT_BASE_CKPT = os.path.join(
    os.path.dirname(WORLDENGINE_ROOT),
    "Worldengine-Diffusion",
    "experiments/navformer/e2e_diffusiondrive/epoch_8.pth",
)
GRPO_REFERENCE_CKPT = os.getenv("GRPO_REFERENCE_CKPT", DEFAULT_BASE_CKPT)
GRPO_POLICY_CKPT = os.getenv("GRPO_POLICY_CKPT", GRPO_REFERENCE_CKPT)
GRPO_ROLLOUT_MANIFEST = os.getenv("GRPO_ROLLOUT_MANIFEST", "")
GRPO_REQUESTED_UPDATES = int(os.getenv("GRPO_MAX_UPDATES", "128"))
GRPO_MAX_UPDATES = GRPO_REQUESTED_UPDATES
if GRPO_ROLLOUT_MANIFEST and os.path.isfile(GRPO_ROLLOUT_MANIFEST):
    with open(GRPO_ROLLOUT_MANIFEST, "rb") as stream:
        _manifest = pickle.load(stream)
    GRPO_MAX_UPDATES = min(
        GRPO_REQUESTED_UPDATES, int(_manifest["num_entries"])
    )
    del stream, _manifest

if not os.path.isfile(GRPO_REFERENCE_CKPT):
    raise FileNotFoundError(
        "Set GRPO_REFERENCE_CKPT to the fixed DiffusionDrive checkpoint"
    )
if not os.path.isfile(GRPO_POLICY_CKPT):
    raise FileNotFoundError("Set GRPO_POLICY_CKPT to the rollout policy checkpoint")
if GRPO_MAX_UPDATES <= 0:
    raise ValueError("GRPO training requires at least one rollout entry")

model = dict(
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_bev_encoder=True,
    freeze_bn=True,
    planning_head=dict(
        _delete_=True,
        type="DiffusionGRPOPlanningHead",
        num_poses=8,
        d_model=256,
        d_ffn=1024,
        num_heads=8,
        dropout=0.0,
        num_bounding_boxes=30,
        num_query_decoder_layers=3,
        query_keyval_size=8,
        num_anchors=20,
        num_diff_decoder_layers=2,
        plan_anchor_path=os.path.join(
            WORLDENGINE_ROOT,
            "data/alg_engine/kmeans_navsim_traj_20.npy",
        ),
        reference_checkpoint=GRPO_REFERENCE_CKPT,
        score_mode="rollout",
        bev_h=200,
        bev_w=200,
        bev_range_x=51.2,
        bev_range_y=51.2,
        num_train_timesteps=1000,
        train_timestep_max=50,
        inference_steps=2,
        trunc_timesteps=8,
        use_nerf=True,
        roll_timesteps=(8, 0),
        scheduler_num_inference_steps=125,
        generation_ddim_eta=1.0,
        generation_final_std=0.05,
        generation_sigma_min=1e-4,
        grpo_clip_ratio=0.2,
        grpo_advantage_eps=1e-3,
        generation_kl_weight=0.1,
        freeze_non_diffusion=True,
    ),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type="NavSimOpenSceneE2EGRPORollout",
        rollout_manifest=GRPO_ROLLOUT_MANIFEST,
        ann_file=None,
        data_root="",
        test_mode=False,
    ),
)

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=1e-6,
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(_delete_=True, policy="fixed")
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=GRPO_MAX_UPDATES)
checkpoint_config = dict(interval=GRPO_MAX_UPDATES, max_keep_ckpts=2)
evaluation = dict(interval=GRPO_MAX_UPDATES)
load_from = GRPO_POLICY_CKPT
resume_from = None
find_unused_parameters = True
