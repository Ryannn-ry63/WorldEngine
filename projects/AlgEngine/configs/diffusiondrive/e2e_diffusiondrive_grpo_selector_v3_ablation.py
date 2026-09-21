"""Structure-only V3 controls. Training budget/data must be matched externally."""
import os
_base_ = ['./e2e_diffusiondrive_grpo_selector_v3.py']
_arm = os.environ['SELECTOR_ABLATION']
_flags = {
    'full': {},
    'feature_only': dict(use_trajectory_geometry=False, use_route_bev=False, use_scene_context=False, use_set_attention=False),
    'no_geometry': dict(use_trajectory_geometry=False),
    'no_route_bev': dict(use_route_bev=False),
    'no_scene': dict(use_scene_context=False),
    'no_set': dict(use_set_attention=False),
}
if _arm not in _flags:
    raise ValueError('Use the explicit original/unary adapter for that control: ' + _arm)
model = dict(planning_head=dict(scene_selector=_flags[_arm]))
