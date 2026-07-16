"""Dataset for replaying versioned WorldEngine GRPO rollout traces."""

import os
import pickle

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS

from .navsim_openscene_nuplan import NavSimOpenSceneE2E


def _load_object(path):
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    with open(path, "rb") as stream:
        return pickle.load(stream)


@DATASETS.register_module()
class NavSimOpenSceneE2EGRPORollout(NavSimOpenSceneE2E):
    """Attach stored diffusion actions and matching PDM rewards to 4-frame inputs."""

    def __init__(self, rollout_manifest, *args, **kwargs):
        self.rollout_manifest = os.path.abspath(rollout_manifest)
        manifest = _load_object(self.rollout_manifest)
        entries = manifest["entries"] if isinstance(manifest, dict) else manifest
        if not entries:
            raise ValueError("GRPO rollout manifest is empty")
        if not kwargs.get("ann_file"):
            if not isinstance(manifest, dict) or not manifest.get("ann_file"):
                raise ValueError("manifest has no consolidated ann_file")
            kwargs["ann_file"] = manifest["ann_file"]
        versions = {entry["policy_version"] for entry in entries}
        if len(versions) != 1:
            raise ValueError("one GRPO dataset may contain exactly one policy_version")
        self.policy_version = next(iter(versions))
        self.grpo_entries = {}
        for entry in entries:
            token = entry["token"]
            if token in self.grpo_entries:
                raise ValueError("duplicate GRPO token in manifest: " + token)
            self.grpo_entries[token] = entry
        super().__init__(*args, **kwargs)
        self.index_map = [
            index
            for index, info in enumerate(self.data_infos)
            if info["token"] in self.grpo_entries
        ]
        if len(self.index_map) != len(self.grpo_entries):
            indexed = {self.data_infos[index]["token"] for index in self.index_map}
            missing = sorted(set(self.grpo_entries) - indexed)
            raise KeyError(
                "manifest tokens missing from annotation file: " + ", ".join(missing[:5])
            )

    def load_pdm_infos(self):
        self.pdm_dict = {}

    def get_pdm_score_info(self, input_dict, index=None, info=None):
        return self.get_zero_pdm(input_dict)

    def _load_grpo_sample(self, token):
        entry = self.grpo_entries[token]
        record = _load_object(entry["record_path"])
        reward = _load_object(entry["reward_path"])
        model_result = record.get("model_result", record)
        required = (
            "grpo_initial_sample",
            "grpo_transition_action",
            "grpo_final_action",
            "grpo_old_log_probs",
        )
        missing = [key for key in required if key not in model_result]
        if missing:
            raise KeyError("rollout record is missing " + ", ".join(missing))
        rewards = np.asarray(reward["score"], dtype=np.float32)
        if rewards.ndim != 1:
            raise ValueError("GRPO reward score must have shape [mode]")
        valid_mask = np.asarray(
            reward.get("valid_mask", np.isfinite(rewards)), dtype=np.bool_
        )
        if valid_mask.shape != rewards.shape:
            raise ValueError("reward valid_mask must match score shape")
        selected_index = int(
            model_result.get("chosen_ind", record.get("plan_idx", -1))
        )
        if selected_index < 0 or selected_index >= rewards.shape[0]:
            raise ValueError("selected GRPO candidate is out of range")
        return {
            "grpo_initial_sample": np.asarray(
                model_result["grpo_initial_sample"], dtype=np.float32
            ),
            "grpo_transition_action": np.asarray(
                model_result["grpo_transition_action"], dtype=np.float32
            ),
            "grpo_final_action": np.asarray(
                model_result["grpo_final_action"], dtype=np.float32
            ),
            "grpo_old_log_probs": np.asarray(
                model_result["grpo_old_log_probs"], dtype=np.float32
            ),
            "grpo_rewards": rewards,
            "grpo_valid_mask": valid_mask,
            "grpo_selected_index": np.asarray(selected_index, dtype=np.int64),
        }

    def prepare_test_data(self, index):
        data = super().prepare_test_data(index)
        if data is None:
            return None
        token = self.data_infos[index]["token"]
        for key, value in self._load_grpo_sample(token).items():
            tensor = torch.from_numpy(value)
            data[key] = DC(tensor, stack=True, cpu_only=False)
        return data
