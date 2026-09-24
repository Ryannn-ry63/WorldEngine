#!/usr/bin/env python3
"""Export a disposable online-finetuned V3 selector for normal deployment.

The export contains one SceneConditionedTrajectorySetSelector state.  The
frozen offline reference is intentionally not added as a second residual at
deployment time; the exported selector already includes the complete online
score ``b + r_theta_online`` through the normal selector interface.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from grpo_selector_v3_cached_common import SceneConditionedTrajectorySetSelector
from .learner import V3_INITIALIZED_SELECTOR_FINETUNE, OnlineV3Learner
from .paths import checked_path, sha256_file


state_fingerprint = OnlineV3Learner.state_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='disposable online-probe .online.pt checkpoint')
    parser.add_argument('--offline-selector', type=Path, required=True,
                        help='registered offline V3 selector payload')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = checked_path(args.checkpoint)
    offline = checked_path(args.offline_selector)
    output = checked_path(args.output.absolute(), must_exist=False)
    manifest = checked_path(args.manifest.absolute(), must_exist=False)
    if output == manifest:
        raise ValueError('Export and manifest must be separate paths')
    code = Path(__file__).resolve().parents[5]
    if any(p == code or code in p.parents for p in (output, manifest)):
        raise ValueError('Export artifacts must be outside code')
    if output.exists() or manifest.exists():
        raise FileExistsError('Use fresh export and manifest paths')
    payload = torch.load(checkpoint, map_location='cpu')
    if payload.get('kind') != 'DISPOSABLE_STRICT_ONLINE_PROBE_NOT_FORMAL_TRAINING':
        raise ValueError('Unexpected online checkpoint kind')
    learner_state = payload.get('learner')
    if not isinstance(learner_state, dict) or learner_state.get('schema_version') != 3:
        raise ValueError('Export requires schema3 initialized-selector checkpoint')
    if learner_state.get('parameterization') != V3_INITIALIZED_SELECTOR_FINETUNE:
        raise ValueError('Export refuses legacy residual parameterization')
    offline_payload = torch.load(offline, map_location='cpu')
    if offline_payload.get('schema_version') != 3 or offline_payload.get('method') != 'scene_conditioned_exact_group_grpo':
        raise ValueError('Unexpected offline V3 selector payload')
    offline_state = offline_payload.get('scene_selector_state')
    if not isinstance(offline_state, dict) or not offline_state:
        raise ValueError('Offline V3 selector state is empty')
    offline_sha = sha256_file(offline)
    provenance = learner_state.get('initialization', {})
    source = provenance.get('source', {})
    if source.get('sha256') != offline_sha:
        raise ValueError('Online checkpoint does not originate from this offline selector SHA256')
    if provenance.get('selector_init_sha256') != state_fingerprint(offline_state):
        raise ValueError('Online checkpoint/offline selector fingerprint mismatch')
    if state_fingerprint(learner_state.get('reference', {})) != state_fingerprint(offline_state):
        raise ValueError('Frozen online reference differs from offline selector')
    model = SceneConditionedTrajectorySetSelector(**offline_payload['scene_selector_config']).eval()
    model.load_state_dict(offline_state, strict=True)
    validated = OnlineV3Learner(model, parameterization=V3_INITIALIZED_SELECTOR_FINETUNE,
        learning_rate=learner_state['learning_rate'], kl_weight=learner_state['kl_weight'],
        initialization_source=dict(kind='offline_selector_file', path=str(offline),
            sha256=offline_sha, payload_schema=3, method=offline_payload['method'],
            scene_selector_config=offline_payload['scene_selector_config']))
    # Export validates source/config/reference without restoring a CUDA RNG on
    # a CPU exporter; no optimizer or stochastic action is used for deployment.
    validated._validate_schema3(learner_state)
    model.load_state_dict(learner_state['selector'], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Non-finite exported selector')
    selector_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    exported = dict(
        schema_version=3,
        method='scene_conditioned_exact_group_grpo',
        ablation=offline_payload.get('ablation', 'full'),
        temperature=1.0,
        learning_rate=learner_state['learning_rate'],
        kl_weight=learner_state['kl_weight'],
        train_seed=None, epoch=None,
        scene_selector_config=copy.deepcopy(offline_payload['scene_selector_config']),
        scene_selector_state=selector_state,
        online_export=dict(
            method='scene_conditioned_causal_h1_online_finetune',
            parameterization=V3_INITIALIZED_SELECTOR_FINETUNE,
            checkpoint_sha256=sha256_file(checkpoint),
            offline_selector_sha256=offline_sha,
            initialization=copy.deepcopy(provenance),
            selector_state_sha256=state_fingerprint(selector_state),
            policy_version=learner_state.get('version'),
            attempts=learner_state.get('attempts'),
            disposable=True,
            formal_ready=False,
            used_for_formal_training=False,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save(exported, stream)
    report = dict(
        schema_version=1,
        status='PASS_DISPOSABLE_ONLINE_SELECTOR_EXPORT',
        output=str(output), output_sha256=sha256_file(output),
        checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        offline_selector=str(offline), offline_selector_sha256=offline_sha,
        parameterization=V3_INITIALIZED_SELECTOR_FINETUNE,
        selector_state_sha256=state_fingerprint(selector_state),
        policy_version=learner_state.get('version'), attempts=learner_state.get('attempts'),
        architecture=copy.deepcopy(offline_payload['scene_selector_config']),
        deploy_contract='SceneConditionedTrajectorySetSelector_single_selector_state',
        frozen_reference_not_exported=True,
        avoids_legacy_residual_double_add=True,
        disposable=True, formal_ready=False, used_for_formal_training=False,
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
