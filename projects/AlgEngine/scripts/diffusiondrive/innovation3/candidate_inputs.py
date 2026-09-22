"""Reward-free input boundary for existing frozen DiffusionDrive context export.

This does not certify that images were live; the observation producer must bind
this export to the current StepIdentity. No files, caches or evaluator scores
are consulted. Caller must supply the expected current sample token.
"""
import torch

FIELDS = {
    'candidate_features': ('candidate_features', (20, 256)),
    'candidate_trajectories_8': ('candidate_trajectories', (20, 8, 3)),
    'route_bev_features': ('route_bev_features', (20, 8, 256)),
    'status_tokens': ('status_token', (1, 256)),
    'ego_queries': ('ego_query', (1, 256)),
    'agents_queries': ('agents_query', (30, 256)),
}


def from_export(result, expected_sample_token, frozen_v3):
    if not expected_sample_token or result.get('token') != expected_sample_token:
        raise ValueError('Stale/missing generator sample token')
    source = result.get('diffusiondrive_rollout_context')
    required = set(FIELDS) | {'schema_version', 'reference_logits', 'current_logits', 'selected_indices'}
    if not isinstance(source, dict) or set(source) != required or source['schema_version'] != 1:
        raise ValueError('Unexpected generator context schema/fields')
    parameter = next(frozen_v3.parameters())
    if frozen_v3.training or any(p.requires_grad for p in frozen_v3.parameters()):
        raise ValueError('V3 inference reference must be eval and frozen')

    def tensor(key, shape):
        value = torch.as_tensor(source[key], device=parameter.device)
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError('Invalid generated field: ' + key)
        return value.detach().to(dtype=parameter.dtype).clone()[None]

    context = {name: tensor(key, shape) for key, (name, shape) in FIELDS.items()}
    base = tensor('reference_logits', (20,))  # original DiffusionDrive selector, not already-V3 logits
    exported = tensor('current_logits', (20,))
    selected_value = torch.as_tensor(source['selected_indices'])
    if selected_value.shape != () or selected_value.dtype not in (torch.int32, torch.int64):
        raise ValueError('Invalid frozen selector index dtype/shape')
    selected = int(selected_value)
    if not 0 <= selected < 20 or int(exported.argmax(-1)) != selected:
        raise ValueError('Exported frozen selector index/logits disagree')
    with torch.no_grad():
        expected = base + frozen_v3(**context)
    if not torch.allclose(expected, exported, atol=1e-5, rtol=1e-4):
        raise RuntimeError('Frozen V3 export logit parity failed; possible wrong checkpoint or double-added V3')
    return context, base, float((expected-exported).abs().max())
