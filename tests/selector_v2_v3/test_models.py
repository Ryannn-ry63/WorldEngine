from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'projects/AlgEngine/scripts/diffusiondrive'))
import grpo_selector_v3_cached_common as v3
import grpo_selector_v2_rare_common as v2
import selector_ablation_models as controls


def inputs():
    torch.manual_seed(3)
    return dict(candidate_features=torch.randn(2,20,16),candidate_trajectories=torch.randn(2,20,8,3),
                route_bev_features=torch.randn(2,20,8,16),status_token=torch.randn(2,1,16),
                ego_query=torch.randn(2,1,16),agents_query=torch.randn(2,3,16))


def configuration():
    return dict(feature_dim=16,model_dim=16,route_bev_dim=16,context_dim=16,
                geometry_hidden_dim=8,num_heads=4,feedforward_dim=32,num_set_layers=1)


def test_zero_residual_preserves_original_ranking_and_trains():
    torch.set_num_threads(1)
    model=v3.SceneConditionedTrajectorySetSelector(**configuration())
    x=inputs();z=model(**x);assert torch.equal(z,torch.zeros_like(z))
    (-z[:,0].mean()).backward()
    assert model.delta_head[-1].weight.grad.abs().sum()>0


def test_unary_matched_retains_parameter_count():
    config=configuration();full=controls.build(config)
    unary=controls.build(dict(config,paper_unary_matched=True))
    assert sum(p.numel() for p in full.parameters())==sum(p.numel() for p in unary.parameters())
    assert unary(**inputs()).shape==(2,20)


def test_original_adapter_matches_v2():
    original=v2.Selector().eval();adapter=controls.build(dict(paper_original_selector=True),original.state_dict()).eval()
    x=torch.randn(2,20,256)
    assert torch.equal(original(x).squeeze(-1),adapter(candidate_features=x))


def test_v2_delta_anchor_is_exact():
    original=v2.Selector();cache=dict(baseline_selector_state=original.state_dict())
    m=v2.model_from_cache(cache,torch.device('cpu'));features=torch.randn(2,20,256)
    assert torch.equal(v2.delta_logits(m,features),torch.zeros(2,20))
