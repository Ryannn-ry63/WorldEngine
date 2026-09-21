"""Small adapters, without changes to frozen projects/ source files."""
import copy
import torch
from torch import nn
import grpo_selector_v3_cached_common as v3

class OriginalSelector(nn.Module):
    def __init__(self, state):
        super().__init__()
        # Architecture is reconstructed from the actual saved original head.
        modules=[]
        for key,value in state.items():
            if not key.endswith('.weight'): continue
            index=int(key.split('.')[0])
            while len(modules)<index: modules.append(nn.ReLU())
            modules.append(nn.Linear(value.shape[1],value.shape[0]) if value.ndim==2 else nn.LayerNorm(value.shape[0]))
        self.head=nn.Sequential(*modules); self.head.load_state_dict(state,strict=True)
    def forward(self,candidate_features,**kwargs): return self.head(candidate_features).squeeze(-1)

def build(config, original_state=None):
    config=dict(config)
    if config.pop('paper_original_selector',False):
        if original_state is None: raise ValueError('Original scorer state is required')
        return OriginalSelector(original_state)
    unary=config.pop('paper_unary_matched',False)
    model=v3.SceneConditionedTrajectorySetSelector(**config)
    if unary:
        # Same parameters as full attention, but each token attends only to itself.
        full=dict(config,use_set_attention=True)
        model=v3.SceneConditionedTrajectorySetSelector(**full)
        model.set_encoder=DiagonalSetEncoder(model.set_encoder)
    return model

class DiagonalSetEncoder(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
    def forward(self,x):
        mask=~torch.eye(x.shape[1],dtype=torch.bool,device=x.device)
        return self.encoder(x,mask=mask)

def forward(model,inputs,base,config):
    out=model(**inputs)
    return out if config.get('paper_original_selector') else base+out
