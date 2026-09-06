"""Zero adapter on frozen V3 tokens; output is residual to ORIGINAL BASE logits."""
import copy
from pathlib import Path
import torch
import selector_cfpi_model as m
import selector_rare_common as r


class AnchoredSelector(torch.nn.Module):
    def __init__(self, incumbent):
        super().__init__()
        self.incumbent = copy.deepcopy(incumbent).eval().requires_grad_(False)
        self.adapter = torch.nn.Sequential(torch.nn.Linear(256,128),torch.nn.GELU(),torch.nn.Linear(128,1))
        torch.nn.init.zeros_(self.adapter[-1].weight)
        torch.nn.init.zeros_(self.adapter[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.incumbent.eval()
        return self

    def forward(self, **visible):
        with torch.no_grad():
            h = self.incumbent.encode_tokens(**visible)
            residual = self.incumbent.delta_head(h).squeeze(-1)
        return residual+self.adapter(h).squeeze(-1)


def initialize(incumbent, method):
    if method not in r.POLICIES:
        raise ValueError('Unregistered rare method')
    return copy.deepcopy(incumbent).requires_grad_(True) if method=='q_full' else AnchoredSelector(incumbent)


def teacher_kl(student, teacher):
    log_p = teacher.detach().log_softmax(-1)
    return (log_p.exp()*(log_p-student.log_softmax(-1))).sum(-1).mean()


def save(path, model, config, provenance):
    path = Path(path)
    p = dict(schema_version=1,method=r.METHOD,score_mode='residual',variant=provenance['method'],
             scene_selector_config=config,scene_selector_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
             provenance=provenance,inference_uses_reward_or_q=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(p,tmp)
    tmp.replace(path)


def load(path):
    p = torch.load(path,map_location='cpu')
    if (p.get('method')!=r.METHOD or p.get('schema_version')!=1 or p.get('score_mode')!='residual'
            or p.get('inference_uses_reward_or_q') is not False or p['provenance']['step']!=500
            or p.get('variant')!=p['provenance']['method']):
        raise RuntimeError('Invalid rare checkpoint; only step500 eligible')
    model = initialize(m.v3.model_from_config(p['scene_selector_config']),p['variant'])
    model.load_state_dict(p['scene_selector_state'],strict=True)
    return model.eval(),p
