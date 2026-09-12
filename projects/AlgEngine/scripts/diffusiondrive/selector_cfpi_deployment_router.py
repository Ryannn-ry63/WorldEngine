"""Small selector bank after ONE frozen planner forward. No simulator labels."""
from __future__ import annotations

import numpy as np
import torch

import cfpi_common as c
import selector_cfpi_model as models
import selector_cfpi_deployment_common as d


def load_bank(entry):
    path = d.verify(entry)
    if entry["kind"] in {"cfpi", "rare"}:
        if entry["kind"] == "rare":
            from selector_rare_model import load
            model, payload = load(path)
        else:
            model, payload = models.load_selector(path)
        if payload["score_mode"] != entry["score_mode"]:
            raise RuntimeError("Selector score semantics changed")
        provenance = payload["provenance"]
        for key, value in entry["provenance"].items():
            if provenance.get(key) != value:
                raise RuntimeError(f"Selector provenance mismatch: {key}")
    elif entry["kind"] == "legacy":
        payload = torch.load(path, map_location="cpu")
        model = models.v3.model_from_config(payload["scene_selector_config"])
        model.load_state_dict(payload["scene_selector_state"], strict=True)
    else:
        raise RuntimeError("Unsupported selector bank format")
    if getattr(model, "requires_frozen_track_states", False):
        raise RuntimeError("Deployment v1 does not export track-state selector inputs")
    return model.eval().requires_grad_(False)


@torch.no_grad()
def selector_scores(model, context, device, mode):
    inputs = models.visible_batch([context], [0], device)
    reference = torch.as_tensor(c.checked_array(context["reference_logits"], (20,), "reference logits"),
                                dtype=torch.float32, device=device)[None]
    if getattr(model, "requires_reference_logits", False):
        inputs["reference_logits"] = reference
    output = model(**inputs)
    if mode not in {"direct_q", "residual"}:
        raise RuntimeError("Invalid deployment score mode")
    scores = (output if mode == "direct_q" else output + reference)[0].cpu().numpy()
    return c.checked_array(scores, (20,), "deployment scores")


class DeploymentRouter:
    def __init__(self, manifest_path, manifest_sha256, expand_to_40, device="cuda"):
        self.manifest_sha256 = manifest_sha256
        self.manifest = d.verified_read(dict(path=str(manifest_path), sha256=manifest_sha256))
        if (self.manifest.get("method") != d.METHOD or self.manifest.get("status") != "PASS"
                or self.manifest.get("terminal_publication_decision") != d.TERMINAL_PUBLICATION
                or self.manifest.get("inference_uses_reward_or_q") is not False):
            raise RuntimeError("Invalid deployment routing manifest")
        # Construction is RNG-neutral, including legacy model initialization.
        with torch.random.fork_rng(devices=[]):
            self.bank = {key: load_bank(entry).to(device)
                         for key, entry in self.manifest["models"].items()}
        self.device, self.expand = device, expand_to_40

    @torch.no_grad()
    def apply(self, result, scene_prefix, decision):
        route, active = d.route_for(self.manifest, str(scene_prefix), int(decision), allow_terminal=True)
        context = result["diffusiondrive_rollout_context"]
        incumbent_logits = c.checked_array(context["current_logits"], (20,), "incumbent logits").copy()
        incumbent_index = int(np.argmax(incumbent_logits))
        if int(result["chosen_ind"]) != incumbent_index:
            raise RuntimeError("Unmodified planner action/argmax mismatch")
        model_key = route["model_key"] if active else None
        entry = self.manifest["models"].get(model_key)
        selected, recompute_error = incumbent_index, None
        if entry is not None:
            model = self.bank[model_key]
            scores = selector_scores(model, context, self.device, entry["score_mode"])
            selected = int(scores.argmax())
            if self.manifest["policy"] == "scalar_v3":
                recompute_error = float(np.max(np.abs(scores - incumbent_logits)))
                if recompute_error > c.MODEL_RECOMPUTE_TOLERANCE or selected != incumbent_index:
                    raise RuntimeError("Same-model bank/loaded planner recomputation parity failed")
            if selected != incumbent_index:
                trajectory = torch.as_tensor(context["candidate_trajectories_8"][selected:selected + 1],
                                             dtype=torch.float32, device=self.device)
                result["trajectory"] = self.expand(trajectory)[0].cpu().numpy()
                # These detector diagnostics described the original selected trajectory.
                result.pop("ade_4s", None)
                result.pop("fde_4s", None)
            result["chosen_ind"] = selected
            context["current_logits"] = scores
            context["selected_indices"] = np.asarray(selected, dtype=np.int64)
        elif model_key is not None:
            raise RuntimeError("Missing active selector bank entry")
        intervention = None
        if self.manifest.get('research_method') in ('selector_feedback_repair_v2', 'selector_decision_feedback_v1'):
            from selector_feedback_transport import apply_intervention
            selected, intervention = apply_intervention(
                result, route, int(decision), selected, self.expand, self.device)
        result["cfpi_deployment"] = dict(
            schema_version=1, routing_sha256=self.manifest_sha256,
            terminal_unexecuted=(int(decision) == d.TERMINAL_PUBLICATION),
            scene_id=route["scene_id"], decision_step=int(decision), fold=route["fold"],
            active=bool(active), model_key=model_key,
            selector_sha256=(entry["sha256"] if entry else self.manifest["incumbent_selector_sha256"]),
            score_mode=(entry["score_mode"] if entry else "residual"),
            incumbent_index=incumbent_index, incumbent_logits=incumbent_logits.tolist(),
            selected_index=selected, same_model_recompute_max_abs=recompute_error,
            generator_forward_count=1, inference_uses_reward_or_q=False)
        if intervention is not None:
            result['cfpi_deployment']['feedback_intervention'] = intervention
        return result
