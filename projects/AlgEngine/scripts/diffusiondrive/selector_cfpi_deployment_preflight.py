"""Resolve the opt-in deployment config and check immutable selector-bank identity."""
import argparse
from pathlib import Path

import cfpi_common as c
import selector_cfpi_deployment_common as d


def cached_routing(run, device):
    import numpy as np
    import torch
    from selector_cfpi_deployment_router import load_bank, selector_scores
    torch.set_num_threads(4)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for real cached-route parity check")
    inputs = d.read(run / "deployment_inputs.json")
    rows = c.validate_cache(c.load_pickle(d.verify(inputs["artifacts"]["cache"])))
    targets = d.verified_read(inputs["artifacts"]["targets"])["targets"]
    target_by_scene = {r["scene_id"]: r for r in targets}
    count, maximum, mismatches = 0, 0.0, []
    for method in d.POLICIES:
        for seed in d.SEEDS:
            policy = d.learned_id(method, seed)
            path = run / "collections" / f"phase_a_{policy}" / "routing.json"
            routing = d.read(path)
            bank = {k: load_bank(v).to(device) for k, v in routing["models"].items()}
            for row in rows:
                target = target_by_scene[row["scene_id"]]
                route, active = d.route_for(routing, target["origin_token"], target["decision_step"])
                if not active or route["fold"] != row["fold"]:
                    raise RuntimeError("Cached target scene/fold/activation drift")
                entry = routing["models"][route["model_key"]]
                if entry["provenance"]["fold"] != route["fold"]:
                    raise RuntimeError("Scene was routed to a model trained on its held-out fold")
                scores = selector_scores(bank[route["model_key"]], row, device, entry["score_mode"])
                actual = int(scores.argmax())
                expected = inputs["oof_predictions"][policy][row["scene_id"]]
                if actual != expected:
                    mismatches.append(dict(policy=policy, scene=row["scene_id"], actual=actual, expected=expected))
                count += 1
            del bank
    incumbent = load_bank(inputs["models"]["scalar_v3"]).to(device)
    for row in rows:
        scores = selector_scores(incumbent, row, device, "residual")
        maximum = max(maximum, float(np.max(np.abs(scores - np.asarray(row["v3_logits"])))))
        if int(scores.argmax()) != row["policy_index"]:
            mismatches.append(dict(policy="scalar_v3", scene=row["scene_id"]))
    if maximum > c.MODEL_RECOMPUTE_TOLERANCE or mismatches or count != 768:
        raise RuntimeError(f"Cached deployment routing failed: scalar_max={maximum}, mismatches={mismatches}")
    result = dict(status="PASS", target_action_checks=count, scalar_action_checks=64,
                  argmax_mismatch_count=0, scalar_recompute_max_abs=maximum, device=device,
                  inputs=d.artifact(run / "deployment_inputs.json"),
                  continuous_rollout_performed=False, trajectory_execution_checked=False)
    c.atomic_json(run / f"cached_routing_audit_{device}.json", result)
    print(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--cached-run", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    if args.cached_run:
        cached_routing(args.cached_run.resolve(), args.device)
        return
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    if (cfg.selector_rollout_contract.experiment != d.METHOD or
            cfg.model.planning_head.online_reward is not None or
            not cfg.model.planning_head.export_rollout_context or
            cfg.model.planning_head.candidate_noise_namespace != c.NOISE_NAMESPACE or
            cfg.selector_rollout_contract.expected_checkpoint_sha256 != c.CHECKPOINT_SHA256):
        raise RuntimeError("Resolved deployment config violates frozen-generator/no-reward contract")
    _ = cfg.pretty_text
    routing = d.verified_read(dict(path=cfg.cfpi_deployment_routing, sha256=cfg.cfpi_deployment_routing_sha256))
    from selector_cfpi_deployment_router import load_bank
    for model in routing["models"].values():
        load_bank(model)
    print("PASS: deployment config / selector provenance")


if __name__ == "__main__":
    main()
