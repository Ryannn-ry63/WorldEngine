"""All-pilot64 refit at fixed LR/step; no held-out selection or extra search."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import cfpi_common as c
import grpo_selector_v3_cached_common as v3
import selector_cfpi_model as models
import selector_cfpi_deployment_common as d
from selector_cfpi_objectives import METHODS, exact_group, q_regression
from train_selector_cfpi import validate_incumbent_recompute


def train(run, method, seed, device):
    policy = d.learned_id(method, seed)
    settings = d.POLICIES[method]
    inputs = d.read(run / "deployment_inputs.json")
    gate = d.read(run / "phase_a_report.json")
    if gate.get("status") != "PASS" or gate.get("phase_b_authorized") is not True:
        raise RuntimeError("Refit requires a complete passing phase A report")
    files = inputs["artifacts"]
    cache = d.verify(files["cache"])
    original_gate = d.verified_read(files["pilot_gate"])
    if original_gate["cache_sha256"] != files["cache"]["sha256"] or not original_gate["training_authorized"]:
        raise RuntimeError("Wrong refit cache / pilot gate")
    rows = c.validate_cache(c.load_pickle(cache))
    if len(rows) != 64:
        raise RuntimeError("Refit consumes exactly the original pilot64")
    provenance = dict(method=method, learning_rate=settings["lr"], seed=seed, step=settings["steps"],
                      max_steps=settings["steps"], batch_size=16, weight_decay=1e-4,
                      training_scope="all_pilot64_no_model_selection", train_scene_count=64,
                      cache_sha256=files["cache"]["sha256"], inputs_sha256=c.sha256_file(run / "deployment_inputs.json"),
                      incumbent_sha256=c.CHECKPOINT_SHA256, code_sha=d.read(run / "run_contract.json")["code_sha"],
                      phase_a_report_sha256=c.sha256_file(run / "phase_a_report.json"))
    provenance["export_same_path_score_tolerance"] = c.ARRAY_TOLERANCE
    folder = run / "refit" / policy
    c.locked_json(folder / "contract.json", provenance)
    report_path = folder / "report.json"
    if report_path.exists():
        report = c.verified_json(report_path)
        if report["provenance"] != provenance:
            raise RuntimeError("Refit resume provenance changed")
        d.verify(report["selector"])
        return
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback for refit")
    incumbent, config, _ = models.load_incumbent(d.verify(files["scalar_manifest"]))
    incumbent.to(device)
    parity = validate_incumbent_recompute(incumbent, rows, device)
    definition = METHODS[method]
    labels = torch.as_tensor(np.stack([r[definition["label"]] for r in rows]), dtype=torch.float32)
    mode = settings["score_mode"]
    model = models.initialize(incumbent, mode, labels).to(device).train()
    del incumbent
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    resume = folder / "resume.pt"
    start, last_loss = 0, None
    if resume.exists():
        state = torch.load(resume, map_location=device)
        if state["provenance"] != provenance:
            raise RuntimeError("Refit optimizer resume provenance changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device=device)
        start, last_loss = state["step"], state["last_loss"]
    for step in range(start + 1, settings["steps"] + 1):
        ids = rng.choice(len(rows), size=16, replace=False).tolist()
        scores = models.score(model, rows, ids, device, mode)
        values = labels[ids].to(device)
        loss = (q_regression(scores, values) if mode == "direct_q" else
                exact_group(scores, values, definition["temperature"]))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite refit loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        v3.clip_grad_norm_cpu_(model.parameters(), 10.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step % 25 == 0 or step == settings["steps"]:
            temporary = resume.with_suffix(".tmp")
            torch.save(dict(provenance=provenance, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), last_loss=last_loss,
                            cuda_rng=(torch.cuda.get_rng_state(device=device) if device.startswith("cuda") else None)), temporary)
            temporary.replace(resume)
    model.eval()
    path = folder / f"step_{settings['steps']}.pt"
    models.save_selector(path, model, config, mode, provenance)
    reloaded, payload = models.load_selector(path)
    reloaded.to(device)
    if model.state_dict().keys() != reloaded.state_dict().keys() or any(
            not torch.equal(value, reloaded.state_dict()[key]) for key, value in model.state_dict().items()):
        raise RuntimeError("Refit exported parameter tensors changed")
    export_error = 0.0
    with torch.no_grad():
        for offset in range(0, 64, 16):
            ids = list(range(offset, offset + 16))
            before = models.score(model, rows, ids, device, mode)
            after = models.score(reloaded, rows, ids, device, payload["score_mode"])
            export_error = max(export_error, float((before - after).abs().max().cpu()))
            if (not bool(torch.isfinite(after).all()) or export_error > c.ARRAY_TOLERANCE
                    or not torch.equal(before.argmax(1), after.argmax(1))):
                raise RuntimeError("Refit export/reload score parity failed")
    c.atomic_json(report_path, dict(status="PASS", method=method, seed=seed, train_scene_count=64,
                  provenance=provenance, selector=d.artifact(path), incumbent_recompute_parity=parity,
                  last_training_loss=last_loss, score_mode=mode, inference_uses_reward_or_q=False,
                  export_score_max_abs=export_error, export_parameter_tensors_exact=True,
                  development_consumed=False, test_consumed=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--method", choices=d.POLICIES, required=True)
    p.add_argument("--seed", type=int, choices=d.SEEDS, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    train(args.run_root.resolve(), args.method, args.seed, args.device)
    print("PASS: fixed pilot64 refit")


if __name__ == "__main__":
    main()
