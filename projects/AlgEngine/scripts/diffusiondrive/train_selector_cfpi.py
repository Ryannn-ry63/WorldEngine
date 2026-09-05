"""One fixed CFPI CV job; never accepts development/test or legacy V4 caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import cfpi_common as common
import grpo_selector_v3_cached_common as v3
import selector_cfpi_model as models
from selector_cfpi_objectives import (
    METHODS, classification_weights, exact_group, q_regression, return_weighted_classification,
)

CHECKPOINT_STEPS = (50, 150, 500)
LEARNING_RATES = (3e-5, 1e-4)
SEEDS = (0, 1, 2)


def job_name(method, lr, fold, seed):
    return f"{method}_lr{lr:g}_fold{fold}_seed{seed}"


def train(args):
    cache_audit = common.verified_json(args.cache_audit, "cache_file")
    if Path(cache_audit["cache_file"]).resolve() != args.cache.resolve():
        raise RuntimeError("Cache/audit path mismatch")
    gate = common.verified_json(args.pilot_gate)
    cache_sha = common.sha256_file(args.cache)
    if gate.get("training_authorized") is not True or gate.get("cache_sha256") != cache_sha:
        raise RuntimeError("Pilot gate does not authorize this cache")
    rows = common.validate_cache(common.load_pickle(args.cache))
    if args.lr not in LEARNING_RATES or args.seed not in SEEDS or args.fold not in range(4):
        raise ValueError("Job outside frozen CV search")
    train_ids = [i for i, row in enumerate(rows) if row["fold"] != args.fold]
    test_ids = [i for i, row in enumerate(rows) if row["fold"] == args.fold]
    if not train_ids or not test_ids or {rows[i]["origin_log"] for i in train_ids} & {
            rows[i]["origin_log"] for i in test_ids}:
        raise RuntimeError("Invalid origin-log folds")
    method = METHODS[args.method]
    mode = "direct_q" if method["objective"] == "mse" else "residual"
    provenance = dict(cache_sha256=cache_sha, method=args.method, learning_rate=args.lr,
                      fold=args.fold, seed=args.seed, batch_size=16, max_steps=500,
                      pilot_gate_sha256=common.sha256_file(args.pilot_gate),
                      implementation={p.name: common.sha256_file(p) for p in [
                          Path(__file__), Path(models.__file__),
                          Path(__file__).with_name("selector_cfpi_objectives.py"),
                          Path(v3.SELECTOR_MODULE.__file__)]})
    job_dir = args.output_root / job_name(args.method, args.lr, args.fold, args.seed)
    contract_path = job_dir / "contract.json"
    common.locked_json(contract_path, provenance)
    report_path = job_dir / "report.json"
    if report_path.exists():
        report = common.verified_json(report_path)
        if report.get("provenance") != provenance:
            raise RuntimeError("Finished training report provenance drifted")
        for row in report["checkpoints"]:
            if common.sha256_file(row["checkpoint"]) != row["checkpoint_sha256"]:
                raise RuntimeError("Finished training checkpoint drifted")
        return report
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently train on CPU")
    incumbent, config, manifest = models.load_incumbent(args.checkpoint_manifest)
    incumbent.to(args.device)
    with torch.no_grad():
        for offset in range(0, len(rows), 16):
            ids = list(range(offset, min(offset + 16, len(rows))))
            scores = models.score(incumbent, rows, ids, args.device, "residual").cpu().numpy()
            expected = np.stack([rows[i]["v3_logits"] for i in ids])
            if (np.max(np.abs(scores - expected)) > 1e-5
                    or not np.array_equal(scores.argmax(1), expected.argmax(1))):
                raise RuntimeError("Cached V3 score/argmax parity failed")
    labels = torch.as_tensor(np.stack([r[method["label"]] for r in rows]), dtype=torch.float32)
    incumbents = torch.tensor([r["policy_index"] for r in rows], dtype=torch.long)
    weights = classification_weights(labels[train_ids], incumbents[train_ids], method["delta"])
    normalizer = max(float(weights.sum(1).mean()), 1e-8)
    model = models.initialize(incumbent, mode, labels[train_ids]).to(args.device).train()
    del incumbent
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = np.random.default_rng(args.seed)
    state_path = job_dir / "resume.pt"
    start, checkpoints = 0, []
    if state_path.exists():
        state = torch.load(state_path, map_location=args.device)
        if state["provenance"] != provenance:
            raise RuntimeError("Training resume provenance drift")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device=args.device)
        start, checkpoints = state["step"], state["checkpoints"]
        for checkpoint in checkpoints:
            if common.sha256_file(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]:
                raise RuntimeError("Resume checkpoint drifted")
    for step in range(start + 1, 501):
        ids = generator.choice(train_ids, size=16, replace=len(train_ids) < 16).tolist()
        values = labels[ids].to(args.device)
        scores = models.score(model, rows, ids, args.device, mode)
        if method["objective"] == "grpo":
            loss = exact_group(scores, values, method["temperature"])
        elif method["objective"] == "mse":
            loss = q_regression(scores, values)
        else:
            loss = return_weighted_classification(
                scores, values, incumbents[ids].to(args.device),
                delta=method["delta"], normalizer=normalizer)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite CFPI loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        v3.clip_grad_norm_cpu_(model.parameters(), 10.0)
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            model.eval()
            with torch.no_grad():
                prediction = models.score(model, rows, test_ids, args.device, mode).argmax(1).cpu().tolist()
            checkpoint = job_dir / f"step_{step}.pt"
            sha = models.save_selector(checkpoint, model, config, mode, dict(provenance, step=step,
                                       normalizer=normalizer, incumbent_sha256=manifest["checkpoint_sha256"]))
            reloaded, exported = models.load_selector(checkpoint)
            reloaded.to(args.device)
            with torch.no_grad():
                restored = models.score(reloaded, rows, test_ids, args.device, exported["score_mode"]).argmax(1).cpu().tolist()
            if restored != prediction:
                raise RuntimeError("Export selection parity failed")
            del reloaded
            checkpoints.append(dict(step=step, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha,
                                    predictions={rows[i]["scene_id"]: p for i, p in zip(test_ids, prediction)},
                                    last_training_loss=float(loss.detach().cpu())))
            model.train()
        if step % 25 == 0:
            temporary = state_path.with_suffix(".tmp")
            torch.save(dict(provenance=provenance, step=step, model=model.state_dict(),
                            optimizer=optimizer.state_dict(), rng=generator.bit_generator.state,
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=(torch.cuda.get_rng_state(device=args.device)
                                      if args.device.startswith("cuda") else None),
                            checkpoints=checkpoints), temporary)
            temporary.replace(state_path)
    report = dict(status="PASS", method=args.method, learning_rate=args.lr,
                  fold=args.fold, seed=args.seed, checkpoints=checkpoints,
                  train_scene_count=len(train_ids), heldout_scene_count=len(test_ids),
                  training_data_consumed=True, development_consumed=False, test_consumed=False,
                  inference_uses_reward_or_q=False, score_mode=mode, provenance=provenance)
    common.atomic_json(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "cache-audit", "pilot-gate", "checkpoint-manifest", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    train(args)
    print(json.dumps(dict(status="PASS", job=job_name(args.method, args.lr, args.fold, args.seed))))


if __name__ == "__main__":
    main()
