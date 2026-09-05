import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/diffusiondrive"


def load(name, filename, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest():
    return {
        "status": "PASS",
        "method": "cpv_e2_existing_navtrain_common_coverage_v1",
        "existing_dataset_only": True,
        "new_annotations": False,
        "training_labels_changed": False,
        "nesting": {"D1_is_subset_of_D2": True},
        "arms": {
            arm: {
                "unique_tokens": unique,
                "rows": unique,
                "duplicate_rows": 0,
                "overlap_with_rare": 0,
                "heldout_log_overlap": 0,
            }
            for arm, unique in {
                "D1_diverse_matched": 5267,
                "D2_diverse_20k": 20000,
            }.items()
        },
    }


def e2_arm(contract, arm, manifest_sha):
    row = dict(contract.BASE_TRAINING_CONTRACT)
    row.update(
        method=f"cpv_e2_{arm.lower()}_source_sign_pair_regret",
        sampling_mode="cpv_decision_source_sign",
        common_arm=arm,
        common_unique_tokens=contract.ARM_UNIQUE[arm],
        common_data_manifest_sha256=manifest_sha,
        common_caches={
            str(seed): {"path": f"seed{seed}.pt", "sha256": f"sha{seed}"}
            for seed in range(3)
        },
    )
    return row


def add_metrics(row, equal, common, scores, gains):
    row.update(
        equal_weight_selected_reward=equal,
        strata={
            "common": {
                "current_reward": common,
                "reference_reward": 0.927,
                "degraded_fraction": 0.05,
                "proposal_benefit_roc_auc": 1.0,
                "proposal_ranking_records": {"evidence": scores, "gain": gains},
            },
            "real_rare": {
                "current_reward": 0.730,
                "proposal_benefit_roc_auc": 0.75,
            },
            "synthetic": {
                "current_reward": 0.800,
                "proposal_benefit_roc_auc": 0.75,
            },
        },
    )
    return row


def write_evaluation(path, row):
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "split": "development",
                "certification_consumed": False,
                "checkpoints": [row],
            }
        )
    )


def test_gate_integrates_identity_bootstrap_and_preregistered_decision(
    tmp_path, monkeypatch
):
    contract = load("cpv_e2_integration_contract", "cpv_e2_gate_contract.py", monkeypatch)
    gate = load(
        "cpv_e2_integration_gate",
        "select_cpv_e2_offline_development.py",
        monkeypatch,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest()))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    gains = [-0.2] * 20 + [0.1] * 20
    perfect = [-1.0] * 20 + [1.0] * 20

    d0 = dict(contract.BASE_TRAINING_CONTRACT)
    d0.update(method="cpv_v1_source_sign_pair_regret", common_arm=None)
    add_metrics(d0, 0.790, 0.907, [0.0] * 40, gains)
    d1 = add_metrics(
        e2_arm(contract, "D1_diverse_matched", manifest_sha),
        0.813,
        0.923,
        perfect,
        gains,
    )
    d2 = add_metrics(
        e2_arm(contract, "D2_diverse_20k", manifest_sha),
        0.811,
        0.922,
        perfect,
        gains,
    )
    proposal = {
        "equal_weight_selected_reward": 0.807,
        "strata": {
            "common": {"current_reward": 0.891},
            "real_rare": {"current_reward": 0.739},
            "synthetic": {"current_reward": 0.804},
        },
    }
    rows = {"p32": proposal, "p48": proposal, "d0": d0, "d1": d1, "d2": d2}
    paths = {}
    for name, row in rows.items():
        paths[name] = tmp_path / f"{name}.json"
        write_evaluation(paths[name], row)

    output = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPTS / "select_cpv_e2_offline_development.py"),
            "--proposal-32",
            str(paths["p32"]),
            "--proposal-48",
            str(paths["p48"]),
            "--d0",
            str(paths["d0"]),
            "--d1",
            str(paths["d1"]),
            "--d2",
            str(paths["d2"]),
            "--data-manifest",
            str(manifest_path),
            "--bootstrap-repetitions",
            "100",
            "--output",
            str(output),
        ],
    )
    gate.main()
    report = json.loads(output.read_text())
    assert report["decision"] == "D1_GATE_PASS"
    assert report["selected_arm"] == "D1"
    assert report["certification_consumed"] is False
