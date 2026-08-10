#!/usr/bin/env python3
"""Finalize three paired DiffusionDrive best-of-20 oracle audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from grpo_selector_oracle_common import (
    COMPONENT_NAMES,
    SELECTION_NAMES,
    classify_oracle_gain,
    load_records,
    paired_bootstrap_ci,
    read_official_csv,
    summarize_official_rows,
    summarize_records,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", type=Path, action="append", required=True)
    parser.add_argument("--failures-filter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--formal-reference-summary", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--formal-current-summary", type=Path, action="append", default=[]
    )
    parser.add_argument("--expected-num-tokens", type=int, default=12146)
    parser.add_argument("--expected-num-failures", type=int, default=288)
    parser.add_argument("--official-atol", type=float, default=1e-5)
    parser.add_argument("--formal-score-atol", type=float, default=1e-8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--selector-limited-threshold", type=float, default=0.005)
    parser.add_argument("--generator-limited-threshold", type=float, default=0.002)
    return parser.parse_args()


def load_filter_tokens(path: Path):
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("tokens"), list):
        raise ValueError(f"filter has no token list: {path}")
    tokens = [str(token) for token in payload["tokens"]]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"filter contains duplicate tokens: {path}")
    return tokens


def formal_scores(paths, expected_seeds):
    if not paths:
        return {}
    if len(paths) != len(expected_seeds):
        raise ValueError("formal summary count must match seed-dir count")
    scores = {}
    for path in paths:
        payload = json.loads(path.read_text())
        seed = int(payload["eval_seed"])
        if seed in scores:
            raise RuntimeError(f"duplicate formal summary seed: {seed}")
        scores[seed] = float(payload["metrics"]["openloop_navtest"]["score"])
    if set(scores) != set(expected_seeds):
        raise RuntimeError("formal summary seeds do not match oracle audit seeds")
    return scores


def seed_result(seed_dir: Path, failure_tokens, args):
    audit_path = seed_dir / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError(f"oracle audit is not PASS: {audit_path}")
    records = load_records(Path(audit["records"]))
    by_token = {str(row["token"]): row for row in records}
    if len(by_token) != args.expected_num_tokens or len(records) != len(by_token):
        raise RuntimeError(f"record coverage mismatch under {seed_dir}")
    if not failure_tokens.issubset(by_token):
        raise RuntimeError(f"failures are not a subset of {seed_dir}")

    official = {}
    max_score_error = 0.0
    max_component_error = 0.0
    for selection in SELECTION_NAMES:
        path = seed_dir / f"{selection}_official_pdms" / "pdm_scores_merged.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_official_csv(path)
        if set(rows) != set(by_token):
            raise RuntimeError(
                f"official {selection} token coverage mismatch under {seed_dir}"
            )
        for token, row in rows.items():
            record = by_token[token]
            max_score_error = max(
                max_score_error,
                abs(float(row["score"]) - float(record[f"{selection}_reward"])),
            )
            for component in COMPONENT_NAMES:
                max_component_error = max(
                    max_component_error,
                    abs(
                        float(row[component])
                        - float(record[f"{selection}_components"][component])
                    ),
                )
        official[selection] = {
            "path": str(path.resolve()),
            "rows": rows,
            "navtest": summarize_official_rows(rows, sorted(by_token)),
            "navtest_failures": summarize_official_rows(
                rows, sorted(failure_tokens)
            ),
        }
    if max_score_error > args.official_atol:
        raise RuntimeError(
            f"online/official PDM score parity failed: {max_score_error}"
        )
    if max_component_error > args.official_atol:
        raise RuntimeError(
            "online/official PDM component parity failed: "
            f"{max_component_error}"
        )

    failure_records = [by_token[token] for token in sorted(failure_tokens)]
    return {
        "seed_dir": str(seed_dir.resolve()),
        "noise_seed": int(audit["noise_seed"]),
        "train_seed": int(audit["train_seed"]),
        "checkpoint": audit["paths"]["checkpoint"],
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "candidate_online": {
            "navtest": summarize_records(records),
            "navtest_failures": summarize_records(failure_records),
        },
        "official": official,
        "max_online_official_score_abs_error": max_score_error,
        "max_online_official_component_abs_error": max_component_error,
        "records_by_token": by_token,
    }


def aggregate_official(seed_results, block):
    aggregate = {}
    for selection in SELECTION_NAMES:
        aggregate[selection] = {
            metric: float(
                np.mean(
                    [
                        seed["official"][selection][block][metric]
                        for seed in seed_results
                    ]
                )
            )
            for metric in (*COMPONENT_NAMES, "score")
        }
    reference = aggregate["reference"]["score"]
    current = aggregate["current"]["score"]
    oracle = aggregate["oracle"]["score"]
    oracle_gain = oracle - reference
    aggregate["gains"] = {
        "current_reference": current - reference,
        "oracle_reference": oracle_gain,
        "oracle_capture_rate": (
            (current - reference) / oracle_gain if oracle_gain > 1e-12 else None
        ),
    }
    return aggregate


def token_averaged_gains(seed_results, tokens, numerator, denominator):
    gains = []
    for token in tokens:
        per_seed = []
        for seed in seed_results:
            rows_a = seed["official"][numerator]["rows"]
            rows_b = seed["official"][denominator]["rows"]
            per_seed.append(rows_a[token]["score"] - rows_b[token]["score"])
        gains.append(float(np.mean(per_seed)))
    return gains


def markdown_report(report):
    diagnosis = report["diagnosis"]
    label = {
        "selector_limited": "Selector 受限：generator 已提供明确可用上限",
        "generator_limited": "Generator 受限：best-of-20 上限不足",
        "mixed_or_inconclusive": "混合或暂不确定",
    }[diagnosis["classification"]]
    lines = [
        "# DiffusionDrive Best-of-20 Oracle 审计",
        "",
        f"结论：**{label}**。Oracle-reference 的完整 navtest 95% CI 为 "
        f"[{diagnosis['navtest_oracle_gain_ci95'][0]:.6f}, "
        f"{diagnosis['navtest_oracle_gain_ci95'][1]:.6f}]。",
        "",
        "> Oracle 使用日志未来信息选取 20 条中的最好轨迹，只表示候选上限，"
        "不是可部署模型，也不是 ClosedLoop oracle。",
        "",
    ]
    for block, title in (
        ("navtest", "OpenLoop-navtest"),
        ("navtest_failures", "OpenLoop-navtest_failures"),
    ):
        aggregate = report["aggregate"][block]
        lines.extend(
            [
                f"## {title}",
                "",
                "| 选择方式 | NC | DAC | EP | TTC | Comfort | PDM |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for selection in SELECTION_NAMES:
            row = aggregate[selection]
            lines.append(
                f"| {selection} | {row['no_at_fault_collisions']:.6f} | "
                f"{row['drivable_area_compliance']:.6f} | "
                f"{row['ego_progress']:.6f} | "
                f"{row['time_to_collision_within_bound']:.6f} | "
                f"{row['comfort']:.6f} | {row['score']:.6f} |"
            )
        gains = aggregate["gains"]
        capture = gains["oracle_capture_rate"]
        capture_text = "n/a" if capture is None else f"{capture:.6f}"
        lines.extend(
            [
                "",
                f"GRPO-reference={gains['current_reference']:+.6f}；"
                f"oracle-reference={gains['oracle_reference']:+.6f}；"
                f"capture rate={capture_text}。",
                "",
            ]
        )
    online = report["candidate_online_aggregate"]["navtest"]
    lines.extend(
        [
            "## 候选诊断",
            "",
            f"- 严格存在更优候选：{online['oracle_strictly_better_fraction']:.4%}",
            f"- Oracle gap > 0.01 / 0.05 / 0.10："
            f"{online['oracle_gap_gt_0.01_fraction']:.4%} / "
            f"{online['oracle_gap_gt_0.05_fraction']:.4%} / "
            f"{online['oracle_gap_gt_0.10_fraction']:.4%}",
            f"- 平均唯一候选数：{online['unique_candidate_count_1e4']:.3f} / 20",
            f"- 平均 pairwise ADE / FDE：{online['mean_pairwise_ade']:.6f} / "
            f"{online['mean_pairwise_fde']:.6f}",
            "",
            "## 完整性",
            "",
            f"- seeds：{report['seeds']}",
            f"- navtest tokens / failures tokens："
            f"{report['num_tokens']} / {report['num_failure_tokens']}",
            f"- online-official 最大 score/component 误差："
            f"{report['max_online_official_score_abs_error']:.3e} / "
            f"{report['max_online_official_component_abs_error']:.3e}",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    if len(args.seed_dir) != 3:
        raise ValueError("formal oracle audit requires exactly three seed dirs")
    failures_filter = args.failures_filter.expanduser().resolve()
    if not failures_filter.is_file():
        raise FileNotFoundError(failures_filter)
    failure_tokens = set(load_filter_tokens(failures_filter))
    if len(failure_tokens) != args.expected_num_failures:
        raise RuntimeError(
            f"failure token count mismatch: {len(failure_tokens)} != "
            f"{args.expected_num_failures}"
        )

    seed_results = [
        seed_result(path.expanduser().resolve(), failure_tokens, args)
        for path in args.seed_dir
    ]
    seed_results.sort(key=lambda row: row["noise_seed"])
    seeds = [row["noise_seed"] for row in seed_results]
    if seeds != [0, 1, 2]:
        raise RuntimeError(f"expected noise seeds [0, 1, 2], got {seeds}")
    common_tokens = set(seed_results[0]["records_by_token"])
    for seed in seed_results[1:]:
        if set(seed["records_by_token"]) != common_tokens:
            raise RuntimeError("oracle audit token sets differ across seeds")
    if len(common_tokens) != args.expected_num_tokens:
        raise RuntimeError("three-seed navtest coverage mismatch")

    reference_formal = formal_scores(args.formal_reference_summary, seeds)
    current_formal = formal_scores(args.formal_current_summary, seeds)
    formal_errors = {"reference": {}, "current": {}}
    for seed in seed_results:
        noise_seed = seed["noise_seed"]
        for selection, expected in (
            ("reference", reference_formal),
            ("current", current_formal),
        ):
            if not expected:
                continue
            actual = seed["official"][selection]["navtest"]["score"]
            error = abs(actual - expected[noise_seed])
            formal_errors[selection][str(noise_seed)] = error
            if error > args.formal_score_atol:
                raise RuntimeError(
                    f"seed {noise_seed} {selection} did not reproduce formal "
                    f"navtest PDM: error={error}"
                )

    navtest_tokens = sorted(common_tokens)
    failure_tokens_sorted = sorted(failure_tokens)
    nav_oracle_gains = token_averaged_gains(
        seed_results, navtest_tokens, "oracle", "reference"
    )
    nav_current_gains = token_averaged_gains(
        seed_results, navtest_tokens, "current", "reference"
    )
    failure_oracle_gains = token_averaged_gains(
        seed_results, failure_tokens_sorted, "oracle", "reference"
    )
    nav_oracle_ci = paired_bootstrap_ci(
        nav_oracle_gains,
        num_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    nav_current_ci = paired_bootstrap_ci(
        nav_current_gains,
        num_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 1,
    )
    failure_oracle_ci = paired_bootstrap_ci(
        failure_oracle_gains,
        num_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 2,
    )
    classification = classify_oracle_gain(
        *nav_oracle_ci,
        selector_limited_threshold=args.selector_limited_threshold,
        generator_limited_threshold=args.generator_limited_threshold,
    )

    all_records = []
    all_failure_records = []
    for seed in seed_results:
        records = list(seed["records_by_token"].values())
        all_records.extend(records)
        all_failure_records.extend(
            [seed["records_by_token"][token] for token in failure_tokens]
        )
    aggregate = {
        "navtest": aggregate_official(seed_results, "navtest"),
        "navtest_failures": aggregate_official(
            seed_results, "navtest_failures"
        ),
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_best_of_20_oracle_audit_three_seed",
        "seeds": seeds,
        "num_tokens": len(common_tokens),
        "num_failure_tokens": len(failure_tokens),
        "num_candidates_per_token": 20,
        "num_candidate_scores": len(common_tokens) * 20 * len(seeds),
        "aggregate": aggregate,
        "candidate_online_aggregate": {
            "navtest": summarize_records(all_records),
            "navtest_failures": summarize_records(all_failure_records),
        },
        "diagnosis": {
            "classification": classification,
            "selector_limited_threshold": args.selector_limited_threshold,
            "generator_limited_threshold": args.generator_limited_threshold,
            "navtest_oracle_gain_ci95": list(nav_oracle_ci),
            "navtest_current_gain_ci95": list(nav_current_ci),
            "navtest_failures_oracle_gain_ci95": list(failure_oracle_ci),
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "max_online_official_score_abs_error": max(
            seed["max_online_official_score_abs_error"] for seed in seed_results
        ),
        "max_online_official_component_abs_error": max(
            seed["max_online_official_component_abs_error"]
            for seed in seed_results
        ),
        "formal_reproduction_errors": formal_errors,
        "per_seed": [
            {
                **{
                    key: value
                    for key, value in seed.items()
                    if key not in ("records_by_token", "official")
                },
                "official": {
                    selection: {
                        key: value
                        for key, value in seed["official"][selection].items()
                        if key != "rows"
                    }
                    for selection in SELECTION_NAMES
                },
            }
            for seed in seed_results
        ],
        "assumptions": {
            "oracle_is_deployable": False,
            "closed_loop_oracle": False,
            "checkpoint_selection_used_navtest": False,
            "failures_are_derived_from_full_navtest": True,
        },
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = output.with_suffix(".tsv")
    markdown_path = output.with_suffix(".md")
    report["artifacts"] = {
        "json": str(output),
        "tsv": str(tsv_path),
        "markdown": str(markdown_path),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with tsv_path.open("w") as stream:
        columns = ["block", "selection", *COMPONENT_NAMES, "score"]
        stream.write("\t".join(columns) + "\n")
        for block in ("navtest", "navtest_failures"):
            for selection in SELECTION_NAMES:
                row = aggregate[block][selection]
                stream.write(
                    "\t".join(
                        [block, selection]
                        + [str(row[key]) for key in (*COMPONENT_NAMES, "score")]
                    )
                    + "\n"
                )
    markdown_path.write_text(markdown_report(report) + "\n")
    print(json.dumps(report, sort_keys=True))
    print(f"PASS three-seed oracle audit: {classification}")


if __name__ == "__main__":
    main()
