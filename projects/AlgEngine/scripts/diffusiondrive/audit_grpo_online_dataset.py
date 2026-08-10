#!/usr/bin/env python3
"""Audit navtrain/metric-cache coverage without loading large annotations."""

import argparse
import json
from pathlib import Path

import yaml
from mmcv import Config
from navsim.common.dataloader import MetricCacheLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(str(args.config.resolve()))
    nav_filter_path = Path(cfg.data.train.nav_filter_path).expanduser().resolve()
    with nav_filter_path.open("r") as stream:
        nav_filter = yaml.safe_load(stream)
    nav_tokens = set(nav_filter.get("tokens") or nav_filter["scenario_tokens"])
    metric_cache_path = Path(
        cfg.model.planning_head.online_reward.metric_cache_path
    ).expanduser().resolve()
    cache_tokens = set(MetricCacheLoader(metric_cache_path).metric_cache_paths)
    covered_tokens = nav_tokens.intersection(cache_tokens)
    if not covered_tokens:
        raise RuntimeError(
            "navtrain and metric cache have no common tokens: "
            f"nav_filter={nav_filter_path} ({len(nav_tokens)} tokens), "
            f"metric_cache={metric_cache_path} ({len(cache_tokens)} tokens)"
        )
    report = {
        "status": "PASS",
        "nav_filter_path": str(nav_filter_path),
        "metric_cache_path": str(metric_cache_path),
        "navtrain_tokens_before_cache_filter": len(nav_tokens),
        "navtrain_cache_covered_tokens": len(covered_tokens),
        "metric_cache_index_tokens": len(cache_tokens),
        "tokens_removed_before_training": len(nav_tokens - cache_tokens),
        "coverage_fraction": len(covered_tokens) / len(nav_tokens),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
