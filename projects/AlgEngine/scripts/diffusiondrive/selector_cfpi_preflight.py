"""Checks only; full GPU kernels are checked by existing Hopper preflights."""
import argparse
import json


def validate_hardware(torch, expected_gpus):
    if str(torch.__version__) != "2.0.1+cu118" or torch.version.cuda != "11.8":
        raise RuntimeError(f"Expected torch 2.0.1+cu118 / CUDA 11.8, got {torch.__version__} / {torch.version.cuda}")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if count != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} visible H100 GPUs, found {count}: {names}")
    if any("H100" not in name for name in names):
        raise RuntimeError(f"Expected H100 allocation, found {names}")
    capabilities = [torch.cuda.get_device_capability(i) for i in range(count)]
    if any(capability != (9, 0) for capability in capabilities):
        raise RuntimeError(f"Expected sm_90, found {capabilities}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.config:
        from mmcv import Config
        config = Config.fromfile(args.config)
        contract = config.selector_rollout_contract
        assert contract.experiment == "diffusiondrive_selector_cfpi_causal_cache_v1"
        assert contract.source_data_split == "train"
        assert not contract.inference_uses_reward_or_q
        assert config.model.planning_head.online_reward is None
        assert config.model.planning_head.export_rollout_context
        _ = config.pretty_text
    else:
        import torch
        validate_hardware(torch, args.gpus)
    print(json.dumps(dict(status="PASS")))


if __name__ == "__main__":
    main()
