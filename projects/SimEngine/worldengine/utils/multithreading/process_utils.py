import os
import logging

logger = logging.getLogger(__name__)

def get_process_id() -> str:

    import ray
    if ray.is_initialized():
        gpu_ids = ray.get_gpu_ids()
        # Feedback planner folders use logical split_0..N-1, while Ray returns
        # physical IDs (or UUIDs) from the parent's CUDA_VISIBLE_DEVICES map.
        feedback_map = os.environ.get('WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP')
        if feedback_map is not None:
            devices = [x.strip() for x in feedback_map.split(',')]
            if (len(gpu_ids)!=1 or str(gpu_ids[0]) not in devices
                    or len(set(devices))!=len(devices) or any(not x for x in devices)):
                raise RuntimeError('Feedback Ray GPU assignment does not match logical worker map')
            return str(devices.index(str(gpu_ids[0])))
        if len(gpu_ids) > 0:
            return str(gpu_ids[0])

    return str(os.getpid())

def resolve_worker_placeholders(config):
    """
    Resolve __WORKER_ID__ placeholders in config with the actual worker/GPU ID.

    This function should be called inside each worker (e.g., in lazy_init)
    to replace placeholders with worker-specific values.

    Supported placeholders:
        - __WORKER_ID__: Replaced with GPU ID, WORLDENGINE_PROCESS_ID, or OS PID

    Args:
        config: OmegaConf DictConfig or dict

    Returns:
        Config with placeholders resolved
    """
    from omegaconf import OmegaConf, DictConfig

    worker_id = get_process_id()

    # Convert to YAML string for placeholder replacement
    if isinstance(config, DictConfig):
        config_str = OmegaConf.to_yaml(config)
    else:
        config_str = OmegaConf.to_yaml(OmegaConf.create(config))

    # Replace placeholder
    if '__WORKER_ID__' in config_str:
        config_str = config_str.replace('__WORKER_ID__', f"{config.worker_id_prefix}{worker_id}")
        logger.info(f"Resolved __WORKER_ID__ placeholder to: {config.worker_id_prefix}{worker_id}")

    # Convert back to DictConfig
    return OmegaConf.create(config_str)
