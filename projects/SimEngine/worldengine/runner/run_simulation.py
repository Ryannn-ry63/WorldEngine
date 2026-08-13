import logging
import pickle
from itertools import islice

import hydra
from omegaconf import DictConfig

from worldengine.runner.builders.worker_pool_builder import build_worker
from worldengine.runner.builders.env_builder import build_envs
from worldengine.runner.executor import run_runners

logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = '../configs'
CONFIG_NAME = 'default_runner'


def limit_input_scenes(scene_dict, maximum_scenarios):
    """Deterministically cap smoke/pilot inputs before worker distribution."""
    if maximum_scenarios is None:
        return scene_dict
    maximum_scenarios = int(maximum_scenarios)
    if maximum_scenarios < 1:
        raise ValueError("max_successful_scenarios must be positive or null")
    if len(scene_dict) <= maximum_scenarios:
        return scene_dict
    return dict(islice(scene_dict.items(), maximum_scenarios))

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info('WorldEngine is running...')

    # Construct builder
    worker = build_worker(cfg)

    # Construct simulations/environments
    scenes_dict = pickle.load(open(cfg.data_file_path, 'rb'))
    original_scene_count = len(scenes_dict)
    scenes_dict = limit_input_scenes(
        scenes_dict, cfg.get("max_successful_scenarios")
    )
    logger.info(
        "Selected %d/%d input scenes (max_successful_scenarios=%s)",
        len(scenes_dict),
        original_scene_count,
        cfg.get("max_successful_scenarios"),
    )
    envs = build_envs(cfg=cfg, worker=worker, scene_dict=scenes_dict)

    logger.info('Running simulation...')
    run_runners(envs=envs, worker=worker, cfg=cfg)
    logger.info('Finished running simulation!')

if __name__ == "__main__":
    main()
