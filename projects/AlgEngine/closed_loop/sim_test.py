import os
import argparse
import warnings
import time
import pickle
import asyncio
import logging
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mmcv import Config, DictAction, ConfigDict
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet3d.utils.logger import get_root_logger

from closed_loop.monitor import AsyncFileMonitor
from closed_loop.post_processor import ScorePostProcessor

warnings.filterwarnings("ignore")
WORLDENGINE_ROOT = os.getenv('WORLDENGINE_ROOT', os.path.abspath('.'))

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--log-dir', type=str, default='', help='log directory')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def merge_ann_files(cfg,ann_files_list, merged_index):
    save_dir = cfg.sim.merged_ann_save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'merged_{merged_index}.pkl')
    merged_data = {'infos': []}

    for ann_path in ann_files_list:
        with open(ann_path, 'rb') as f:
            data = pickle.load(f)
            merged_data['infos'].append(data)

    with open(save_path, 'wb') as f:
        pickle.dump(merged_data, f)

    return save_path

def clean_related_files(cfg, logger):
    clean_temp_files = getattr(cfg.sim, 'clean_temp_files', False)
    clean_record_data = getattr(cfg.sim, 'clean_record_data', False)
    frames_path = getattr(cfg.sim, 'monitored_folder', '')
    plan_save_path = getattr(cfg.sim, 'plan_save_path', '')
    merged_ann_save_dir = getattr(cfg.sim, 'merged_ann_save_dir', '')

    logger.info("Cleaning related files...")
    if clean_temp_files:
        if frames_path and os.path.exists(frames_path):
            for file in Path(frames_path).glob('*.pkl'):
                file.unlink()

        if plan_save_path and os.path.exists(plan_save_path):
            for file in Path(plan_save_path).glob('*.npy'):
                file.unlink()

        if merged_ann_save_dir and os.path.exists(merged_ann_save_dir):
            for file in Path(merged_ann_save_dir).glob('*.pkl'):
                file.unlink()

    if clean_record_data:
        data_root = getattr(cfg, 'data_root', '')
        if data_root and os.path.exists(data_root):
            for folder in os.listdir(data_root):
                folder_path = os.path.join(data_root, folder)
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)

def save_grpo_candidate_sidecar(cfg, result, prefix, step):
    """Atomically publish exact GRPO candidates for same-state scoring."""
    candidate_dir = getattr(cfg.sim, "grpo_candidate_path", "")
    if not candidate_dir or "grpo_final_action" not in result:
        return None
    if "all_trajectories" not in result:
        raise KeyError("GRPO rollout result has no all_trajectories")
    os.makedirs(candidate_dir, exist_ok=True)
    output_path = os.path.join(candidate_dir, f"{prefix}_{step}.npz")
    temporary_path = output_path + ".tmp"
    payload = {
        "candidate_trajectories": np.asarray(
            _to_rollout_record_value(result["all_trajectories"]),
            dtype=np.float64,
        ),
        "selected_index": np.asarray(
            int(result.get("chosen_ind", -1)), dtype=np.int64
        ),
        "token": np.asarray(str(result.get("token", ""))),
        "policy_version": np.asarray(
            str(getattr(cfg.sim, "policy_version", "unknown"))
        ),
    }
    with open(temporary_path, "wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary_path, output_path)
    return output_path


def _to_rollout_record_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_rollout_record_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_rollout_record_value(v) for v in value]
    return value


def save_rollout_record(
    cfg,
    logger,
    result,
    plan_result,
    plan_idx,
    current_queue,
    ann_file,
    prefix,
    step,
    value=None,
):
    record_dir = getattr(cfg.sim, "rollout_record_path", "")
    if not record_dir:
        return

    os.makedirs(record_dir, exist_ok=True)
    token = result.get("token", "")
    record_name = f"{prefix}_{step}.pkl"
    record_path = os.path.join(record_dir, record_name)

    metric_keys = [
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "score",
        "ade_4s",
        "fde_4s",
    ]
    result_keys = [
        "trajectory",
        "trajectory_8",
        "all_trajectories",
        "all_trajectories_8",
        "poses_cls",
        "metric_cache_path",
        "chosen_ind",
        "grpo_initial_sample",
        "grpo_transition_action",
        "grpo_final_action",
        "grpo_old_log_probs",
    ] + metric_keys

    record = {
        "prefix": prefix,
        "step": step,
        "token": token,
        "score_mode": cfg.sim.get("score_mode", "lookup"),
        "policy_version": cfg.sim.get("policy_version", "unknown"),
        "plan_idx": int(plan_idx) if plan_idx is not None else -1,
        "current_pkl": current_queue[-1] if current_queue else None,
        "history_queue": list(current_queue),
        "merged_ann_file": ann_file,
        "plan_trajectory": _to_rollout_record_value(plan_result),
        "value": _to_rollout_record_value(value) if value is not None else None,
        "model_result": {
            key: _to_rollout_record_value(result[key])
            for key in result_keys
            if key in result
        },
    }

    tmp_path = record_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.rename(tmp_path, record_path)

    index_path = os.path.join(record_dir, "rollout_index.csv")
    row = {
        "prefix": prefix,
        "step": step,
        "token": token,
        "record_path": record_path,
        "current_pkl": current_queue[-1] if current_queue else "",
        "merged_ann_file": ann_file,
        "plan_idx": int(plan_idx) if plan_idx is not None else -1,
    }
    for key in metric_keys:
        row[key] = result.get(key, np.nan)

    write_header = not os.path.exists(index_path)
    pd.DataFrame([row]).to_csv(index_path, mode="a", header=write_header, index=False)
    logger.info(f"Saved rollout record at {record_path}")

async def run_inference_loop(model, cfg, logger):
    """async run inference loop"""
    MONITORED_FOLDER = cfg.sim.monitored_folder
    logger.info(f"MONITORED_FOLDER: {MONITORED_FOLDER}")
    stop_file_path = os.path.join(os.path.dirname(cfg.data_root.rstrip('/')), 'simulation_completed.flag')
    logger.info(f"STOP_FILE_PATH: {stop_file_path}")

    # initialize file monitor
    file_monitor = AsyncFileMonitor(
        folder_path=MONITORED_FOLDER,
        check_interval=5,
        max_history=cfg.queue_length,
        max_frame_per_scene=12,
        wait_for_files=True,
        max_wait_time=99999,
        logger=logger,
        stop_file_path=stop_file_path
    )

    # Create csv file to save plan_idx
    save_path = cfg.sim.plan_save_path
    csv_path = os.path.join(save_path, 'plan_idx.csv')
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=['prefix', 'step', 'plan_idx']).to_csv(csv_path, index=False)

    try:
        current_queue = list(file_monitor.file_history)

        if len(set(current_queue)) != cfg.queue_length:
            if os.path.exists(stop_file_path):
                logger.info("Simulation already completed before inference started. Nothing to do.")
                return
            raise ValueError(f"Initial files: {current_queue} is not equal to queue_length: {cfg.queue_length}")

        step = 0
        scene_step = 0
        force_reinitialize = False
        MAX_STEP = cfg.sim.maximum_step

        while step <= MAX_STEP:
            retry_count = 0
            logger.info(f"processing {os.path.basename(current_queue[-1])}")
            ann_file = merge_ann_files(cfg, current_queue, scene_step + cfg.queue_length)
            # model inference
            cfg.data.test.ann_file = ann_file
            cfg.data.test.pipeline[0].img_root = cfg.data_root + "sensor_blobs/"
            dataset = build_dataset(cfg.data.test)
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=1,
                workers_per_gpu=1,
                dist=False,
                shuffle=False,
                nonshuffler_sampler=cfg.data.nonshuffler_sampler,
            )

            result_list = model_inference(model, data_loader)
            result = result_list[0] # TODO: need to save reward_dict, especially values
            if 'value' in result.keys():
                value = result['value']
            else:
                value = None

            # post-processing
            post_processor = ScorePostProcessor(
                cfg.sim.post_process_path,
                current_queue[-1],
                score_mode=cfg.sim.get('score_mode', 'lookup'),
            )
            plan_result, plan_idx = post_processor.process(result)
            output_step = cfg.queue_length + scene_step
            save_grpo_candidate_sidecar(
                cfg, result, file_monitor.prefix, output_step)

            # save result
            tmp_path = os.path.join(save_path, f'{file_monitor.prefix}_{output_step}_tmp.npy')
            np.save(tmp_path, plan_result)
            os.rename(tmp_path, os.path.join(save_path, f'{file_monitor.prefix}_{output_step}.npy'))

            if value is not None:
                np.save(
                    os.path.join(save_path, f'value_{file_monitor.prefix}_{output_step}.npy'),
                    value
                )

            save_rollout_record(
                cfg=cfg,
                logger=logger,
                result=result,
                plan_result=plan_result,
                plan_idx=plan_idx,
                current_queue=current_queue,
                ann_file=ann_file,
                prefix=file_monitor.prefix,
                step=output_step,
                value=value,
            )

            # Save plan_idx to csv file
            new_row = pd.DataFrame({
                'prefix': [file_monitor.prefix],
                'step': [output_step],
                'plan_idx': [plan_idx]
            })
            new_row.to_csv(csv_path, mode='a', header=False, index=False)
            logger.info(f"Saved traj at step {output_step}")

            step += 1
            scene_step += 1

            reinit = file_monitor._reinitialize_queue(force_reinitialize=force_reinitialize)
            if not reinit:
                if os.path.exists(stop_file_path):
                    logger.info(f"Stop signal file detected in sim_test: {stop_file_path}")
                    logger.info("Exiting inference loop due to stop signal")
                    break
                update_queue = file_monitor.get_current_queue()
                current_queue = list(update_queue)
                while not update_queue:
                    if retry_count < 300:
                        update_queue = file_monitor.get_current_queue()
                        current_queue = list(update_queue)
                        retry_count += 1
                        time.sleep(1)
                    else:
                        logger.info("Retry count exceeded, reinitializing queue")
                        reinit = file_monitor._reinitialize_queue(force_reinitialize=True)
                        current_queue = list(file_monitor.file_history)
                        scene_step = 0
                        break
            else:
                current_queue = list(file_monitor.file_history)
                scene_step = 0

    finally:
        file_monitor.stop()
        clean_related_files(cfg, logger)
        logger.info("Inference loop finished")

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.log_dir:
        time_postfix = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(args.log_dir, f'mmdet_client_{time_postfix}.log')
        logger = get_root_logger(log_level=logging.INFO, name="mmdet", log_file=log_file)
    else:
        logger = get_root_logger(log_level=logging.INFO, name="mmdet")

    if 'sim' not in cfg:
        logger.info("Setting Simulation Config")
        # set sim config
        sim_cfg = ConfigDict(dict(
            max_wait_time=50,
            maximum_step=100000000,
            post_process_path=os.path.join(WORLDENGINE_ROOT, "data/alg_engine/test_8192_kmeans.npy"),
            plan_save_path='',
            merged_ann_save_dir = '',
            monitored_folder = '',
            rollout_record_path = '',
            grpo_candidate_path = '',
            grpo_reward_path = '',
            policy_version = 'unknown',
        ))
        cfg.sim = sim_cfg

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.sim.setdefault('score_mode', cfg.model.planning_head.get('score_mode', 'lookup'))
    cfg.data.test.type = "NavSimOpenSceneE2EClosedLoop"
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if hasattr(cfg.model, 'img_backbone'):
        # let SyncBatchNorm switch into BatchNorm
        cfg.model.img_backbone.norm_cfg = dict(type='BN', requires_grad=True)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDataParallel(
        model.cuda(),
        device_ids=[0])  # single gpu mode
    model.eval()

    # After the model is ready, run the asynchronous inference loop
    try:
        asyncio.run(run_inference_loop(model, cfg, logger))
    except KeyboardInterrupt:
        logger.info("Inference stopped by user")
    except Exception as e:
        raise e

def model_inference(model, data_loader):
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            return result

if __name__ == '__main__':
    main()
