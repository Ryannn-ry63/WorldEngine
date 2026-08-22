import os
import argparse
import warnings
import time
import pickle
import asyncio
import logging
import shutil
import hashlib
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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_diffusiondrive_rollout_sidecar(
    result, cfg, file_monitor, planner_step, provenance, source_frame_path=None
):
    """Atomically export the exact 20-candidate action set for SimEngine."""
    output_dir = getattr(cfg.sim, 'diffusiondrive_rollout_sidecar_path', '')
    if not output_dir:
        return None
    context = result.get('diffusiondrive_rollout_context')
    if context is None:
        raise RuntimeError(
            'rollout sidecar was requested but the planning head exported no context'
        )
    required_shapes = {
        'candidate_features': (20, 256),
        'candidate_trajectories_8': (20, 8, 3),
        'route_bev_features': (20, 8, 256),
        'status_tokens': (1, 256),
        'ego_queries': (1, 256),
        'agents_queries': (30, 256),
        'reference_logits': (20,),
        'current_logits': (20,),
        'selected_indices': (),
    }
    for key, shape in required_shapes.items():
        value = np.asarray(context.get(key))
        if value.shape != shape:
            raise RuntimeError(
                f'DiffusionDrive rollout {key} shape {value.shape} != {shape}'
            )
        if not np.isfinite(value).all():
            raise RuntimeError(f'DiffusionDrive rollout {key} is non-finite')

    selected_index = int(np.asarray(context['selected_indices']).item())
    chosen_index = int(result['chosen_ind'])
    if selected_index != chosen_index:
        raise RuntimeError('rollout selected index disagrees with deployed action')
    contract = cfg.get('selector_rollout_contract', {})
    if contract.get('deployed_action_parity_required', False):
        logits_error = float(np.max(np.abs(
            np.asarray(context['current_logits'], dtype=np.float64)
            - np.asarray(context['reference_logits'], dtype=np.float64)
        )))
        if logits_error > 1e-5:
            raise RuntimeError(
                f'base rollout residual selector is non-zero: {logits_error}'
            )
    else:
        logits_error = None
    payload = {
        'schema_version': 1,
        'record_type': 'diffusiondrive_closed_loop_candidate_context',
        'scene_prefix': str(file_monitor.prefix),
        'planner_step': int(planner_step),
        'source_sample_token': str(result['token']),
        'selected_index': selected_index,
        'deployed_trajectory': np.asarray(result['trajectory'], dtype=np.float32),
        'raw_observation_path': (
            None
            if source_frame_path is None
            else str(Path(source_frame_path).expanduser().resolve())
        ),
        'current_reference_logits_max_abs_error': logits_error,
        **provenance,
    }
    for key in required_shapes:
        payload[key] = np.asarray(context[key])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f'{file_monitor.prefix}_{planner_step}.pkl'
    temporary_path = final_path.with_suffix('.pkl.tmp')
    with temporary_path.open('wb') as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, final_path)
    return final_path

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

async def run_inference_loop(model, cfg, logger, rollout_provenance):
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
            if not current_queue:
                if os.path.exists(stop_file_path):
                    logger.info(
                        "Inference queue is empty after the simulation completion "
                        "signal; exiting normally"
                    )
                    break
                raise RuntimeError(
                    "Inference queue became empty without a simulation completion signal"
                )
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
            )
            plan_result, plan_idx = post_processor.process(result)

            # Publish the sidecar before the matching action.  The final plan
            # rename is the synchronization signal consumed by SimEngine.
            planner_step = cfg.queue_length + scene_step
            tmp_path = os.path.join(save_path, f'{file_monitor.prefix}_{planner_step}_tmp.npy')
            np.save(tmp_path, plan_result)
            sidecar_path = save_diffusiondrive_rollout_sidecar(
                result,
                cfg,
                file_monitor,
                planner_step,
                rollout_provenance,
                source_frame_path=current_queue[-1],
            )
            os.rename(tmp_path, os.path.join(save_path, f'{file_monitor.prefix}_{planner_step}.npy'))
            if sidecar_path is not None:
                logger.info(f'Saved DiffusionDrive rollout sidecar {sidecar_path}')

            if value is not None:
                np.save(
                    os.path.join(save_path, f'value_{file_monitor.prefix}_{cfg.queue_length + scene_step}.npy'),
                    value
                )

            # Save plan_idx to csv file
            new_row = pd.DataFrame({
                'prefix': [file_monitor.prefix],
                'step': [cfg.queue_length + scene_step],
                'plan_idx': [plan_idx]
            })
            new_row.to_csv(csv_path, mode='a', header=False, index=False)
            logger.info(f"Saved traj at step {cfg.queue_length + scene_step}")

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
    rollout_provenance = {
        'config_path': str(Path(args.config).expanduser().resolve()),
        'config_sha256': sha256_file(args.config),
        'checkpoint_path': str(Path(args.checkpoint).expanduser().resolve()),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'candidate_noise_namespace': None,
    }
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
            monitored_folder = ''
        ))
        cfg.sim = sim_cfg
        cfg.data.test.type = "NavSimOpenSceneE2EClosedLoop"

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    planning_head_cfg = cfg.model.get('planning_head', {})
    rollout_provenance['candidate_noise_namespace'] = planning_head_cfg.get(
        'candidate_noise_namespace'
    )
    rollout_provenance['resolved_config_sha256'] = hashlib.sha256(
        cfg.pretty_text.encode('utf-8')
    ).hexdigest()
    rollout_provenance['code_sha'] = os.getenv(
        'DIFFUSIONDRIVE_ROLLOUT_CODE_SHA'
    )
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
        asyncio.run(run_inference_loop(model, cfg, logger, rollout_provenance))
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
