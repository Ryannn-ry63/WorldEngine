"""
A script to convert nuPlan data into WorldEngine format.
"""
r"""
python worldengine/utils/dataset_utils/nuplan/digitaltwin_nuplan_converter_navsim_filter.py \
    --navsim-filters $ALGENGINE_ROOT/configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_collision.yaml \
        $ALGENGINE_ROOT/configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_ep_1pct.yaml \
        $ALGENGINE_ROOT/configs/navsim_splits/navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_off_road.yaml \
    --out-dir data/sim_engine/scenarios/original/navtrain_vadv2_50pct_failures \
    --num-processes 2
"""

import argparse
import hashlib
import json
import numpy as np
from pyquaternion import Quaternion
import os
from pathlib import Path
import pickle
from tqdm import tqdm
import yaml
import multiprocessing
from multiprocessing import Pool
from functools import partial

from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from worldengine.utils.dataset_utils.nuplan.nuplan_utils import (
    EGO, extract_traffic, extract_traffic_light, extract_map_features, NUPLAN_REAL_LIDAR2EGO_ROTATION, NUPLAN_REAL_LIDAR2EGO_TRANSLATION
)
from worldengine.utils.dataset_utils.nuplan.digitaltwin_config import load_config, VideoScene


def read_openscene_data_infos(openscene_dataroot, log_name, lidar_pc_tokens):
    openscene_data_infos = os.path.join(openscene_dataroot, 'meta_datas', 'trainval', f"{log_name}.pkl")
    if not os.path.exists(openscene_data_infos):
        openscene_data_infos = os.path.join(openscene_dataroot, 'meta_datas', 'test', f"{log_name}.pkl")
    assert os.path.exists(openscene_data_infos), (
        f"OpenScene meta_data not found for log '{log_name}'. "
        f"Searched: {openscene_dataroot}/meta_datas/{{trainval,test}}/{log_name}.pkl"
    )
    openscene_data_infos = pickle.load(open(openscene_data_infos, 'rb'))
    openscene_data_dict = {frame['token']: frame for frame in openscene_data_infos}
    openscene_data_dict = {token: openscene_data_dict[token] for token in lidar_pc_tokens}
    return openscene_data_dict

def create_scenario_description(
    args,
    digitaltwin_config,
    lidar_pcs,
    openscene_dataroot,
    start_frame_idx,
    end_frame_idx,
    video_info,
    log_db,
    log_file,
    map_api,
    map_location,
    video_name,
):
    lidar_pcs = lidar_pcs[start_frame_idx:end_frame_idx + 1]
    lidar_pcs = lidar_pcs[::args.sample_interval]
    lidar_pc_tokens = [lidar_pc.token for lidar_pc in lidar_pcs]

    log_length = len(lidar_pcs)

    info_dict = dict(
        id=video_name,
        name=video_name,
        dataset='digitaltwin.nuplan',
        map=map_location,
        token=video_name,
        log_length=log_length,
        sample_rate=args.sample_interval,
        base_timestamp=lidar_pcs[0].timestamp,
        metadata=dict(
            nuplan_lidar_pc_tokens=lidar_pc_tokens,
            openscene_data_infos_dict=read_openscene_data_infos(openscene_dataroot, video_info['log_name'], lidar_pc_tokens),
            digitaltwin_asset_id=digitaltwin_config.road_block_name,
        )
    )

    ego_pose = lidar_pcs[0].ego_pose
    q = Quaternion(ego_pose.qw, ego_pose.qx, ego_pose.qy, ego_pose.qz)
    initial_ego_state = EgoState.build_from_rear_axle(
        StateSE2(ego_pose.x, ego_pose.y, q.yaw_pitch_roll[0]),
        tire_steering_angle=0.0,
        vehicle_parameters=get_pacifica_parameters(),
        time_point=TimePoint(ego_pose.timestamp),
        rear_axle_velocity_2d=StateVector2D(ego_pose.vx, y=ego_pose.vy),
        rear_axle_acceleration_2d=StateVector2D(
            x=ego_pose.acceleration_x, y=ego_pose.acceleration_y),
    )
    initial_center = [initial_ego_state.waypoint.x,
                        initial_ego_state.waypoint.y]
    info_dict['metadata']['old_origin_in_current_coordinate'] = -np.asarray(initial_center)
    info_dict['metadata']['digitaltwin_ego2globals'] = [info['ego2global'] for info in video_info['frame_infos']]

    # do ego sensor info extraction.
    log_cam_infos = {camera.token : camera for camera in log_db.log.cameras}
    cams = {}
    for cam_info in log_cam_infos.values():
        cam_name = cam_info.channel

        # obtain calibrated camera info in digitaltwin
        digitaltwin_cam_info = video_info['frame_infos'][0]['cams'][cam_name]
        if 'colmap_param' in digitaltwin_cam_info:
            intrinsic = digitaltwin_cam_info['colmap_param']['cam_intrinsic']
            distortion = digitaltwin_cam_info['colmap_param']['distortion']
        else:
            intrinsic = digitaltwin_cam_info['cam_intrinsic']
            distortion = digitaltwin_cam_info['distortion']

        cams[cam_name] = {}
        cams[cam_name]['channel'] = cam_info.channel
        cams[cam_name]['sensor2ego_rotation'] = cam_info.quaternion.elements
        cams[cam_name]['sensor2ego_translation'] = cam_info.translation_np
        cams[cam_name]['intrinsic'] = intrinsic
        cams[cam_name]['distortion'] = distortion
        cams[cam_name]['height'] = 1080
        cams[cam_name]['width'] = 1920

    info_dict['cameras'] = cams
    info_dict['lidar'] = {
        'channel': 'LIDAR_TOP',
        'sensor2ego_rotation': np.array(NUPLAN_REAL_LIDAR2EGO_ROTATION),
        'sensor2ego_translation': np.array(NUPLAN_REAL_LIDAR2EGO_TRANSLATION),
    }

    # do object track statistics.
    info_dict.update(dict(
        object_track=extract_traffic(
            log_file=log_file,
            lidar_pcs=lidar_pcs,
            initial_ego_center=initial_center,
            sample_interval=args.sample_interval),
        sdc_id=EGO,
    ))
    # CAUTION: the velocity may calculate again.
    # for track_token, track_info in info_dict['object_track'].items():
    #     # use position difference to calculate velocity
    #     if track_info['type'] == 'VEHICLE':
    #         track_vel = (track_info['state']['position'][1:] - track_info['state']['position'][:-1]) / (0.05 * args.sample_interval)
    #         track_info['state']['velocity'][:-1] = track_vel[..., :2]  # only x,y direction velocity
    #         track_info['state']['velocity'][-1] = track_info['state']['velocity'][-2]  # last frame copy the velocity of the second last frame

    # do traffic light statistics.
    #  dynamic_map_states: elements related to traffic light,
    #  some changeable components in the scene.
    info_dict.update(dict(
        dynamic_map_states=extract_traffic_light(
            log_file=log_file,
            lidar_pcs=lidar_pcs,
            map_api=map_api,
            initial_ego_center=initial_center,
        )
    ))

    # do map element extraction.
    # extract polygons of each map element.
    info_dict.update(
        map_features=extract_map_features(
            map_api=map_api,
            initial_ego_center=initial_center)
    )

    return info_dict

def create_digitaltwin_info_central(video_scene: VideoScene, args=None):
    """Process one video scene and save results to individual pickle files."""
    processed_scenarios = []
    nuplan_root_path = args.nuplan_root_path
    nuplan_db_path = args.nuplan_db_path
    nuplan_map_version = args.nuplan_map_version
    nuplan_map_root = args.nuplan_map_root
    openscene_dataroot = args.openscene_dataroot
    out_dir = args.out_dir

    # Create chunks directory for individual scenario files
    chunks_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    video_scene_dict = video_scene.video_scene_dict
    digitaltwin_config = video_scene.config
    video_info = list(video_scene_dict.values())[0]

    expected_names = [
        f"{digitaltwin_config.central_log}-{token}"
        for token in digitaltwin_config.central_tokens
    ]
    if args.resume_chunks and expected_names:
        reusable = []
        for scenario_name in expected_names:
            chunk_file = os.path.join(chunks_dir, f"{scenario_name}.pkl")
            if not os.path.isfile(chunk_file):
                break
            try:
                with open(chunk_file, "rb") as stream:
                    payload = pickle.load(stream)
                if set(payload) != {scenario_name}:
                    break
            except (OSError, EOFError, ValueError, pickle.PickleError):
                break
            reusable.append(scenario_name)
        if len(reusable) == len(expected_names):
            return reusable

    log_file = os.path.join(nuplan_db_path, f"{video_info['log_name']}.db")
    assert os.path.exists(log_file), f"Log file {log_file} does not exist."

    log_db = NuPlanDB(nuplan_root_path, log_file, None, verbose=True)
    map_location = log_db.log.map_version
    map_api = get_maps_api(nuplan_map_root, nuplan_map_version, map_location)  # NOTE: lru cached

    lidar_pcs = log_db.lidar_pc # check your SQLAlchemy version, 1.4.27 works well but lower may introduce strange problem
    lidar_pc_tokens = [pc.token for pc in lidar_pcs]
    for central_token in digitaltwin_config.central_tokens:
        if central_token in lidar_pc_tokens:
            central_frame_idx = lidar_pc_tokens.index(central_token)
            start_frame_idx = max(central_frame_idx - int(20 * 1.5), 0)
            end_frame_idx = min(central_frame_idx + 20 * 10, len(lidar_pcs) - 1)
            current_video_name = f"{digitaltwin_config.central_log}-{central_token}"
            try:
                info_dict = create_scenario_description(
                    args=args,
                    digitaltwin_config=digitaltwin_config,
                    lidar_pcs=lidar_pcs,
                    openscene_dataroot=openscene_dataroot,
                    start_frame_idx=start_frame_idx,
                    end_frame_idx=end_frame_idx,
                    video_info=video_info,
                    log_db=log_db,
                    log_file=log_file,
                    map_api=map_api,
                    map_location=map_location,
                    video_name=current_video_name,
                )
                # Save individual scenario to avoid large pipe transfers
                chunk_file = os.path.join(chunks_dir, f"{current_video_name}.pkl")
                with open(chunk_file, "wb") as f:
                    pickle.dump({current_video_name: info_dict}, f, protocol=pickle.HIGHEST_PROTOCOL)
                processed_scenarios.append(current_video_name)
            except KeyError as e:
                print(f"{current_video_name} failed due to mismatch OpenScene info")
                continue

    # Return only scenario names, not the full data
    return processed_scenarios

def parse_args():
    parser = argparse.ArgumentParser(description="Create world engine pkl from Digital Twin config.")
    parser.add_argument(
        '--digitaltwin-asset-root', type=str,
        default="data/sim_engine/assets/navtrain")
    parser.add_argument(
        '--navsim-filters',
        type=str,
        required=True,
        nargs='+',
        help='Path(s) to the navsim config file(s). Can be a single path or a list of paths separated by space.'
    )
    
    parser.add_argument(
        "--nuplan-root-path", help="the path to nuplan root path.",
        default='data/raw/nuplan/dataset/nuplan-v1.1')
    parser.add_argument(
        "--nuplan-db-path", help="the dir saving nuplan db.",
        default='data/raw/nuplan/dataset/nuplan-v1.1/splits/all_sensor')
    parser.add_argument(
        "--nuplan-sensor-path", help="the dir to nuplan sensor data.",
        default='data/raw/nuplan/dataset/nuplan-v1.1/sensor_blobs')
    parser.add_argument(
        "--nuplan-map-version", help="nuplan mapping dataset version.",
        default='nuplan-maps-v1.0')
    parser.add_argument(
        "--nuplan-map-root", help="path to nuplan map data.",
        default='data/raw/nuplan/dataset/maps')
    parser.add_argument(
        "--openscene-dataroot", help="path to openscene data.",
        default='data/raw/openscene-v1.1')
    parser.add_argument("--out-dir", help="output path.")

    # data configurations.
    parser.add_argument(
        "--sample-interval", type=int, default=10, help="interval of key frame samples."
    )

    parser.add_argument('--num-processes', type=int, default=multiprocessing.cpu_count() - 1)
    parser.add_argument(
        '--num-splits',
        type=int,
        default=1,
        help='Write this many deterministic scenario shard pickles.',
    )
    parser.add_argument(
        '--shards-only',
        action='store_true',
        help='Do not also write all_scenarios.pkl (useful for very large sets).',
    )
    parser.add_argument(
        '--expected-scenarios',
        type=int,
        help='Fail unless conversion produces exactly this many scenarios.',
    )
    parser.add_argument(
        '--resume-chunks',
        action='store_true',
        help='Reuse already complete per-scenario chunks after validating them.',
    )

    args = parser.parse_args()
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_scenario_shards(scenario_names, num_splits):
    """Return stable, balanced, disjoint scenario-name shards."""
    if num_splits < 1:
        raise ValueError('--num-splits must be positive')
    ordered = sorted(set(scenario_names))
    if len(ordered) != len(scenario_names):
        raise RuntimeError('conversion produced duplicate scenario names')
    base_size, remainder = divmod(len(ordered), num_splits)
    shards = []
    start = 0
    for split_index in range(num_splits):
        size = base_size + (1 if split_index < remainder else 0)
        shards.append(ordered[start:start + size])
        start += size
    if start != len(ordered):
        raise RuntimeError('scenario shard accounting drifted')
    return shards


def load_scenario_chunks(chunks_dir, scenario_names):
    scenarios = {}
    for scenario_name in tqdm(scenario_names, desc='Loading chunks'):
        chunk_file = os.path.join(chunks_dir, f'{scenario_name}.pkl')
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f'chunk file not found: {chunk_file}')
        with open(chunk_file, 'rb') as stream:
            chunk_data = pickle.load(stream)
        if set(chunk_data) != {scenario_name}:
            raise RuntimeError(f'invalid scenario chunk: {chunk_file}')
        scenarios.update(chunk_data)
    return scenarios


def write_scenario_outputs(
    out_dir,
    chunks_dir,
    scenario_names,
    num_splits,
    shards_only=False,
):
    """Materialize deterministic lane files without requiring one giant pickle."""
    os.makedirs(out_dir, exist_ok=True)
    shards = deterministic_scenario_shards(scenario_names, num_splits)
    outputs = []
    for split_index, names in enumerate(shards):
        shard_data = load_scenario_chunks(chunks_dir, names)
        shard_path = os.path.join(
            out_dir,
            f'scenario_shard_{split_index:02d}_of_{num_splits:02d}.pkl',
        )
        with open(shard_path, 'wb') as stream:
            pickle.dump(shard_data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        outputs.append(
            {
                'index': split_index,
                'path': os.path.abspath(shard_path),
                'sha256': sha256_file(shard_path),
                'num_scenarios': len(shard_data),
                'first_scenario': names[0] if names else None,
                'last_scenario': names[-1] if names else None,
            }
        )

    all_path = None
    if not shards_only:
        all_scenarios = load_scenario_chunks(chunks_dir, sorted(scenario_names))
        all_path = os.path.join(out_dir, 'all_scenarios.pkl')
        with open(all_path, 'wb') as stream:
            pickle.dump(all_scenarios, stream, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        'schema_version': 1,
        'status': 'PASS',
        'method': 'deterministic_sorted_contiguous_scenario_shards_v1',
        'num_scenarios': len(scenario_names),
        'num_splits': num_splits,
        'shards_only': bool(shards_only),
        'all_scenarios_path': os.path.abspath(all_path) if all_path else None,
        'all_scenarios_sha256': sha256_file(all_path) if all_path else None,
        'shards': outputs,
    }
    manifest_path = os.path.join(out_dir, 'scenario_shards_manifest.json')
    with open(manifest_path, 'w') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return manifest


if __name__ == "__main__":
    args = parse_args()

    if args.num_splits < 1:
        raise ValueError('--num-splits must be positive')
    if args.expected_scenarios is not None and args.expected_scenarios < 1:
        raise ValueError('--expected-scenarios must be positive')

    out_dir = args.out_dir

    if isinstance(args.navsim_filters, str):
        navsim_filters = [args.navsim_filters]
    else:
        navsim_filters = args.navsim_filters

    # load navsim filters and collect selected tokens
    selected_tokens = []
    for navsim_config in navsim_filters:
        navsim_config = yaml.load(open(navsim_config, 'r'), Loader=yaml.FullLoader)
        selected_tokens.extend(navsim_config['tokens'])
    selected_tokens = set(selected_tokens)

    configs = Path(args.digitaltwin_asset_root) / "configs"
    configs = list(configs.glob("*.yaml"))

    filtered_video_scenes = []
    for config_path in configs:
        config = load_config(config_path.as_posix())
        config.central_tokens = [token for token in config.central_tokens if token in selected_tokens]
        if len(config.central_tokens) == 0:
            continue
        else:
            video_scene = VideoScene(config)
            video_scene.load_pickle(f"{args.digitaltwin_asset_root}/assets/{video_scene.name}/video_scene_dict.pkl")
            filtered_video_scenes.append(video_scene)

    print("Total tokens in navsim filters:", len(selected_tokens))
    print("Total video scenes:", len(configs))
    print("Total filtered video scenes:", len(filtered_video_scenes))

    # DEBUG: single process
    # all_scenarios = {}
    # for video_scene in filtered_video_scenes:
    #     scenario_names = create_digitaltwin_info_central(video_scene, args)
    #     print(f"Processed {len(scenario_names)} scenarios")

    # Create chunks directory
    chunks_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # Process scenes and save directly to disk.  nuPlan/SQLAlchemy/map objects can
    # deadlock after fork on some hosts, so num_processes=1 must be a genuinely
    # serial path rather than a one-worker multiprocessing.Pool.
    all_scenario_names = []
    if args.num_processes == 1:
        for video_scene in tqdm(
            filtered_video_scenes,
            total=len(filtered_video_scenes),
            desc="Processing Video Scenes (serial)",
        ):
            scenario_names = create_digitaltwin_info_central(video_scene, args=args)
            all_scenario_names.extend(scenario_names)
    else:
        with Pool(processes=args.num_processes) as pool:
            for scenario_names in tqdm(
                pool.imap_unordered(
                    partial(create_digitaltwin_info_central, args=args),
                    filtered_video_scenes,
                ),
                total=len(filtered_video_scenes),
                desc="Processing Video Scenes",
            ):
                all_scenario_names.extend(scenario_names)

    all_scenario_names = sorted(all_scenario_names)
    print(f"\nTotal scenarios processed: {len(all_scenario_names)}")
    if (
        args.expected_scenarios is not None
        and len(all_scenario_names) != args.expected_scenarios
    ):
        raise RuntimeError(
            f'converted {len(all_scenario_names)} scenarios; '
            f'expected {args.expected_scenarios}'
        )
    manifest = write_scenario_outputs(
        args.out_dir,
        chunks_dir,
        all_scenario_names,
        args.num_splits,
        shards_only=args.shards_only,
    )
    print(json.dumps(manifest, sort_keys=True))
