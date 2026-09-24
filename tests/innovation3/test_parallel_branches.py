"""Real spawned dynamics/reward parity and failure isolation, without CUDA."""
import copy
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
from shapely.geometry import box
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'projects/SimEngine'))
sys.path.insert(0,str(Path(__file__).parent))
from test_headless_snapshot import scene_fixture
from worldengine.online.headless import HeadlessSimulator,diagnostic_candidates
from worldengine.online.parallel_branches import ParallelBranchPool,parallel_reward_group
from worldengine.online.reward import capture_reward_frame,RewardHistory
from worldengine.online.reward_runtime import RewardSession
from worldengine.online.state import structural_hash
from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def pool_for(sim,scene,workers=2,reaction='R'):
 return ParallelBranchPool('fixture',scene,reaction,sim.engine.global_config.max_step,17,
   sim.codec.bundle,np.zeros(2),{k:v['type'] for k,v in scene['object_track'].items()},workers=workers,timeout=60,global_config=sim.engine.global_config)


@pytest.mark.parametrize('reaction',['NR','R'])
def test_parallel_matches_serial_complete_states_frames_and_repeated_groups(reaction):
 scene=scene_fixture();sim=HeadlessSimulator('fixture',scene,reaction,8,17)
 try:
  with pool_for(sim,scene,reaction=reaction) as pool:
   for _ in range(2):
    actions=diagnostic_candidates(sim); bank_hash=structural_hash(actions);before=sim.snapshot()
    _,main,serial=sim.branch_group(actions,12)
    frames=[]
    for snap,_ in serial:
     sim.restore(snap)
     frames.append(capture_reward_frame(sim.engine,np.zeros(2),{k:v['type'] for k,v in scene['object_track'].items()}).state_hash)
    sim.restore(main)
    parallel=pool.run(before,actions)
    assert [x['snapshot'].state_hash for x in parallel]==[x[0].state_hash for x in serial]
    assert [x['frame'].state_hash for x in parallel]==frames
    assert all(x['states']==serial[i][1] for i,x in enumerate(parallel))
    assert sim.snapshot().state_hash==main.state_hash
    assert structural_hash(actions)==bank_hash
    assert all(x['idm_fallbacks']==0 for x in parallel)
 finally:sim.close()


def reward_fixture():
 scene=scene_fixture();scene['log_length']=30
 for track in scene['object_track'].values():
  for key,arr in track['state'].items():
   track['state'][key]=np.concatenate([arr,np.repeat(arr[-1:],30-len(arr),axis=0)],axis=0)
 sim=HeadlessSimulator('fixture',scene,'R',22,17)
 session=RewardSession.__new__(RewardSession);session.sim=sim;session.offset=np.zeros(2)
 session.actor_types={k:v['type'] for k,v in scene['object_track'].items()}
 session.history=RewardHistory();session.history.append(session.capture())
 session.centerline=PDMPath([StateSE2(-100.,0.,0.),StateSE2(400.,0.,0.)])
 session.route_lanes={'lane':object()}
 session.map_api=PDMDrivableMap(['lane'],[SemanticMapLayer.LANE],[box(-100,-6,400,6)])
 for _ in range(13):session.warmup(diagnostic_candidates(sim)[12])
 return scene,sim,session


def test_full20_rewards_history_and_canonical_match_serial_then_failure_restores_main():
 scene,sim,session=reward_fixture()
 try:
  with pool_for(sim,scene) as pool, patch.object(PDMDrivableMap,'from_simulation',return_value=session.map_api):
   before=sim.snapshot();history=session.history.snapshot();actions=diagnostic_candidates(sim)
   original_capture=session.capture;frames=[]
   def recorded_capture():
    frame=original_capture();frames.append(frame.state_hash);return frame
   with patch.object(session,'capture',side_effect=recorded_capture):reference=session.group(actions,12)
   expected=sim.snapshot().state_hash
   sim.restore(before);session.history.restore(history)
   receipts=[]
   run=pool.run
   def checked_run(*args):
    assert len(receipts)==1  # Canonical receipt must precede any branch dispatch.
    return run(*args)
   with patch.object(pool,'run',side_effect=checked_run):
    result=parallel_reward_group(session,pool,actions,12,
       on_main=lambda snap,states,reward:receipts.append((snap.state_hash,reward)))
   assert receipts==[(reference['main_hash'],reference['main'])]
   assert all(result[k]==v for k,v in reference.items())
   assert result['frame_hashes']==frames[1:]
   assert sim.snapshot().state_hash==expected
   sim.restore(before);session.history.restore(history);digest=session.history.state_hash
   actions[0].waypoints[0,0]=np.nan
   children=list(pool._processes)
   with pytest.raises(RuntimeError,match='Branch worker failed'):parallel_reward_group(session,pool,actions,12)
   assert sim.snapshot().state_hash==expected and session.history.state_hash==digest
   assert all(not child.is_alive() for child in children)
 finally:sim.close()


def test_dead_worker_fails_without_hanging_or_changing_parent():
 scene=scene_fixture();sim=HeadlessSimulator('fixture',scene,'R',8,17)
 try:
  with pool_for(sim,scene,workers=1) as pool:
   before=sim.snapshot();pool._processes[0].terminate();pool._processes[0].join(5)
   with pytest.raises((EOFError,OSError,RuntimeError)):pool.run(before,diagnostic_candidates(sim))
   assert sim.snapshot().state_hash==before.state_hash and pool._closed
 finally:sim.close()


def test_observer_config_causes_old_restore_failure_and_is_preserved_in_workers():
 scene=scene_fixture();sim=HeadlessSimulator('fixture',scene,'R',8,17)
 try:
  original=OmegaConf.to_container(sim.engine.global_config,resolve=True)
  # Same assignments as CanonicalObserver.__init__, without initializing CUDA.
  sim.engine.global_config.asset_folder_path='/audited/assets'
  sim.engine.global_config.renderer=OmegaConf.load(
    ROOT/'projects/SimEngine/worldengine/configs/common/renderer/mtgs.yaml')
  changed=OmegaConf.to_container(sim.engine.global_config,resolve=True)
  assert {k for k in changed if k not in original or changed[k]!=original[k]}=={'asset_folder_path','renderer'}
  before=sim.snapshot();actions=diagnostic_candidates(sim)
  with ParallelBranchPool('fixture',scene,'R',8,17,sim.codec.bundle,np.zeros(2),
       {k:v['type'] for k,v in scene['object_track'].items()},workers=1,timeout=60) as old_pool:
   with pytest.raises(RuntimeError,match='Snapshot schema/config'):
    old_pool.run(before,actions)
   assert sim.snapshot().state_hash==before.state_hash
  with pool_for(sim,scene) as pool:
   assert all(r['config_hash']==before.config_hash and r['cuda_visible_devices']==''
              for r in pool.worker_receipts)
   _,main,serial=sim.branch_group(actions,12)
   parallel=pool.run(before,actions)
   assert [x['snapshot'].state_hash for x in parallel]==[x[0].state_hash for x in serial]
   assert sim.snapshot().state_hash==main.state_hash
   # Keep the strict rejection if config is changed after the pool is initialized.
   sim.engine.global_config.dt=0.25
   with pytest.raises(RuntimeError,match='Snapshot schema/config'):
    pool.run(sim.snapshot(),actions)
 finally:sim.close()
