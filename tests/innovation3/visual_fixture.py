import numpy as np


def raw_frame(step, scene='scene'):
    transform = np.eye(4)
    angle = .02*step
    transform[:2,:2] = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    transform[0,3] = 2*step
    bus = np.zeros(18); bus[10] = 4.; bus[7] = .1
    cameras = {}
    for i,name in enumerate(('CAM_F0','CAM_L0','CAM_R0','CAM_L1','CAM_R1','CAM_L2','CAM_R2','CAM_B0')):
        cameras[name] = dict(data_path=f'{scene}/{name}/{step}.jpg',
            sensor2lidar_rotation=np.eye(3), sensor2lidar_translation=np.array([.1*i,0.,0.]),
            cam_intrinsic=np.array([[100.,0.,32.],[0.,100.,24.],[0.,0.,1.]]), distortion=np.zeros(5))
    return dict(token=f'{scene}-{step}',frame_idx=step,timestamp=1000000+step*500000,
        log_name=scene,log_token=scene,scene_name=scene,scene_token=scene,lidar_path=None,
        sample_prev=None,sample_next=None,lidar2ego=np.eye(4),lidar2global=transform.copy(),
        ego2global=transform,ego2global_translation=transform[:3,3].copy(),
        ego2global_rotation=np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)]),
        cams=cameras,can_bus=bus,driving_command=np.array([0,1,0]),
        anns=dict(gt_boxes=np.zeros((0,7)),gt_velocity_3d=np.zeros((0,3)),gt_names=np.array([],dtype=str),track_tokens=[]))
