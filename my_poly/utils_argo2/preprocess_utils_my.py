'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import sys, os
import numpy as np
import json
import pickle
import multiprocessing
import glob
import pandas as pd
from tqdm.auto import tqdm
import warnings
from scipy.linalg import block_diag

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.map_api import ArgoverseStaticMap

from utils_common.helper_utils import OBJECT_TYPES, TRACK_CATEGORIES
import utils_argo2.tracking_utils as tracking_utils 
import utils_argo2.map_utils as map_utils 

sys.path.append('../cmotap/continental-multi-object-tracker-and-predictor/')
from src.cmotap.trajectory import Trajectory 
from src.cmotap.basisfunctions.bernsteinpolynomials import BernsteinPolynomials
from src.cmotap.motionmodels.trajectory_motion import TrajectoryMotion
from src.cmotap.observationmodels.trajectory_observation import TrajectoryObservation

from src.cmotap import utils

# Load prior parameters
with open('/home/zhaozishan/everything-polynomial/Argo2/priors_argo2/vehicle/vehicle_5s.json', "r") as read_file:
    prior_vechicle = json.load(read_file)
    
with open('/home/zhaozishan/everything-polynomial/Argo2/priors_argo2/cyclist/cyclist_5s.json', "r") as read_file:
    prior_cyclist = json.load(read_file)
    
with open('/home/zhaozishan/everything-polynomial/Argo2/priors_argo2/pedestrian/pedestrian_5s.json', "r") as read_file:
    prior_pedestrian = json.load(read_file)
    
with open('/home/zhaozishan/everything-polynomial/Argo2/priors_argo2/ego/ego_5s.json', "r") as read_file:
    prior_ego = json.load(read_file)


HIST_TIMESCALE = 4.9
HIST_DEGREE = 5 # According to AIC in https://arxiv.org/abs/2211.01696v4
PATH_DEGREE = 3 # Degree for lane segments and cross walks
SPACEDIM = 2
HIST_LEN = 50
MIN_OBS_LEN = 1

BASIS = BernsteinPolynomials(HIST_DEGREE)
TRAJ = Trajectory(basisfunctions=BASIS, spacedim=SPACEDIM, timescale=HIST_TIMESCALE)
                    
outlier_lane_dict = {}


def track_ego_trajectory(ego_traj_points, timestamps):
    """跟踪自车轨迹
    Args:
        ego_traj_points: 自车轨迹点 [N, 5] (x, y, heading, vx, vy)
                         N为观测点数，最大不超过HIST_LEN(50)
        timestamps: 时间戳序列 [N]，单位秒
    Returns:
        tuple: (控制点均值 [N_ctrl_pts,2], 控制点协方差 [N_ctrl_pts*2, N_ctrl_pts*2])
               N_ctrl_pts = BASIS.size (由HIST_DEGREE决定)
    """
    # 输入校验
    assert ego_traj_points.shape[0] <= HIST_LEN  # 确保不超过最大历史长度
    assert ego_traj_points.shape[0] == len(timestamps)  # 数据与时间戳对齐
    
    # 获取先验参数（历史轨迹和未来轨迹）
    FUTUREPRIOR, HISTORYPRIOR = tracking_utils.get_prior(
        degree=HIST_DEGREE, 
        timescale=HIST_TIMESCALE, 
        prior_data=prior_ego
    )

    # 过程噪声矩阵（控制点运动模型噪声）
    Q = np.kron(np.diag([0, 0, 0, .3, .2, .1])**2, np.eye(SPACEDIM))
    # 运动先验（大方差表示弱先验）
    motion_prior = np.eye(BASIS.size * SPACEDIM) * 200000  

    ego_state = None  # 自车状态（高斯控制点分布）
    last_timestamp = None  # 上一时刻时间戳
    
    # 获取位置观测噪声协方差
    pos_obs_cov = tracking_utils.build_ego_observation_noise_covariance(
        degree = HIST_DEGREE, 
        prior_data = prior_ego
    )
    
    # 逐帧处理轨迹点
    for i, (timestamp, ego_traj_point) in enumerate(zip(timestamps, ego_traj_points)):
        # 分解轨迹点信息
        ego_pos = ego_traj_point[:2]    # 位置 [x,y]
        ego_vel = ego_traj_point[-2:]   # 速度 [vx,vy]
        ego_heading = ego_traj_point[2] # 航向角（弧度）
        
        # 构建旋转矩阵（车身坐标系到世界坐标系）
        rot = np.array([
            [np.cos(ego_heading), -np.sin(ego_heading)],
            [np.sin(ego_heading), np.cos(ego_heading)]
        ])
        
        # 速度观测噪声（车身坐标系）
        vel_obs_cov = np.diag([0.5, 0.1])**2  # 纵向0.5m/s，横向0.1m/s
        vel_obs_cov = rot @ vel_obs_cov @ rot.T  # 转换到世界坐标系

        # 卡尔曼滤波流程
        if ego_state is not None:  # 非首帧执行预测+更新
            # 计算时间间隔
            dt = timestamp - last_timestamp
            
            # 1. 预测步骤（使用轨迹运动模型）
            MM = TrajectoryMotion(TRAJ, Q, Prior=motion_prior)
            predicted_ego_state = ego_state.predict(MM, dt)

            # 2. 更新步骤（融合观测数据）
            # 构建观测模型：位置+速度观测
            OM = TrajectoryObservation(
                TRAJ, 
                t=HIST_TIMESCALE, 
                derivatives=[[0,1]],  # 观测位置(0阶)和速度(1阶导数)
                R=[block_diag(pos_obs_cov, vel_obs_cov)]  # 组合观测噪声
            )
            # 执行卡尔曼更新
            ego_state = predicted_ego_state.update(
                np.concatenate([ego_pos, ego_vel]),  # 观测向量 [x,y,vx,vy]
                OM
            )
        else:  # 首帧初始化
            ego_state = tracking_utils.get_initial_state_ego(
                ego_pos, ego_vel, HISTORYPRIOR, 
                TRAJ, BASIS, HIST_TIMESCALE
            )
        
        last_timestamp = timestamp  # 更新时间戳
        
    # 返回最终状态（控制点均值和协方差）
    return (
        np.array(ego_state.x, dtype=np.float32).reshape(-1,2),  # 均值 [N_ctrl_pts,2]
        np.array(ego_state.P, dtype=np.float32)                 # 协方差 [2N_ctrl_pts,2N_ctrl_pts]
    )


def track_obj_trajectory(obj_traj_points, ego_traj_points, timestamps, object_type):
    assert len(timestamps) <= HIST_LEN
    assert len(timestamps) == obj_traj_points.shape[0] == ego_traj_points.shape[0]
    
    prior_data = None
    if object_type == 0: # vehicle
        prior_data = prior_vechicle
    elif object_type == 1: # pedestrian
        prior_data = prior_pedestrian
    elif object_type == 3: # cyclist
        prior_data = prior_cyclist
    elif object_type == 2: # motor_cyclist
        prior_data = prior_cyclist
    elif object_type == 8: # riderless_bicycle
        prior_data = prior_cyclist
    else: # TODO: what is the prior for unknown objects?
        prior_data = prior_vechicle
    
    FUTUREPRIOR, HISTORYPRIOR = tracking_utils.get_prior(degree=HIST_DEGREE, timescale=HIST_TIMESCALE, prior_data=prior_data)
    
    Q = np.kron(np.diag([0, 0, 0, .3, .2, .1])**2, np.eye(SPACEDIM))
    
    motion_prior = np.eye(BASIS.size * SPACEDIM) * 200000 # uninformative prior for motion  
    obj_state = None
    last_timestamp = None    
    
    
    pos_obs_cov_temp = tracking_utils.build_obj_observation_noise_covariance_batch(ego_pos=ego_traj_points[:,:2], 
                                                                                   ego_heading = ego_traj_points[:, 2], 
                                                                                   obj_pos = obj_traj_points[:,:2], 
                                                                                   degree= HIST_DEGREE, 
                                                                                   prior_data= prior_data)
    
    for i, (timestamp, ego_traj_point, obj_traj_point) in enumerate(zip(timestamps, ego_traj_points, obj_traj_points)):        
        ego_pos = ego_traj_point[:2]
        ego_psi = ego_traj_point[2]
        obj_pos = obj_traj_point[:2]
        obj_vel = obj_traj_point[-2:]      
        obj_heading = obj_traj_point[2]
        
        vel_obs_cov = np.diag([3, 3])**2
        
        if obj_state is not None:
            # Prediction Step
            dt = timestamp - last_timestamp

            MM = TrajectoryMotion(TRAJ, Q, Prior=motion_prior)

            predicted_obj_state = obj_state.predict(MM, dt)

            # Update Step
            pos_obs_cov = pos_obs_cov_temp[i] #tracking_utils.build_obj_observation_noise_covariance(ego_pos, ego_psi, obj_pos, degree= HIST_DEGREE, prior_data= prior_data)
             
            if i == obj_traj_points.shape[0] -1: #assume less observation noise for the last point
                pos_obs_cov = pos_obs_cov / 9.
                
            OM = TrajectoryObservation(TRAJ, t=HIST_TIMESCALE, derivatives=[[0,1]], R=[block_diag(pos_obs_cov, vel_obs_cov)])
            obj_state = predicted_obj_state.update(np.concatenate([obj_pos, obj_vel]), OM)
        else:
            obj_state = tracking_utils.get_initial_state_agt(obj_pos, obj_vel, HISTORYPRIOR, TRAJ, BASIS, HIST_TIMESCALE)
        
        last_timestamp = timestamp
    
    return np.array(obj_state.x).reshape(-1,2),  np.array(obj_state.P)

def process_argo2_data_with_scenario_parquet(src_file, 
                                             output_path=None, 
                                             process_track = True,
                                             process_map = True):


    # 读取pkl文件 没有读取成功  '/data1/data_zzs/dataset_unitraj_split/AG2_train/AG2_train_0_tmp/sd_av2_v2_00a0a3e0-1508-45f2-9cf5-e427e1446a33.pkl/'
    # 读取成功  '/data1/data_zzs/dataset_unitraj_split/AG2_train/AG2_train_0_tmp/sd_av2_v2_00a0a3e0-1508-45f2-9cf5-e427e1446a33.pkl/sd_av2_v2_00a0a3e0-1508-45f2-9cf5-e427e1446a33.pkl'
    with open(src_file, 'rb') as f:
        scenario_data = pickle.load(f)

    focal_track_id = str(list(scenario_data['metadata']['tracks_to_predict'].keys())[0])
    scenario_id=scenario_data['metadata']['scenario_id']
    city_name=scenario_data['metadata']['scenario_id']


    # map_file = glob.glob(os.path.join(src_file, '*.json*'))[0]
    
    #bool_test_data= 'test' in src_file

    # scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_file)
    # mi = map_utils.MapInterpreter(src_file, path_degree = PATH_DEGREE)
    
    file_infos = {
        'scenario_id': scenario_id,
        # 'map_id': scenario.map_id,
        'focal_track_id': focal_track_id,
        'slice_id': None,
        'city_name': city_name}
    
    if output_path is not None:
        os.makedirs(os.path.join(output_path, f'{scenario_id}'), exist_ok=True)
    timestamps = scenario_data['metadata']['ts']
    # 获取tracks_to_predict中第一个键
    first_key = list(scenario_data['metadata']['tracks_to_predict'].keys())[0]

    # 提取历史时间步的数据

   
    historical_df = {key: {k: v[:50] for k, v in track['state'].items()} for key, track in scenario_data['tracks'].items()}

    historical_df = scenario_data['tracks'][first_key]['state']['valid']
    valid_count = sum(historical_df)
    is_valid = valid_count < HIST_LEN

    #筛选筛选历史时间步（前50帧）的数据
    historical_df= scenario_data['metadata']['tracks_to_predict'][first_key]['valid'] + [False] * (HIST_LEN - valid_count)

    historical_df = df[df['timestep'] < HIST_LEN] ## 筛选历史时间步（前50帧）的数据
    timesteps = list(np.sort(df['timestep'].unique())) ## 获取所有不重复的时间步并排序
    #获取所有参与者的ID
    actor_ids = list(historical_df['track_id'].unique())
    #过滤掉观测次数小于MIN_OBS_LEN的参与者
    actor_ids = list(filter(lambda actor_id: np.sum(historical_df[historical_df['track_id'] == actor_id]['observed'])>=MIN_OBS_LEN, actor_ids))
    # 筛选出在actor_ids中的参与者
    historical_df = historical_df[historical_df['track_id'].isin(actor_ids)]
 
    df = df[df['track_id'].isin(actor_ids)]
    
    # DataFrame for AV ang Agent
    av_df = df[df['track_id'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['track_id'])
    agt_df = df[df['track_id'] == scenario.focal_track_id].iloc
    agent_index = actor_ids.index(agt_df[0]['track_id'])
    
    
    num_actors = len(actor_ids)
    timestep_mask = np.zeros((num_actors, 110), dtype=bool) # booleans indicate if object is observed at each timestamp
    time_window = np.zeros((num_actors, 2), dtype=float) # start and end timestamps for the control points
    objects_type = np.zeros((num_actors), dtype=int)
    tracks_category = np.zeros((num_actors), dtype=int)
    x = np.zeros((num_actors, 110, 5), dtype=float) # [x, y, heading, vx, vy]
    x_mean = np.zeros((num_actors, HIST_DEGREE+1, SPACEDIM), dtype=float) 
    x_cov = np.zeros((num_actors, (HIST_DEGREE+1) * SPACEDIM, (HIST_DEGREE+1) * SPACEDIM), dtype=float)
    agent_id = [None] * num_actors
    
    # make the scene centered at AGT
    origin = np.array([agt_df[HIST_LEN-1]['position_x'], agt_df[HIST_LEN-1]['position_y']])
    theta = np.array(agt_df[HIST_LEN-1]['heading'])
    rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    R_mat = np.kron(np.eye(HIST_DEGREE+1), rotate_mat)
    
    ego_positions = np.array([av_df[:HIST_LEN]['position_x'].values, av_df[:HIST_LEN]['position_y'].values]).T
    ego_headings = np.array(av_df[:HIST_LEN]['heading'].values)
    ego_velocities = np.array([av_df[:HIST_LEN]['velocity_x'].values, av_df[:HIST_LEN]['velocity_y'].values]).T
    
    ego_traj = np.concatenate([ego_positions, ego_headings[:, None], ego_velocities], axis=1) # This is raw data
    
    obj_trajs = []
    
    if process_track:
        av_last_fit_error = None
        agent_last_fit_error = None

        for actor_id, actor_df in df.groupby('track_id'):
            actor_idx = actor_ids.index(actor_id)
            agent_id[actor_idx] = actor_id
            actor_hist_steps = [timesteps.index(timestep) for timestep in historical_df[historical_df['track_id']==actor_id]['timestep']]
            actor_steps = [timesteps.index(timestep) for timestep in df[df['track_id'] == actor_id]['timestep']]
            timestep_mask[actor_idx, actor_steps] = True


            objects_type[actor_idx] = OBJECT_TYPES.index(actor_df['object_type'].unique()[0])
            tracks_category[actor_idx] = actor_df['object_category'].unique()[0]
            
            positions = np.array([actor_df[:]['position_x'].values, actor_df[:]['position_y'].values]).T
            headings = np.array(actor_df[:]['heading'].values)
            velocities = np.array([actor_df['velocity_x'].values, actor_df['velocity_y'].values]).T
            
            obj_traj = np.concatenate([positions, headings[:, None], velocities], axis=1) # This is raw data
            obj_trajs.append(obj_traj)
            

            x[actor_idx, actor_steps, :2] = (positions - origin) @ rotate_mat
            x[actor_idx, actor_steps, 2] = headings - theta
            x[actor_idx, actor_steps, 3:5] = velocities @ rotate_mat

            T_hist = timestamps[actor_hist_steps]
            time_window[actor_idx] = np.array([np.min(T_hist), np.max(T_hist)])

            if actor_id == 'AV':
                cps_mean, cps_cov = track_ego_trajectory(obj_traj[np.where(np.array(actor_hist_steps) < HIST_LEN)], 
                                                         T_hist)
            else:                
                cps_mean, cps_cov = track_obj_trajectory(obj_traj[np.where(np.array(actor_hist_steps) < HIST_LEN)], 
                                                         ego_traj[actor_hist_steps],  
                                                         T_hist, 
                                                         objects_type[actor_idx])


            x_mean[actor_idx] = (cps_mean - origin) @ rotate_mat
            x_cov[actor_idx] = R_mat.T @ cps_cov @ R_mat

            if actor_id == 'AV':
                av_last_fit_error = np.linalg.norm(x_mean[actor_idx, -1] - x[actor_idx, HIST_LEN -1, :2], axis = -1)
            elif actor_id == scenario.focal_track_id:
                agent_last_fit_error = np.linalg.norm(x_mean[actor_idx, -1] - x[actor_idx, HIST_LEN-1, :2], axis = -1)



        track_infos = {
                    'object_type': objects_type, # [N]
                    'track_category': tracks_category, # [N]
                    'timestamps_seconds': timestamps, # [110]
                    'x': x[:, :HIST_LEN], # [N, 50, 5]
                    'y': None if bool_test_data else x[:, HIST_LEN:], # [N, 60, 5]
                    'cps_mean': x_mean, # [N, 6, 2]
                    'cps_cov': x_cov, # [N, 12, 12]
                    'timestep_x_mask': timestep_mask[:, :50], #[N, 50]
                    'timestep_y_mask': timestep_mask[:, 50:], #[N, 60]
                    'time_window': time_window, # [N, 2]
                    'av_index': av_index,
                    'agent_index': agent_index,
                    'agent_ids': agent_id, # [N]
                    'origin': origin, # [2]
                    'ro_mat': rotate_mat, # [2, 2]
                    'av_fit_error': av_last_fit_error,
                    'agent_fit_error': agent_last_fit_error,
                    'num_objects': len(actor_ids),
                    'HIST_DEG': HIST_DEGREE
                   }
    
        track_infos.update(file_infos)
        track_output_file = os.path.join(output_path, f'{scenario.scenario_id}', f'{scenario.scenario_id}_track_infos.pkl')
        with open(track_output_file, 'wb') as f:
            pickle.dump(track_infos, f)
    
    
    if process_map:
        map_infos, outlier_lane_ids =  mi.get_map_features(origin, rotate_mat, city_map = global_map[scenario.city_name])
        
        if len(outlier_lane_ids) > 0:
            #warnings.warn("Ourlier Lane with {} in Scenario {}".format(outlier_lane_ids, scenario.scenario_id))
            outlier_lane_dict[scenario.scenario_id] = outlier_lane_ids
        
        map_infos.update(file_infos)
        map_output_file = os.path.join(output_path, f'{scenario.scenario_id}', f'{scenario.scenario_id}_map_infos.pkl')
    
        with open(map_output_file, 'wb') as f:
            pickle.dump(map_infos, f)
    
    return [len(actor_ids), 
            len(mi.all_lane_ids),
            len(list(mi.avm.vector_pedestrian_crossings.keys())) * 2
           ]

def process_argo2_data_with_scenario_parquet_my(src_file, output_path=None, process_track=True, process_map=True):
    """处理单个场景数据
    参数:
        src_file: 源数据目录
        output_path: 输出路径
        process_track: 是否处理轨迹数据
        process_map: 是否处理地图数据
    """
    # 加载场景文件
    scenario_file = glob.glob(os.path.join(src_file, '*.parquet*'))[0]  # 场景数据
    # map_file = glob.glob(os.path.join(src_file, '*.json*'))[0]          # 地图数据
    bool_test_data = 'test' in src_file  # 是否为测试数据

    # 初始化场景和地图解释器
    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_file)
    mi = map_utils.MapInterpreter(src_file, path_degree=PATH_DEGREE)

    # # 基础场景信息
    # file_infos = {
    #     'scenario_id': scenario.scenario_id,
    #     'map_id': scenario.map_id,
    #     'focal_track_id': scenario.focal_track_id,
    #     'slice_id': scenario.slice_id,
    #     'city_name': scenario.city_name
    # }

    # 创建输出目录
    if output_path is not None:
        os.makedirs(os.path.join(output_path, f'{scenario.scenario_id}'), exist_ok=True)
    
    # 时间戳处理：转换为秒，并将起始时间归零
    timestamps = scenario.timestamps_ns/ 1e9  # 转换为秒
    timestamps = timestamps - timestamps[0]    # 形状: (110,) - 包含50帧历史数据和60帧未来数据
    
    df = pd.read_parquet(scenario_file)
    historical_df = df[df['timestep'] < HIST_LEN]  # 历史数据（前50帧）
    timesteps = list(np.sort(df['timestep'].unique()))

    actor_ids = list(historical_df['track_id'].unique())
    actor_ids = list(filter(lambda actor_id: np.sum(historical_df[historical_df['track_id'] == actor_id]['observed'])>=MIN_OBS_LEN, actor_ids))
    historical_df = historical_df[historical_df['track_id'].isin(actor_ids)]
    df = df[df['track_id'].isin(actor_ids)]
    
    # DataFrame for AV ang Agent
    av_df = df[df['track_id'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['track_id'])
    agt_df = df[df['track_id'] == scenario.focal_track_id].iloc
    agent_index = actor_ids.index(agt_df[0]['track_id'])
    
    
    num_actors = len(actor_ids)
    timestep_mask = np.zeros((num_actors, 110), dtype=bool)  # 形状:(N, 110) - 标记每个时间步是否有观测值
    time_window = np.zeros((num_actors, 2), dtype=float)     # 形状:(N, 2) - 每个物体的起止时间戳
    objects_type = np.zeros((num_actors), dtype=int)         # 形状:(N,) - 物体类型编号
    tracks_category = np.zeros((num_actors), dtype=int)      # 形状:(N,) - 轨迹类别编号
    x = np.zeros((num_actors, 110, 5), dtype=float)         # 形状:(N, 110, 5) - [x坐标, y坐标, 朝向角, x方向速度, y方向速度]
    x_mean = np.zeros((num_actors, HIST_DEGREE+1, SPACEDIM), dtype=float)  # 形状:(N, 6, 2) - 控制点均值
    x_cov = np.zeros((num_actors, (HIST_DEGREE+1) * SPACEDIM, (HIST_DEGREE+1) * SPACEDIM), dtype=float)  # 形状:(N, 12, 12) - 控制点协方差
    agent_id = [None] * num_actors  # 长度:N - 存储每个物体的ID
    
    # 将场景中心设置为目标车辆(AGT)在历史轨迹最后一帧的位置
    origin = np.array([agt_df[HIST_LEN-1]['position_x'], agt_df[HIST_LEN-1]['position_y']])  # 形状:(2,)
    theta = np.array(agt_df[HIST_LEN-1]['heading'])  # 标量 - 目标车辆朝向角
    rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])  # 形状:(2, 2) - 旋转矩阵
    R_mat = np.kron(np.eye(HIST_DEGREE+1), rotate_mat)     # 形状:(12, 12) - 扩展旋转矩阵
    
    # 提取自车(AV)轨迹数据
    ego_positions = np.array([av_df[:HIST_LEN]['position_x'].values, av_df[:HIST_LEN]['position_y'].values]).T  # 形状:(50, 2)
    ego_headings = np.array(av_df[:HIST_LEN]['heading'].values)  # 形状:(50,)
    ego_velocities = np.array([av_df[:HIST_LEN]['velocity_x'].values, av_df[:HIST_LEN]['velocity_y'].values]).T  # 形状:(50, 2)
    
    # 合并自车完整状态信息
    ego_traj = np.concatenate([ego_positions, ego_headings[:, None], ego_velocities], axis=1)  # 形状:(50, 5) - 原始数据
    
    obj_trajs = [] # 存储原始轨迹数据的列表
    
    if process_track:
        av_last_fit_error = None  # 自动驾驶车辆最后一帧拟合误差
        agent_last_fit_error = None  # 目标智能体最后一帧拟合误差

        # 遍历每个参与者的轨迹数据
        for actor_id, actor_df in df.groupby('track_id'):
            actor_idx = actor_ids.index(actor_id)
            agent_id[actor_idx] = actor_id
            # 获取历史时间步索引
            actor_hist_steps = [timesteps.index(timestep) for timestep in historical_df[historical_df['track_id']==actor_id]['timestep']]
            # 获取所有时间步索引
            actor_steps = [timesteps.index(timestep) for timestep in df[df['track_id'] == actor_id]['timestep']]
            timestep_mask[actor_idx, actor_steps] = True

            # 记录参与者类型和类别
            objects_type[actor_idx] = OBJECT_TYPES.index(actor_df['object_type'].unique()[0])
            tracks_category[actor_idx] = actor_df['object_category'].unique()[0]
            
            # 提取位置、朝向和速度信息
            positions = np.array([actor_df[:]['position_x'].values, actor_df[:]['position_y'].values]).T  # 形状: [T, 2]
            headings = np.array(actor_df[:]['heading'].values)  # 形状: [T]
            velocities = np.array([actor_df['velocity_x'].values, actor_df['velocity_y'].values]).T  # 形状: [T, 2]
            
            # 合并原始轨迹数据 [T, 5]: (x, y, heading, vx, vy)
            obj_traj = np.concatenate([positions, headings[:, None], velocities], axis=1)
            obj_trajs.append(obj_traj)
            
            # 坐标变换并存储处理后的轨迹数据
            x[actor_idx, actor_steps, :2] = (positions - origin) @ rotate_mat  # 位置坐标变换
            x[actor_idx, actor_steps, 2] = headings - theta  # 朝向角度变换
            x[actor_idx, actor_steps, 3:5] = velocities @ rotate_mat  # 速度向量变换

            # 记录时间窗口信息
            T_hist = timestamps[actor_hist_steps]
            time_window[actor_idx] = np.array([np.min(T_hist), np.max(T_hist)])

            # 根据参与者类型选择不同的轨迹拟合方法
            if actor_id == 'AV':
                # 自动驾驶车辆轨迹拟合
                cps_mean, cps_cov = track_ego_trajectory(obj_traj[np.where(np.array(actor_hist_steps) < HIST_LEN)], 
                                                         T_hist)
            else:                
                # 其他参与者轨迹拟合
                cps_mean, cps_cov = track_obj_trajectory(obj_traj[np.where(np.array(actor_hist_steps) < HIST_LEN)], 
                                                         ego_traj[actor_hist_steps],  
                                                         T_hist, 
                                                         objects_type[actor_idx])

            # 控制点坐标变换
            x_mean[actor_idx] = (cps_mean - origin) @ rotate_mat  # 形状: [6, 2]
            x_cov[actor_idx] = R_mat.T @ cps_cov @ R_mat  # 形状: [12, 12]

            # 计算最后一帧拟合误差
            if actor_id == 'AV':
                av_last_fit_error = np.linalg.norm(x_mean[actor_idx, -1] - x[actor_idx, HIST_LEN -1, :2], axis = -1)
            elif actor_id == scenario.focal_track_id:
                agent_last_fit_error = np.linalg.norm(x_mean[actor_idx, -1] - x[actor_idx, HIST_LEN-1, :2], axis = -1)

        # 整理轨迹信息字典
        track_infos = {
                    'object_type': objects_type,         # 形状:[N] - 物体类型
                    'track_category': tracks_category,   # 形状:[N] - 轨迹类别
                    'timestamps_seconds': timestamps,     # 形状:[110] - 时间戳序列
                    'x': x[:, :HIST_LEN],               # 形状:[N, 50, 5] - 历史轨迹数据
                    'y': None if bool_test_data else x[:, HIST_LEN:],  # 形状:[N, 60, 5] - 未来轨迹数据(测试集为None)
                    'cps_mean': x_mean,                 # 形状:[N, 6, 2] - 控制点均值
                    'cps_cov': x_cov,                   # 形状:[N, 12, 12] - 控制点协方差
                    'timestep_x_mask': timestep_mask[:, :50],  # 形状:[N, 50] - 历史数据观测掩码
                    'timestep_y_mask': timestep_mask[:, 50:],  # 形状:[N, 60] - 未来数据观测掩码
                    'time_window': time_window,         # 形状:[N, 2] - 时间窗口范围
                    'av_index': av_index,              # 自动驾驶车辆索引
                    'agent_index': agent_index,        # 目标智能体索引
                    'agent_ids': agent_id,             # 形状:[N] - 参与者ID列表
                    'origin': origin,                  # 形状:[2] - 场景原点坐标
                    'ro_mat': rotate_mat,              # 形状:[2, 2] - 旋转矩阵
                    'av_fit_error': av_last_fit_error, # 自动驾驶车辆拟合误差
                    'agent_fit_error': agent_last_fit_error, # 目标智能体拟合误差
                    'num_objects': len(actor_ids),     # 场景中参与者总数
                    'HIST_DEG': HIST_DEGREE           # 历史轨迹拟合多项式阶数
                   }
    
        track_infos.update(file_infos)
        track_output_file = os.path.join(output_path, f'{scenario.scenario_id}', f'{scenario.scenario_id}_track_infos.pkl')
        with open(track_output_file, 'wb') as f:
            pickle.dump(track_infos, f)
    
    
    if process_map:
        # 处理地图信息
        map_infos, outlier_lane_ids =  mi.get_map_features(origin, rotate_mat, city_map = global_map[scenario.city_name])
        
        if len(outlier_lane_ids) > 0:
            outlier_lane_dict[scenario.scenario_id] = outlier_lane_ids
        
        map_infos.update(file_infos)
        map_output_file = os.path.join(output_path, f'{scenario.scenario_id}', f'{scenario.scenario_id}_map_infos.pkl')
    
        with open(map_output_file, 'wb') as f:
            pickle.dump(map_infos, f)
    
    # 返回场景统计信息：[参与者数量, 车道数量, 人行横道数量*2]
    return [len(actor_ids), 
            len(mi.all_lane_ids),
            len(list(mi.avm.vector_pedestrian_crossings.keys())) * 2
           ]

        
def get_infos_from_data(data_path, 
                        output_path=None, 
                        num_workers=16,
                        process_track = True,
                        process_map = True):
    from functools import partial
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)


    #src_files = glob.glob(data_path + "/*/", recursive = True)
    src_files = ['/data1/data_zzs/dataset_unitraj_split/AG2_train/AG2_train_0_tmp/sd_av2_v2_00a0a3e0-1508-45f2-9cf5-e427e1446a33.pkl']
    src_files.sort() 
    results, error_files= [], []

    for src_file in tqdm(src_files):
        try:
            result = process_argo2_data_with_scenario_parquet(src_file, output_path, process_track = process_track, process_map = process_map)
            results.append(result)
        except:
            error_files.append(src_file)
            warnings.warn("Error with " + src_file)
            continue
    
    results = np.max(np.array(results, dtype = int), axis = 0)
    
    print("Max Obj {}, Max Lane {}, Max CW {}".format(*results))
    
    return results


def process_single_pkl(pkl_path, output_path=None, process_track=True, process_map=True):
    """处理单个pkl文件
    
    Args:
        pkl_path: pkl文件的完整路径
        output_path: 输出路径，如果为None则不保存处理结果
        process_track: 是否处理轨迹数据
        process_map: 是否处理地图数据
    
    Returns:
        result: 处理结果，包含最大目标数、最大车道数和最大人行道数
    """
    try:
        # 获取文件所在目录
        src_file = os.path.dirname(pkl_path) 
        result = process_argo2_data_with_scenario_parquet(
            src_file, 
            output_path,
            process_track=process_track,
            process_map=process_map
        )
        print("处理成功：", pkl_path)
        return result
    except Exception as e:
        print(f"处理文件 {pkl_path} 时出错：{str(e)}")
        return None



def create_infos_from_data_my(raw_data_path,      # 原始数据存储路径
                           output_path,        # 处理后的输出路径
                           splits,             # 数据集划分类型 (train/val/test等)
                           num_workers=16,     # 多进程工作线程数
                           process_track=True, # 是否处理轨迹数据
                           process_map=True,   # 是否处理地图数据
                           global_map_path=None): # 全局地图缓存路径
    if not isinstance(splits, list):
        splits = [splits]  # 统一转换为列表格式处理
    
    global global_map  # 声明使用全局地图变量
    # 初始化全局地图（用于加速地图处理）
    if global_map_path is None:
        try:
            # 尝试加载预处理的全局地图缓存
            with open('data/global_map_A2.pkl', 'rb') as f: 
                global_map = pickle.load(f)
            print('发现已保存的A2全局地图')
        except:
            # 初始化各城市空地图结构（包含Argo2的6个城市）
            print('未找到保存的A2全局地图，初始化新地图')
            global_map = {'austin': {},       # 奥斯汀
                          'washington-dc': {}, # 华盛顿特区 
                          'pittsburgh': {},    # 匹兹堡
                          'palo-alto': {},     # 帕罗奥图
                          'dearborn': {},      # 迪尔伯恩
                          'miami': {}}         # 迈阿密
    else:
        # 加载指定的全局地图文件
        with open(global_map_path, 'rb') as f: 
            global_map = pickle.load(f)
        print('发现已保存的A2全局地图')
    
    # 遍历处理每个数据划分
    for split in splits:
        print('---------------- 正在预处理: ' + split + ' ----------------')
        # 核心处理函数：多进程处理原始数据
 
        data_infos = process_single_pkl(
            pkl_path=os.path.join(raw_data_path, split),  # 原始数据路径
            output_path=os.path.join(output_path, split + '_processed'),  # 处理结果输出路径
            process_track=process_track, # 是否处理轨迹数据
            process_map=process_map      # 是否处理地图数据
        )
        # 保存处理后的场景信息
        filename = os.path.join(output_path, 'processed_scenarios_' + split + '_infos.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(data_infos, f)  # 序列化存储场景信息
        
        # 保存异常车道数据（用于后续分析）
        filename = os.path.join(output_path, 'outlier_lane_dict_' + split + '.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(outlier_lane_dict, f)  # 存储异常车道字典
        
        # 保存更新后的全局地图（如果未指定路径则使用默认路径）
        filename = os.path.join(output_path, 'global_map_A2.pkl') if global_map_path is None else global_map_path
        with open(filename, 'wb') as f:
            pickle.dump(global_map, f)  # 持久化存储全局地图
        
        print('----------------Argo2 %s 数据集信息已保存至 %s----------------' % (split, filename))


def create_infos_from_data(raw_data_path,      # 原始数据存储路径
                           output_path,        # 处理后的输出路径
                           splits,             # 数据集划分类型 (train/val/test等)
                           num_workers=16,     # 多进程工作线程数
                           process_track=True, # 是否处理轨迹数据
                           process_map=True,   # 是否处理地图数据
                           global_map_path=None): # 全局地图缓存路径
    if not isinstance(splits, list):
        splits = [splits]  # 统一转换为列表格式处理
    
    global global_map  # 声明使用全局地图变量
    # 初始化全局地图（用于加速地图处理）
    if global_map_path is None:
        try:
            # 尝试加载预处理的全局地图缓存
            with open('data/global_map_A2.pkl', 'rb') as f: 
                global_map = pickle.load(f)
            print('发现已保存的A2全局地图')
        except:
            # 初始化各城市空地图结构（包含Argo2的6个城市）
            print('未找到保存的A2全局地图，初始化新地图')
            global_map = {'austin': {},       # 奥斯汀
                          'washington-dc': {}, # 华盛顿特区 
                          'pittsburgh': {},    # 匹兹堡
                          'palo-alto': {},     # 帕罗奥图
                          'dearborn': {},      # 迪尔伯恩
                          'miami': {}}         # 迈阿密
    else:
        # 加载指定的全局地图文件
        with open(global_map_path, 'rb') as f: 
            global_map = pickle.load(f)
        print('发现已保存的A2全局地图')
    
    # 遍历处理每个数据划分
    for split in splits:
        print('---------------- 正在预处理: ' + split + ' ----------------')
        # 核心处理函数：多进程处理原始数据

        data_infos = get_infos_from_data(
            data_path=os.path.join(raw_data_path, split),  # 原始数据路径
            output_path=os.path.join(output_path, split + '_processed'),  # 处理结果输出路径
            num_workers=num_workers,     # 并行工作进程数
            process_track=process_track, # 是否处理轨迹数据
            process_map=process_map      # 是否处理地图数据
        )
        
        # 保存处理后的场景信息
        filename = os.path.join(output_path, 'processed_scenarios_' + split + '_infos.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(data_infos, f)  # 序列化存储场景信息
        
        # 保存异常车道数据（用于后续分析）
        filename = os.path.join(output_path, 'outlier_lane_dict_' + split + '.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(outlier_lane_dict, f)  # 存储异常车道字典
        
        # 保存更新后的全局地图（如果未指定路径则使用默认路径）
        filename = os.path.join(output_path, 'global_map_A2.pkl') if global_map_path is None else global_map_path
        with open(filename, 'wb') as f:
            pickle.dump(global_map, f)  # 持久化存储全局地图
        
        print('----------------Argo2 %s 数据集信息已保存至 %s----------------' % (split, filename))


