'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue

数据编码：
列表操作：
数据转换：
对象车道类型的转换：
多项式基函数：
'''

import json
import numpy as np
import pandas as pd
##import tensorflow as tf
import torch  # 使用 PyTorch 替换 TensorFlow

# 自定义 JSON 编码器，用于保存 numpy 和 PyTorch 数据
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 将 numpy 数组转换为列表
        if isinstance(obj, np.float32):
            return float(obj)  # 将 float32 转换为 float
        if isinstance(obj, np.int16):
            return int(obj)  # 将 int16 转换为 int
        if isinstance(obj, np.int64):
            return int(obj)  # 将 int64 转换为 int
        # if isinstance(obj, tf.Tensor):
        #     return float(obj)  # 将 TensorFlow 张量转换为 float
        if isinstance(obj, torch.Tensor):  # 检查是否为 PyTorch 张量
            return float(obj)  # 将 PyTorch 张量转换为 Python 标量
        return json.JSONEncoder.default(self, obj)  # 默认处理

# 计算两个列表的交集
def list_intersection(l1, l2):
    return list(set(l1) & set(l2))

# 根据给定的索引切片列表
def list_slice(l, indicies):
    assert isinstance(l, list)  # 确保 l 是列表
    return [l[i] for i in indicies]  # 返回切片后的列表

# 将 Waymo protobuf 数据转换为 Pandas DataFrame
def waymo_protobuf_to_dataframe(scenario, hist_len, min_obs_len):
    # 将场景数据转换为 DataFrame
    data_list = [
        {
            'track_id': track.id,
            'object_type': track.object_type,
            'valid': state.valid,
            'position_x': state.center_x,
            'position_y': state.center_y,
            'heading': state.heading,
            'velocity_x': state.velocity_x,
            'velocity_y': state.velocity_y,
            'timestep': timestep,
        }
        for track in scenario.tracks for (timestep, state) in enumerate(track.states)
    ]

    df = pd.DataFrame(data_list)  # 创建 DataFrame
    historical_df = df[df['timestep'] < hist_len]  # 过滤历史数据
    timesteps = list(np.sort(df['timestep'].unique()))  # 获取唯一的时间步

    # 过滤有效的 actor_id
    actor_ids = list(historical_df['track_id'].unique())
    actor_ids = list(filter(lambda actor_id: np.sum(historical_df[historical_df['track_id'] == actor_id]['valid']) >= min_obs_len, actor_ids))
    historical_df = historical_df[historical_df['track_id'].isin(actor_ids)]
    df = df[df['track_id'].isin(actor_ids)]

    av_id = scenario.tracks[scenario.sdc_track_index].id  # 获取自动驾驶车辆的 ID
    to_predict_agt_id = [scenario.tracks[tracks_to_predict.track_index].id for tracks_to_predict in scenario.tracks_to_predict if tracks_to_predict.track_index != scenario.sdc_track_index]
    
    if len(to_predict_agt_id) == 0:
        return {}, False  # 如果没有需要预测的代理，返回空字典和 False

    # 计算观察到的步骤
    observed_steps = np.sum(np.array([(df[df['track_id'] == agt_id].iloc)[hist_len-1:]['valid'] for agt_id in to_predict_agt_id]), axis=1)
    valid_idx = np.where(observed_steps == 42)[0]  # 应该在当前时间和未来都是有效的
    if len(valid_idx) == 0:  # 如果没有有效的焦点代理
        return {}, False

    focal_track_id = np.array(to_predict_agt_id)[valid_idx][0]  # 获取焦点轨迹 ID
    scored_track_id = [agt_id for agt_id in to_predict_agt_id if agt_id != focal_track_id]  # 获取其他代理 ID

    # 为自动驾驶车辆和代理创建 DataFrame
    av_df = df[df['track_id'] == av_id].iloc
    av_index = actor_ids.index(av_df[0]['track_id'])
    agt_df = df[df['track_id'] == focal_track_id].iloc
    agent_index = actor_ids.index(agt_df[0]['track_id'])

    return {
        'df': df,
        'av_df': av_df,
        'agt_df': agt_df,
        'historical_df': historical_df,
        'actor_ids': actor_ids,
        'av_id': av_id,
        'focal_track_id': focal_track_id,
        'scored_track_id': scored_track_id,
        'av_index': av_index,
        'agent_index': agent_index,
    }, True  # 返回数据和 True

# 将 Waymo 对象类型转换为 Argo2 对象类型
def waymo_object_type_converter(waymo_obj_type):
    '''
        将 Waymo 对象类型转换为 Argo2 对象类型
    '''
    if waymo_obj_type == 0:  # 未设置
        return 9  # 未知
    elif waymo_obj_type == 1:  # 车辆
        return 0  # 车辆
    elif waymo_obj_type == 2:  # 行人
        return 1  # 行人
    elif waymo_obj_type == 3:  # 骑行者
        return 3  # 骑行者
    else:  # 其他
        return 9  # 未知

# 将 Waymo 车道类型转换为 Argo2 车道类型
def waymo_lane_type_converter(waymo_lane_type):
    '''
        将 Waymo 车道类型转换为 Argo2 车道类型
    '''
    if waymo_lane_type == 0:  # 类型未定义
        return 0  # '车辆'
    elif waymo_lane_type == 1:  # 高速公路
        return 0  # '车辆'
    elif waymo_lane_type == 2:  # 地面街道
        return 0  # '车辆'
    elif waymo_lane_type == 3:  # 自行车道
        return 1  # '自行车'
    elif waymo_lane_type == 4:  # 人行道
        return 3  # '行人'

# 将 Waymo 边界类型转换为 Argo2 边界类型
def waymo_boundary_type_converter(waymo_boundary_type):
    '''
        将 Waymo 边界类型转换为 Argo2 边界类型
    '''
    if waymo_boundary_type == 0:  # 类型未知
        return 14  # '未知'
    elif waymo_boundary_type == 1:  # 单白线破损
        return 2  # '虚线白'
    elif waymo_boundary_type == 2:  # 单白线实线
        return 9  # '实线白'
    elif waymo_boundary_type == 3:  # 双白线实线
        return 5  # '双实线白'
    elif waymo_boundary_type == 4:  # 单黄线破损
        return 3  # '虚线黄'
    elif waymo_boundary_type == 5:  # 双黄线破损
        return 6  # '双虚线黄'
    elif waymo_boundary_type == 6:  # 单黄线实线
        return 8  # '实线黄'
    elif waymo_boundary_type == 7:  # 双黄线实线
        return 4  # '双实线黄'
    elif waymo_boundary_type == 8:  # 双黄线交叉
        return 11  # '实线虚线黄'
    elif waymo_boundary_type == 9:  # 人行道
        return 15  # '人行道'
    elif waymo_boundary_type == 10:  # 中线
        return 16  # '中线'
    else:  # 其他
        return 14  # '未知'

# 定义对象类型
OBJECT_TYPES = ['vehicle',  # 0
                'pedestrian',  # 1
                'motorcyclist',  # 2
                'cyclist',  # 3
                'bus',  # 4
                'static',  # 5
                'background',  # 6
                'construction',  # 7
                'riderless_bicycle',  # 8
                'unknown'  # 9
               ] 

# 定义跟踪类别
TRACK_CATEGORIES = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']

# 定义多边形类型
POLYGON_TYPES = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
POLYGON_IS_INTERSECTIONS = [True, False, None]  # 多边形是否为交叉口
POINT_TYPES = ['DASH_SOLID_YELLOW',  # 0
               'DASH_SOLID_WHITE',  # 1
               'DASHED_WHITE',  # 2
               'DASHED_YELLOW',  # 3
               'DOUBLE_SOLID_YELLOW',  # 4 
               'DOUBLE_SOLID_WHITE',  # 5
               'DOUBLE_DASH_YELLOW',  # 6
               'DOUBLE_DASH_WHITE',  # 7
               'SOLID_YELLOW',  # 8
               'SOLID_WHITE',  # 9
               'SOLID_DASH_WHITE',  # 10 
               'SOLID_DASH_YELLOW',  # 11
               'SOLID_BLUE',  # 12
               'NONE',  # 13
               'UNKNOWN',  # 14
               'CROSSWALK',  # 15
               'CENTERLINE']  # 16

# 多项式基函数
def polynomial_basis_function(x, power):
    return x ** power  # 返回 x 的 power 次方

# 扩展函数
def expand(x, bf, bf_args=None):
    if bf_args is None:
        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)  # 连接常数项和基函数
    else:
        return np.array([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args]).T  # 返回扩展后的数组