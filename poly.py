import pickle
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
from read_pkl import analyze_tracks, get_keys_from_pkl

#读取轨迹信息
def analyze_poly_tracks(file_path: str) -> None:
    """
    分析Polynomial数据集中的轨迹信息
    Args:
        file_path: PKL文件路径
    """
    try:
        # 读取数据
        keys_list, data = get_keys_from_pkl(file_path)
        
        print(f"\n=== Polynomial轨迹分析 ===")
        print(f"文件路径: {file_path}")
        
        # 构建符合analyze_tracks格式的数据结构
        tracks_data = {}
        
        # 添加主车数据 track
        if 'tracks' in data:
            #如果tracks中存在AV，则添加主车数据
            if 'AV' in data['tracks']:
                tracks_data['AV'] = {
                    'state': {
                        'position': data['tracks']['AV']['state']['position'][:, :2],  # 假设前两列是x,y坐标
                        'velocity': data['tracks']['AV']['state']['velocity'][:, :2],  # 假设3,4列是速度
                        'heading': data['tracks']['AV']['state']['heading'],
                    },
                    'metadata': {
                        'type': 'VEHICLE',
                        'track_length': len(data['tracks']['AV']['state']['position']),
                        'object_id': 'AV'
                    }
            }
        
        # 添加其他车辆数据
        if 'tracks' in data:
            for idx, other in enumerate(data['tracks']):
                if other == 'AV':  # 跳过已处理的AV
                    continue
                    
                state_data = data['tracks'][other].get('state', {})
                position = state_data.get('position', None)
                velocity = state_data.get('velocity', None)
                heading = state_data.get('heading', None)
                
                # 确保数据存在并且维度正确
                position = position[:, :2] if position is not None and position.shape[1] >= 2 else None
                velocity = velocity[:, :2] if velocity is not None and velocity.shape[1] >= 2 else None
                heading = heading if heading is not None else None
                
                tracks_data[f'agent_{other}'] = {
                    'state': {
                        'position': position,
                        'velocity': velocity,
                        'heading': heading,
                    },
                    'metadata': {
                        'type': data['tracks'][other]['type'],
                        'track_length': len(position) if position is not None else 0,
                        'object_id': data['tracks'][other].get('metadata', {}).get('object_id', other)
                    }
                }
        
        # 使用analyze_tracks分析处理后的数据
        analyze_tracks(tracks_data)
        
        # 额外的Polynomial特定分析
        print("\n=== 额外的轨迹信息 ===")
        if 'trajectory' in data:
            traj = data['trajectory']
            print(f"主车轨迹:")
            print(f"- 轨迹长度: {len(traj)}帧")
            print(f"- 特征维度: {traj.shape[1]}个")
            print(f"- 第一帧数据: {traj[0]}")
        
        if 'others' in data:
            print(f"\n其他车辆信息:")
            print(f"- 车辆数量: {len(data['others'])}辆")
            if len(data['others']) > 0:
                print("- 第一辆车轨迹示例:")
                first_other = data['others'][0]
                if 'trajectory' in first_other:
                    print(f"  轨迹长度: {len(first_other['trajectory'])}帧")
                    print(f"  特征维度: {first_other['trajectory'].shape[1]}个")
                    print(f"  第一帧数据: {first_other['trajectory'][0]}")
        
    except Exception as e:
        print(f"分析文件时发生错误: {str(e)}")
#将数据储存为新的pkl文件
def save_data_to_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
#将轨迹转化为五次多项式，替换原始轨迹存于数组之中
def convert_to_polynomial(trajectory):
    """
    将轨迹转化为五次多项式
    Args:
        trajectory: 轨迹数据
    """
    None



def main():
    """主函数"""
    file_path = "/data1/data_zzs/dataset_unitraj_split/AG2_train/AG2_train_0_tmp/sd_av2_v2_00a0adb0-6c55-4df6-88cd-6a524f4edb39.pkl"
    analyze_poly_tracks(file_path)

if __name__ == "__main__":
    main()