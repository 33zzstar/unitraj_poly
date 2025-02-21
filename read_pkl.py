import pickle
import numpy as np
from typing import Dict, Any, Tuple
import os
from pathlib import Path

def get_keys_from_pkl(file_path: str) -> Tuple[list, Dict[str, Any]]:
    """
    读取PKL文件并返回键列表和数据
    Args:
        file_path: PKL文件路径
    Returns:
        keys: 键列表
        data: 数据字典
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        keys = list(data.keys())
    return keys, data

def analyze_tracks(tracks_data: Dict) -> None:
    """
    分析tracks数据的结构和内容
    Args:
        tracks_data: tracks字典数据
    """
    av_count = sum(1 for k in tracks_data if k == "AV")
    print(f"\n=== Tracks 分析 ===")
    print(f"总车辆数: {len(tracks_data)}")
    print(f"自车: {av_count}")
    print(f"其他车辆数: {len(tracks_data) - av_count}")
    
    # 分析一个示例车辆的数据结构
    sample_vehicle = next(iter(tracks_data.values()))
    print("\n车辆数据结构:")
    for key, value in sample_vehicle.items():
        if key == "state":
            print("\nstate包含:")
            for state_key, state_value in value.items():
                #打印出首个元素
                print(f"  {state_key}: shape={state_value.shape}, first_element={state_value[0]}")
        elif key == "metadata":
            print("\nmetadata包含:")
            for meta_key, meta_value in value.items():
                #打印出具体信息
                print(f"  {meta_key}: {meta_value}")


def analyze_map_features(map_data: Dict) -> None:
    """
    分析地图特征数据
    Args:
        map_data: 地图特征字典
    """
    print(f"\n=== 地图特征分析 ===")
    print(f"特征数量: {len(map_data)}")
    
    # 分析特征类型分布
    feature_types = {}
    for feature in map_data.values():
        feature_type = feature.get('type')
        feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
    
    print("\n特征类型分布:")
    for ftype, count in feature_types.items():
        print(f"{ftype}: {count}个")

def print_metadata_details(metadata: Dict) -> None:
    """
    打印metadata的详细信息
    Args:
        metadata: metadata字典数据
    """
    # metadata字段说明
    field_descriptions = {
        'id': '场景的唯一标识符',
        'coordinate': '坐标系统信息',
        'ts': '时间戳信息',
        'metadrive_processed': '是否经过MetaDrive处理',
        'sdc_id': '自动驾驶车辆的唯一标识符',
        'dataset': '数据来源的数据集名称',
        'scenario_id': '场景ID',
        'source_file': '原始数据文件的路径或名称',
        'track_length': '轨迹长度',
        'current_time_index': '当前时间步的索引',
        'objects_of_interest': '感兴趣的对象列表',
        'sdc_track_index': '自动驾驶车辆在轨迹数据中的索引',
        'tracks_to_predict': '需要预测的轨迹列表',
        'object_summary': '场景中所有对象的概要信息',
        'number_summary': '场景中各类对象的数量统计'
    }

    print("\n=== Metadata 详细信息 ===")
    print(f"总字段数: {len(metadata)}")
    
    for key, value in metadata.items():
        print(f"\n{'-'*50}")
        print(f"字段名: {key}")
        print(f"说明: {field_descriptions.get(key, '未知字段')}")
        
        # 根据不同类型进行处理
        if isinstance(value, (list, np.ndarray)):
            print(f"类型: {type(value).__name__}")
            print(f"长度: {len(value)}")
            if len(value) > 0:
                print(f"示例值: {value[0]}")
                if isinstance(value, np.ndarray):
                    print(f"形状: {value.shape}")
                    print(f"数据类型: {value.dtype}")
        
        elif isinstance(value, dict):
            keys = list(value.keys())
            if len(keys) == 1:
                print(f"包含的键: {keys[0]}")
            else:
                print(f"包含 {len(keys)} 个键")
            if len(value) > 0:
                first_key = next(iter(value))
                print(f"第一个键值对: {first_key}: {value[first_key]}")
        
        elif isinstance(value, bool):
            print(f"类型: 布尔值")
            print(f"值: {value}")
        
        elif isinstance(value, (int, float)):
            print(f"类型: {type(value).__name__}")
            print(f"值: {value}")
        
        elif isinstance(value, str):
            print(f"类型: 字符串")
            print(f"值: {value}")
        
        else:
            print(f"类型: {type(value).__name__}")
            print(f"值: {value}")

    # # 打印一些统计信息
    # print("\n=== 统计信息 ===")
    # type_counts = {}
    # for value in metadata.values():
    #     type_name = type(value).__name__
    #     type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    # print("数据类型分布:")
    # for type_name, count in type_counts.items():
    #     print(f"- {type_name}: {count}个")

def identify_dataset(file_path: str) -> str:
    """
    识别数据集类型
    Args:
        file_path: 文件路径
    Returns:
        str: 数据集类型 ('unitraj' 或 'polynomial' 或 'other')
    """
    # 将路径转换为Path对象以便操作
    path = Path(file_path)
    
    if 'unitraj' in str(path).lower():
        return 'unitraj'
    elif 'polynomial' in str(path).lower() or 'everything-polynomial' in str(path).lower():
        return 'polynomial'
    else:
        return 'other'

def read_unitraj(file_path: str) -> None:
    """
    读取UniTraj数据集的PKL文件
    Args:
        file_path: PKL文件路径
    """
    keys_list, data = get_keys_from_pkl(file_path)
    
    print(f"\n=== UniTraj数据分析 ===")
    print(f"文件路径: {file_path}")
    print(f"包含的主要键: {keys_list}")
    
    if 'tracks' in data:
        analyze_tracks(data['tracks'])
    
    if 'map_features' in data:
        analyze_map_features(data['map_features'])
    
    if 'metadata' in data:
        print_metadata_details(data['metadata'])



 

def process_directory(directory_path: str) -> None:
    """
    处理目录中的所有PKL文件
    Args:
        directory_path: 目录路径
    """
    # 获取目录中所有的PKL文件
    pkl_files = list(Path(directory_path).rglob("*.pkl"))
    
    if not pkl_files:
        print(f"在目录 {directory_path} 中未找到PKL文件")
        return
    
    print(f"找到 {len(pkl_files)} 个PKL文件")
    
    # 处理每个文件
    for file_path in pkl_files:
        dataset_type = identify_dataset(str(file_path))
        print(f"\n处理文件: {file_path}")
        print(f"识别为: {dataset_type} 数据集")
        
        try:
            if dataset_type == 'unitraj':
                read_unitraj(str(file_path))
            elif dataset_type == 'polynomial':
                read_poly(str(file_path))
            else:
                read_poly(str(file_path))
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {str(e)}")

def main():
    """
    主函数
    """
    # 可以处理单个文件或整个目录
    path = '/data1/data_zzs/dataset_unitraj_split/AG2_train/AG2_train_0_tmp/sd_av2_v2_00a0adb0-6c55-4df6-88cd-6a524f4edb39.pkl'
    # path = "/data1/data_zzs/everything-polynomial_data/train_A2_processed/00a0adb0-6c55-4df6-88cd-6a524f4edb39/00a0adb0-6c55-4df6-88cd-6a524f4edb39_track_infos.pkl"
    if os.path.isdir(path):
        process_directory(path)
    elif os.path.isfile(path):
        dataset_type = identify_dataset(path)
        if dataset_type == 'unitraj':
            read_unitraj(path)
        elif dataset_type == 'polynomial':
            read_poly(path)
        else:
            read_poly(path)
    else:
        print(f"错误：路径 {path} 不存在")

if __name__ == "__main__":
    main()