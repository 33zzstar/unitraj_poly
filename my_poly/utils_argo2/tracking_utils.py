'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import numpy as np
from scipy.linalg import block_diag
from scipy.special import binom

from utils_common.helper_utils import polynomial_basis_function, expand

import sys
sys.path.append('/home/zhaozishan/UniTraj/unitraj/my_poly/cmotap/continental-multi-object-tracker-and-predictor/')
from src.cmotap.trajectory import Trajectory 
from src.cmotap.basisfunctions.bernsteinpolynomials import BernsteinPolynomials, Monomials
from src.cmotap.statedensities.gaussian_control_point_density import GaussianControlPointDensity
from src.cmotap.motionmodels.trajectory_motion import TrajectoryMotion
from src.cmotap.observationmodels.trajectory_observation import TrajectoryObservation


# get the prior form the Empirical Bayes analysis
def get_prior(degree, timescale, prior_data, spacedim=2):
    
    # The prior was calculatd for monomials and is
    # organized in the following way
    # x0x0 x0x1 x0x2 x0y0 x0y1 x0y2
    # x1x0 x1x1 x1x2 x1y0 x1y1 x1y2
    # x2x0 x2x1 x2x2 x2y0 x2y1 x2y2
    # y0x0 y0x1 y0x2 y0y0 y0y1 y0y2
    # y1x0 y1x1 y0x2 y1y0 y1y1 y1y2
    # y2x0 y2x1 y2x2 y2y0 y2y1 y2y2
    #prior_data = json.load(open('logs/gradient_tape/agt_xy_polar_plus_const_50/result_summary.json', 'r'))

    # We reorganize the prior into the following structure
    # x0x0 x0y0 x0x1 x0y1 x0x2 x0y2
    # y0x0 y0y0 y0x1 y0y1 y0x2 y0y2
    # x1x0 x1y0 x1x1 x1y1 x1x2 x1y2
    # ....
    perm = np.zeros((2 * degree, 2 * degree))
    for i in range(degree):
        perm[2 * i, i] = 1
        perm[2 * i + 1, degree + i] = 1

    # all trajectories in EB analysis start at (0, 0), so there is little uncertainty about start point
    monomial_cov_unscaled = np.diag(np.block([1e-3, 1e-3, np.zeros(spacedim * degree)]))
    monomial_cov_unscaled[2:, 2:] = np.array(perm @ prior_data['A_list'][degree - 1] @ perm.T)
    
    monomial_scale = np.diag([timescale ** deg for deg in range(0, degree + 1)])
    monomial_scale = np.kron(monomial_scale, np.eye(spacedim))
    monomial_cov =  monomial_scale @ monomial_cov_unscaled @ monomial_scale.T
    
    monomial_mean = monomial_scale @ np.kron(np.zeros(degree + 1), np.ones(spacedim))
    
    # Now we transform to Bernstein Polynomials
    M = np.zeros((degree + 1, degree + 1))

    for k in range(0, degree + 1):
        for i in range(k, degree + 1):
            M[i, k] = (-1)**(i - k) * binom(degree, i) * binom(i, k)
        
    M_inv = np.linalg.solve(M, np.eye(degree + 1))
    future_prior = np.kron(M_inv, np.eye(2)) @ monomial_cov @ np.kron(M_inv, np.eye(2)).T
    
    # reorder entries for history
    perm = np.zeros((degree + 1, degree + 1))
    perm[np.arange(degree + 1), np.arange(degree, -1, -1)] = 1
    perm = np.kron(perm, np.eye(2))
    history_prior = perm @ future_prior @ perm.T
    
    return future_prior, history_prior


# The Noise Model from the Emprical Bayes Analysis
def build_ego_observation_noise_covariance(degree, prior_data):        
    """
    构建自车观测噪声协方差矩阵
    
    输入参数:
        degree (int): 多项式次数
        prior_data (dict): 包含先验参数的字典，结构为:
            {
                'B_list': [
                    {
                        'B_diag': [float],  # 对角线噪声项
                        'B_by_diag': [float] # 非对角线噪声项
                    },
                    ... # 不同degree对应的参数
                ]
            }
            
    输出维度:
        R_world (np.ndarray): 2x2的噪声协方差矩阵，结构:
        [[对角线噪声, 非对角线噪声],
         [非对角线噪声, 对角线噪声]]
    """
    # 从先验数据中获取当前degree对应的噪声参数
    b_diag = prior_data['B_list'][degree - 1]['B_diag'][0]      # 标量值，对角线元素
    b_by_diag = prior_data['B_list'][degree - 1]['B_by_diag'][0] # 标量值，非对角线元素
    
    # 构建对称的协方差矩阵
    R_world = np.array([[b_diag, b_by_diag],
                        [b_by_diag, b_diag]])
    
    return R_world


# The Lidar Noise Model from the Emprical Bayes Analysis
def build_obj_observation_noise_covariance(ego_pos, ego_heading, obj_pos, degree, prior_data):        
    r = np.linalg.norm(obj_pos - ego_pos)
    alpha = np.arctan2(*(obj_pos - ego_pos)[::-1]) - ego_heading

    beta0 =  prior_data['B_list'][degree - 1]['B_d'][0][0]
    beta1 = prior_data['B_list'][degree - 1]['B_d'][1][0]
    beta2 = prior_data['B_list'][degree - 1]['B_d'][2][0]
    sigma_r = np.sqrt(beta0 + beta1 * r + beta2 * r**2)
    
    R_ego, R_world = cartesian_noise_from_polar_noise(
       r, alpha, sigma_r=sigma_r, sigma_alpha=np.sqrt(prior_data['B_list'][degree - 1]['B_theta'][0]), sigma_c=np.sqrt(prior_data['B_list'][degree - 1]['B_const'][0]), ego_heading=ego_heading
    )

    return R_world 


# The Lidar Noise Model from the Emprical Bayes Analysis
def build_obj_observation_noise_covariance_batch(ego_pos, ego_heading, obj_pos, degree, prior_data):        
    """批量构建目标观测噪声协方差矩阵
    
    基于经验贝叶斯分析的激光雷达噪声模型，计算目标物体在极坐标系下的观测噪声，
    并将其转换到世界坐标系。
    
    Args:
        ego_pos (np.ndarray): 自车位置数组 [N,2]
        ego_heading (np.ndarray): 自车航向角数组 [N]
        obj_pos (np.ndarray): 目标位置数组 [N,2]
        degree (int): 多项式次数
        prior_data (dict/list): 先验参数数据
        
    Returns:
        np.ndarray: 世界坐标系下的噪声协方差矩阵 [N,2,2]
    """
    
    # 计算相对位置向量
    d = obj_pos - ego_pos  # [N,2]
    
    # 计算相对距离
    r = np.linalg.norm(d, axis=-1)  # [N]
    
    # 初始化临时协方差矩阵
    R_temp = np.zeros((d.shape[0], 2, 2))  # [N,2,2]
    
    # 计算相对方位角（自车坐标系下）
    alpha = np.arctan2(d[:,1], d[:,0]) - ego_heading  # [N]
    
    # 构建自车到地图的旋转矩阵
    c_ego_to_map, s_ego_to_map = np.cos(ego_heading), np.sin(ego_heading)
    R_ego_to_map = np.transpose(np.array(
        ((c_ego_to_map, -s_ego_to_map), 
         (s_ego_to_map, c_ego_to_map))), 
        (2, 0, 1))  # [N,2,2]
    
    # 扩展径向距离特征
    phi_r = expand(r, bf=polynomial_basis_function, bf_args=range(1, 2+1))  # [N,2]
    
    # 加载先验参数
    if isinstance(prior_data, list):
        # 多类别先验参数
        beta_r =  np.array([p['B_list'][degree - 1]['B_d'] for p in prior_data])  # [C,2,1]
        var_alpha =  np.array([p['B_list'][degree - 1]['B_theta'] for p in prior_data])  # [C,1,1]
        var_const =  np.array([p['B_list'][degree - 1]['B_const'] for p in prior_data])  # [C,1,1]
        
        var_r = (phi_r[:, None, :] @ beta_r)[:, 0, 0]
        var_alpha = var_alpha.squeeze(-1)
        var_const = var_const.squeeze(-1)
    else:
        beta_r =  prior_data['B_list'][degree - 1]['B_d']
        var_alpha =  prior_data['B_list'][degree - 1]['B_theta']
        var_const =  prior_data['B_list'][degree - 1]['B_const']
    
        var_r = (phi_r @ beta_r)[:, 0]
    

    var_lon_lon = var_r * np.power(np.cos(alpha), 2) + var_alpha * np.power(r, 2) * np.power(np.sin(alpha), 2)
    var_lat_lat = var_r * np.power(np.sin(alpha), 2) + var_alpha * np.power(r, 2) * np.power(np.cos(alpha), 2)
    var_lon_lat = var_r * np.sin(alpha) * np.cos(alpha) - var_alpha *  np.power(r, 2) * np.sin(alpha) * np.cos(alpha)


    R_temp[:, 0, 0] += (var_lon_lon + var_const)
    R_temp[:, 1, 1] += (var_lat_lat + var_const)
    R_temp[:, 0, 1] += var_lon_lat
    R_temp[:, 1, 0] += var_lon_lat

    R_world = R_ego_to_map @ R_temp @ R_ego_to_map.transpose((0,2,1))

    return R_world


def cartesian_noise_from_polar_noise(
    r, alpha, sigma_r, sigma_alpha, sigma_c, ego_heading
):
    """Generate a Cartesian Observation Noise covariance from a polar one

    Parameters
    ----------
    r: float
        distance between sensor and object

    alpha: float, radians
        bearing angle of object seen from sensor

    sigma_r: float
        standard deviation of r

    sigma_alpha: float
        standard deviation of alpha

    sigma_c: float
        standard deviation for timing jitter

    ego_heading: float, radians
        sensor heading in world coordinate system

    Returns
    -------
    R_ego: ndarray
        cartesian position noise covariance in sensor coordinates

    R_world: ndarray
        cartesian position noise covariance in world coordinates,
        i.e. rotated by ego_heading
    """

    s_xx = (sigma_r * np.cos(alpha)) ** 2 + (sigma_alpha * r * np.sin(alpha)) ** 2
    s_yy = (sigma_r * np.sin(alpha)) ** 2 + (sigma_alpha * r * np.cos(alpha)) ** 2
    s_xy = (sigma_r**2 - sigma_alpha**2 * r**2) * np.sin(alpha) * np.cos(alpha)

    R_ego = np.array([[s_xx, s_xy], [s_xy, s_yy]])

    Rot = np.array(
        [
            [np.cos(ego_heading), np.sin(ego_heading)],
            [-np.sin(ego_heading), np.cos(ego_heading)],
        ]
    )

    R_world = Rot.T @ (R_ego + sigma_c**2 * np.eye(2)) @ Rot

    return R_ego, R_world


def get_initial_state_ego(pos, v0, prior_cov, TRAJ, BASIS, TIMESCALE):
    """
    初始化自车状态分布
    
    输入参数维度:
        pos (np.ndarray): 初始位置坐标 [2] (x, y)
        v0 (np.ndarray): 初始速度向量 [2] (vx, vy)
        prior_cov (np.ndarray): 先验协方差矩阵 [N_ctrl_pts*2, N_ctrl_pts*2]
                             (N_ctrl_pts = BASIS.size 为控制点数量)
        TRAJ: 轨迹对象 (包含运动模型参数)
        BASIS: 基函数对象 (需包含size属性)
        TIMESCALE (float): 时间尺度参数（秒）
        
    输出维度:
        GaussianControlPointDensity: 包含以下属性的状态分布对象
            - x: 控制点均值向量 [N_ctrl_pts*2]
            - P: 控制点协方差矩阵 [N_ctrl_pts*2, N_ctrl_pts*2]
    """
    # 初始化控制点状态（倒序排列）
    # np.kron生成控制点速度的初始猜测，[::-1]实现时间倒序
    # 负号用于补偿运动方向，使初始速度方向与实际运动方向一致
    initial_state = GaussianControlPointDensity(
        x=np.kron(np.arange(BASIS.size)[::-1], -v0),  # [N_ctrl_pts*2]
        P=prior_cov  # [N_ctrl_pts*2, N_ctrl_pts*2]
    )
    
    # 构建观测模型参数
    OM = TrajectoryObservation(
        TRAJ, 
        t=np.linspace(0, TIMESCALE, BASIS.size - 1),  # 时间采样点 [N_obs_points]
        derivatives=[
            [1] for _ in range(BASIS.size - 2)] +    # 前N-2个观测点使用一阶导数（速度）
            [[1, 0]],                                # 最后一个点观测位置和速度
        R=[0.5**2 * np.eye(2)] * (BASIS.size - 2) +  # 速度观测噪声 [2x2] * (N-2)
          [np.diag([0.5, 0.5, 0.1, 0.1])**2]        # 最终点位置+速度噪声 [4x4]
    )

    # 构建观测向量：重复速度观测 + 零加速度
    # 前N-1个观测点为速度，最后两个零值表示加速度为零
    z = np.block([v0 for _ in range(BASIS.size - 1)] + [0, 0])  # [N_obs*2 + 2]
    
    # 执行卡尔曼滤波更新步骤
    initial_state = initial_state.update(z, OM)
    
    # 将状态平移到实际观测位置
    # 使用克罗内克积将位置扩展到所有控制点
    initial_state._x += np.kron(np.ones(BASIS.size), pos)  # [N_ctrl_pts*2]
    
    return initial_state


def get_initial_state_agt(pos, v0, prior_cov, TRAJ, BASIS, TIMESCALE):
    """
    初始化目标对象的状态分布
    
    Args:
        pos (np.ndarray): 初始位置坐标 [x, y]
        v0 (np.ndarray): 初始速度向量 [vx, vy]
        prior_cov (np.ndarray): 先验协方差矩阵
        TRAJ: 轨迹对象
        BASIS: 基函数对象
        TIMESCALE (float): 时间尺度参数
        
    Returns:
        GaussianControlPointDensity: 高斯控制点密度表示的状态分布
    """
    
    # 初始化高斯控制点密度
    # 使用克罗内克积生成初始状态向量，[::-1]实现倒序排列控制点
    # 注意速度前的负号用于补偿运动方向
    initial_state = GaussianControlPointDensity(
        x=np.kron(np.arange(BASIS.size)[::-1], -v0), # note the minus here!
        P=prior_cov
    )
    
    # 构建观测模型
    OM = TrajectoryObservation(
        TRAJ, 
        t=np.linspace(0, TIMESCALE, BASIS.size - 1),  # 时间采样点
        derivatives=[
            [1] for _ in range(BASIS.size - 2)] +  # 前N-2个观测点使用一阶导数(速度)
            [[1, 0]],  # 最后一个观测点使用位置和一阶导数(位置+速度)
        R=[3**2 * np.eye(2)] * (BASIS.size - 2) +  # 速度观测噪声 (3m/s)^2
          [np.diag([3, 3, 0.1, 0.1])**2]  # 最终位置和速度噪声 (3m, 0.1m/s)^2
    )

    # 构建观测向量：重复速度观测 + 零加速度
    z = np.block([v0 for _ in range(BASIS.size - 1)] + [0, 0])
    
    # 更新状态估计
    initial_state = initial_state.update(z, OM)
    
    # 将状态平移到实际观测位置
    # 使用克罗内克积将位置扩展到所有控制点
    initial_state._x += np.kron(np.ones(BASIS.size), pos)
    
    return initial_state


def D_matrix(degree, derivative): 
    '''
    A derivative matrix for monomial basis function
    param degree: the order of polynomial
    param derivative: the order of derivatives
    '''
    D=np.eye(degree+1)
    for i in range(derivative):
        D = D @ np.diag(np.arange(1,degree+1), 1)

    return D 


def get_monomial_prior(prior_data, degree, spacedim = 2):
    # The prior was calculatd for monomials and is
    # organized in the following way
    # x0x0 x0x1 x0x2 x0y0 x0y1 x0y2
    # x1x0 x1x1 x1x2 x1y0 x1y1 x1y2
    # x2x0 x2x1 x2x2 x2y0 x2y1 x2y2
    # y0x0 y0x1 y0x2 y0y0 y0y1 y0y2
    # y1x0 y1x1 y0x2 y1y0 y1y1 y1y2
    # y2x0 y2x1 y2x2 y2y0 y2y1 y2y2
    #prior_data = json.load(open('logs/gradient_tape/agt_xy_polar_plus_const_50/result_summary.json', 'r'))

    # We reorganize the prior into the following structure
    # x0x0 x0y0 x0x1 x0y1 x0x2 x0y2
    # y0x0 y0y0 y0x1 y0y1 y0x2 y0y2
    # x1x0 x1y0 x1x1 x1y1 x1x2 x1y2
    # ....
    perm = np.zeros((2 * degree, 2 * degree))
    for i in range(degree):
        perm[2 * i, i] = 1
        perm[2 * i + 1, degree + i] = 1
    
    # all trajectories in EB analysis start at (0, 0), so there is little uncertainty about start point
    monomial_cov_unscaled = np.diag(np.block([1e8, 1e8, np.zeros(spacedim * degree)]))
    monomial_cov_unscaled[2:, 2:] = np.array(perm @ prior_data['A_list'][degree - 1] @ perm.T)
    
    return monomial_cov_unscaled

def monomial_to_bernstein(monomial_mean, monomial_cov, timescale, degree, spacedim = 2):
    monomial_scale = np.diag([timescale ** deg for deg in range(0, degree + 1)])
    monomial_scale = np.kron(monomial_scale, np.eye(spacedim))
    
    monomial_cov_scaled =  monomial_scale @ monomial_cov @ monomial_scale.T
    monomial_mean_scaled = monomial_scale @ monomial_mean
    
    # Now we transform to Bernstein Polynomials
    M = np.zeros((degree + 1, degree + 1))

    for k in range(0, degree + 1):
        for i in range(k, degree + 1):
            M[i, k] = (-1)**(i - k) * binom(degree, i) * binom(i, k)
        
    M_inv = np.linalg.solve(M, np.eye(degree + 1))
    bernstein_cov = np.kron(M_inv, np.eye(spacedim)) @ monomial_cov_scaled @ np.kron(M_inv, np.eye(spacedim)).T
    bernstein_mean = np.kron(M_inv, np.eye(spacedim)) @ monomial_mean_scaled
    
    return bernstein_mean, bernstein_cov