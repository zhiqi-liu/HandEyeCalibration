"""
公共函数
"""
import cv2
import numpy as np
import os
from camera import xvisio


def readline(file_: str, line_: int) -> list[float]:
    """
    从指定文件中读取指定行的内容，并将其转换为浮点数列表。

    :param file_: 包含数据的文件路径（字符串）。
    :param line_: 要读取的行号（从 0 开始计数）。
    :return: 包含行中浮点数的列表。如果行号超出范围或行内容为空，则返回空列表。
    """
    numbers = []
    current_line = -1

    with open(file_) as f:
        for data in f:
            current_line += 1
            if current_line == line_:
                data = data.strip()
                if data:
                    for temp in data.split(','):
                        numbers.append(float(temp))
                break
    return numbers


def p_img2cam(pt_, z_, K_, dist_coeffs_):
    """
    将像素点转换到相机坐标系

    :param pt_: tuple、list、ndarray
        像素点坐标
    :param z_: float、list、ndarray
        像素点深度
    :param K_: np.ndarray
        相机内参矩阵，3x3 浮点数数组。
    :param dist_coeffs_: np.ndarray
        相机畸变系数，1x5 浮点数数组。
    :return: ndarray
        相机坐标系下的 3d 坐标， nx3 浮点数数组
    """
    pt_ = np.array(pt_, dtype=np.float32).reshape(-1, 1, 2)
    pt_undist = cv2.undistortPoints(pt_, K_, dist_coeffs_)
    pt_undist = pt_undist.reshape(-1, 2)
    z_ = np.array(z_, dtype=np.float32).reshape(-1, 1)
    p_cam = np.hstack((pt_undist, z_))
    p_cam[:, 0] *= p_cam[:, 2]
    p_cam[:, 1] *= p_cam[:, 2]
    return p_cam


def p_cam2base(P_cam_, T_cam2base_):
    """
    相机坐标系转换到机械臂基坐标系

    :param P_cam_: ndarray
        相机坐标系下的 3d 坐标，nx3 浮点数数组
    :param T_cam2base_: ndarray
        相机到基座的变换矩阵，4x4 浮点数数组
    :return: ndarray
        机械臂基坐标系下的 3d 坐标，nx3 浮点数数组
    """
    P_cam_ = P_cam_.transpose()
    ones_row_ = np.ones((1, P_cam_.shape[1]))
    points_homogeneous_ = np.vstack((P_cam_, ones_row_))
    P_base_ = T_cam2base_ @ points_homogeneous_
    P_base_ = P_base_.transpose()[:, :3]
    return P_base_


def get_best_T_cam2base(calib_data_dir: str):
    """
    读取手眼标定结果中最优的标定矩阵

    :param calib_data_dir: str
        存储了手眼标定结果的文件
    :returns: ndarray
        最优手眼标定矩阵，4x4浮点数数组
    """
    calib_data = np.load(os.path.join(calib_data_dir, "T_cam2base.npz"))

    candidates = [
        (calib_data["residual_opencv"], calib_data["T_opencv"]),
        (calib_data["residual_my1"], calib_data["T_my1"]),
        (calib_data["residual_my2"], calib_data["T_my2"])
    ]

    residual_min, T_best = min(candidates, key=lambda x: x[0])

    print("residual_min:", residual_min)
    print(f"T_best:\n{T_best}")

    return T_best

def p_img2base(pt, z, resolution):
    """
    将 2d 像素位置转换到机械臂基坐标系下 3d 位置

    :param pt: tuple
        像素坐标
    :param z: float
        pt 坐标的深度
    :param resolution: str
        图像分辨率，"low", "mid", "high"，分别对应 640x480、1280x620、1920x1080
    :returns: ndarray
        pt 点在机械臂坐标系下的 3d 位置
    """
    K_cam, dist_cam, _ = xvisio(resolution=resolution)
    P_cam = p_img2cam(pt, z, K_cam, dist_cam)
    print(f"P_cam:\n{P_cam}")
    calib_data_dir = "calib_data640x480"
    if resolution == "mid":
        calib_data_dir = "calib_data1280x720"
    elif resolution == "high":
        calib_data_dir = "calib_data1920x1080"
    T_cam2base = get_best_T_cam2base(calib_data_dir)
    P_base = p_cam2base(P_cam, T_cam2base)
    print(f"P_base:\n{P_base}")
    return P_base
