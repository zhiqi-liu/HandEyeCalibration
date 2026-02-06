"""
公共函数
"""
import cv2
import numpy as np


def readline(file_: str, line_: int) -> list[float]:
    """
    从指定文件中读取指定行的内容，并将其转换为浮点数列表。

    :param file_: 包含数据的文件路径（字符串）。
    :param line_: 要读取的行号（从 0 开始计数）。
    :return: 包含行中浮点数的列表。如果行号超出范围或行内容为空，则返回空列表。
    """
    numbers = []
    current_line = -1

    with open(file_, 'r') as f:
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
    p_cam[:, 0] = p_cam[:, 0] * p_cam[:, 2]
    p_cam[:, 1] = p_cam[:, 1] * p_cam[:, 2]
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
