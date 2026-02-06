"""
相机参数获取函数
"""
import numpy as np


def xvisio(resolution="low"):
    """
    获取xvisio相机的内参矩阵、畸变系数和分辨率。

    Parameters
    ----------
    resolution : str, optional
        相机分辨率，可选值为 "low"（低分辨率）、"mid"（中分辨率）、"high"（高分辨率）。
        默认值为 "low"。

    Returns
    -------
    k_cam : np.ndarray
        相机内参矩阵，3x3 浮点数数组。
    dist_cam : np.ndarray
        相机畸变系数，1x5 浮点数数组。
    resolution_cam : tuple
        相机分辨率，格式为 (宽度, 高度)。

    """
    k_cam1 = np.array([
        [497.598, 0, 330.764],
        [0, 497.598, 251.883],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_cam1 = np.array([[0.141131], [-0.49418], [0.00038984], [0.00085806], [0.524062]], dtype=np.float64)
    resolution_cam1 = (640, 480)

    k_cam2 = np.array([
        [994.836, 0, 660.996],
        [0, 994.836, 503.231],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_cam2 = np.array([[0.144364], [-0.490998], [0.00104896], [0.00082048], [0.510932]], dtype=np.float64)
    resolution_cam2 = (1280, 720)

    k_cam3 = np.array([
        [1469.68, 0, 976.251],
        [0, 1469.68, 743.169],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_cam3 = np.array([[0.144883], [-0.495154], [0.00105921], [0.00080702], [0.516633]], dtype=np.float64)
    resolution_cam3 = (1920, 1080)
    if resolution == "low":
        return k_cam1, dist_cam1, resolution_cam1
    elif resolution == "mid":
        return k_cam2, dist_cam2, resolution_cam2
    elif resolution == "high":
        return k_cam3, dist_cam3, resolution_cam3
    return None
