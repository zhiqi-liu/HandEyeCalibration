"""
提供手眼标定相关功能，包括：
- 使用 OpenCV 进行手眼标定 (eye-to-hand)
- 自定义手眼标定算法
- SE(3) 残差计算
- 处理棋盘格图像并生成角点可视化

环境
- python 3.10
- roboticstoolbox-python 1.1.1
- numpy 1.26.4
- scipy 1.15.3
- opencv-python 4.9.0.80

Author: lzq
Date: 2026-01-30
"""

import os
import cv2
import numpy as np
import shutil
import yaml
from scipy.spatial.transform import Rotation

from robotic_arm import rm65_6FB_model
from camera import xvisio
from common import readline


class HandEyeCalib:
    """
    手眼标定类，用于计算相机到机械臂基坐标的变换矩阵。

    Attributes
    ----------
    _board_size : tuple
        棋盘格的尺寸，格式为 (宽度, 高度)。
    _square_size : float
        棋盘格中每个方格的尺寸（单位：米）。
    _K_cam : np.ndarray
        相机内参矩阵，3x3 浮点数数组。
    _dist_coeffs : np.ndarray
        相机畸变系数，1x5 浮点数数组。
    _robot : DHRobot
        机械臂模型对象，用于计算关节角度。
    """

    def __init__(self):
        self._board_size = (6, 4)
        self._square_size = 0.05
        self._robot = rm65_6FB_model()
        self._K_cam, self._dist_coeffs, _ = xvisio(resolution="high")

    def _get_pose_target2cam(self, images_dir_):
        """
        获取棋盘格到相机坐标系的位姿变换。

        :param images_dir_: 包含棋盘格图像的目录路径（字符串）。
        :return: 一个包含棋盘格到相机坐标系位姿变换矩阵的列表。
        """
        # =====  构造棋盘格 3D 点 =====
        object_points = []
        w, h = self._board_size
        for i in range(h):
            for j in range(w):
                object_points.append(
                    [(j - (w - 1) / 2) * self._square_size, (i - (h - 1) / 2) * self._square_size, 0.0])
        object_points = np.array(object_points, dtype=np.float32)

        # save_images_dir = images_dir_ + "_corners"
        # if os.path.exists(save_images_dir):
        #     shutil.rmtree(save_images_dir)
        #     os.makedirs(save_images_dir)
        # else:
        #     os.makedirs(save_images_dir)

        idx = 0
        idxs = []
        T_target2cams = []
        for fname in sorted(os.listdir(images_dir_), key=lambda x: int(os.path.splitext(x)[0])):
            path = os.path.join(images_dir_, fname)
            if not os.path.isfile(path):
                continue

            image = cv2.imread(path, 0)
            if image is None:
                idx += 1
                continue

            # ===== 检测角点 =====
            found, corners = cv2.findChessboardCorners(image, self._board_size)

            if found:
                # ===== 亚像素优化 =====
                cv2.cornerSubPix(
                    image,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
                )

                # color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # for i in range(len(corners)):
                #     if i != len(corners) - 1:
                #         cv2.line(color_image, tuple(corners[i][0].astype(int)), tuple(corners[i + 1][0].astype(int)), (0, 255, 0), 2)
                #     cv2.circle(color_image, tuple(corners[i][0].astype(int)), 5, (0, 0, 255), 2)
                # save_image_file = os.path.join(save_images_dir, fname)
                # cv2.imwrite(save_image_file, color_image)

                # ===== PnP 求解位姿 target → camera =====
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    corners,
                    self._K_cam,
                    self._dist_coeffs
                )

                if success:
                    # 验证解
                    # proj_pts, _ = cv2.projectPoints(
                    #     object_points,
                    #     rvec,
                    #     tvec,
                    #     self._K_cam,
                    #     self._dist_coeffs
                    # )
                    # proj_pts = proj_pts.reshape(-1, 2)
                    # corners = corners.reshape(-1, 2)
                    # err = np.linalg.norm(proj_pts - corners, axis=1)
                    # mean_err = err.mean()
                    # max_err = err.max()
                    # print("mean reprojection pixel error:", mean_err)
                    # print("max reprojection pixel error:", max_err)
                    idxs.append(idx)
                    R, _ = cv2.Rodrigues(rvec)
                    T_target2cam = np.eye(4)
                    T_target2cam[:3, :3] = R
                    T_target2cam[:3, 3] = tvec.squeeze()
                    T_target2cams.append(T_target2cam)

            idx += 1

        return T_target2cams, idxs

    def _get_pose_base2end(self, hand_angle_file_, idxs_, degree_):
        """
        获取机械臂基座到末端的位姿变换矩阵。

        :param hand_angle_file_: 包含机械臂关节角度数据（每组6个关节角度，单位：弧度）的文件路径（字符串）。
        :param idxs_: 要获取位姿的行索引列表（整数列表）。
        :param degree_: 输入角度单位是角度还是弧度（布尔值）。
        :return: 一个包含机械臂末端到基座的位姿变换矩阵的列表。
        """
        T_base2ends = []
        for idx in idxs_:
            hand_angle = readline(file_=hand_angle_file_, line_=idx)
            hand_angle = np.array(hand_angle, dtype=np.float64)
            if degree_:
                hand_angle = np.deg2rad(hand_angle)
            T_end2base = self._robot.fkine(hand_angle)
            T_base2ends.append(np.linalg.inv(T_end2base.A))
        return T_base2ends

    def _split_R_t(self, Ts_):
        """
        从变换矩阵中分离出旋转矩阵和平移向量。

        :param Ts_: 包含变换矩阵的列表，每个元素为 4x4 浮点数数组。
        :return: 一个包含旋转矩阵列表和平移向量列表的元组。
        """
        Rs_ = []
        ts_ = []
        for i in range(len(Ts_)):
            Rs_.append(Ts_[i][:3, :3])
            ts_.append(Ts_[i][:3, 3])
        return Rs_, ts_

    def run_opencv_hand_to_eye(self, images_dir_, hand_angle_file_, degree_=False):
        """
        运行手眼标定，计算相机到基座的位姿变换矩阵。

        :param images_dir_: 包含棋盘格图像的目录路径（字符串）。
        :param hand_angle_file_: 包含机械臂关节角度数据（每组6个关节角度，单位：弧度）的文件路径（字符串）。
        :param degree_: 输入角度单位是角度还是弧度（布尔值）。默认值为弧度。
        :return: 相机到基座的变换矩阵（4x4 浮点数数组）。
        """
        # 1️⃣ 获取相机对棋盘的位姿
        T_target2cams, idxs = self._get_pose_target2cam(images_dir_)

        # 2️⃣ 获取机械臂末端到基座的位姿
        T_base2ends = self._get_pose_base2end(hand_angle_file_, idxs, degree_=degree_)

        # 分离R， t
        R_target2cams, t_target2cams = self._split_R_t(T_target2cams)
        R_base2ends, t_base2ends = self._split_R_t(T_base2ends)

        # 3️⃣ 手眼标定
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_base2ends, t_base2ends,
            R_target2cams, t_target2cams,
            method=cv2.CALIB_HAND_EYE_HORAUD
        )

        X_ = np.eye(4)
        X_[0:3, 0:3] = R_cam2base
        X_[0:3, 3] = t_cam2base.squeeze()
        se3_residual_ = self._SE3_residual(X_, T_base2ends, T_target2cams)

        return X_, se3_residual_

    def run_my_hand_to_eye(self, images_dir_, hand_angle_file_, degree_=False):
        """
        运行手眼标定，计算相机到基座的位姿变换矩阵。

        :param images_dir_: 包含棋盘格图像的目录路径（字符串）。
        :param hand_angle_file_: 包含机械臂关节角度数据（每组6个关节角度，单位：弧度）的文件路径（字符串）。
        :param degree_: 输入角度单位是角度还是弧度（布尔值）。默认值为弧度。
        :return: 相机到基座的变换矩阵（4x4 浮点数数组）。
        """
        # 1️⃣ 获取相机对棋盘的位姿
        T_target2cams, idxs = self._get_pose_target2cam(images_dir_)
        # 2️⃣ 获取机械臂末端到基座的位姿
        T_base2ends = self._get_pose_base2end(hand_angle_file_, idxs, degree_=degree_)

        TAs = []
        TBs = []
        Ras = []
        Rbs = []
        for i in range(len(T_target2cams)):
            for j in range(i + 1, len(T_target2cams)):
                TA = np.linalg.inv(T_base2ends[j]) @ T_base2ends[i]
                TB = T_target2cams[j] @ np.linalg.inv(T_target2cams[i])
                Ra = Rotation.from_matrix(TA[:3, :3]).as_rotvec()
                Rb = Rotation.from_matrix(TB[:3, :3]).as_rotvec()
                TAs.append(TA)
                TBs.append(TB)
                Ras.append(Ra)
                Rbs.append(Rb)

        remove = 0
        M = np.zeros((3, 3))
        S_all = []
        I3 = np.eye(3, dtype=np.float64)
        for i in range(len(TAs)):
            if np.linalg.norm(Ras[i]) < 0.1 or np.linalg.norm(Rbs[i]) < 0.1:  # ~6°
                remove += 1
                continue
            if abs(np.linalg.norm(Ras[i]) - np.pi) < 1e-3:
                remove += 1
                continue
            M += np.outer(Rbs[i], Ras[i])
            S_all.append(np.kron(TAs[i][:3, :3], I3) - np.kron(I3, TBs[i][:3, :3].transpose()))

        # 方法一求解
        U, S, Vt = np.linalg.svd(M)
        R_cam2base1 = Vt.T @ U.T

        if np.linalg.det(R_cam2base1) < 0:
            Vt[-1, :] *= -1
            R_cam2base1 = Vt.T @ U.T

        # 方法二求解
        S_all = np.vstack(S_all)
        U, S, Vt = np.linalg.svd(S_all)
        vec_RX = Vt[-1]
        RX_star = vec_RX.reshape((3, 3))
        Ux, Sx, Vx = np.linalg.svd(RX_star)
        R_cam2base2 = Ux @ Vx

        if np.linalg.det(R_cam2base2) < 0:
            Vx[-1, :] *= -1
            R_cam2base2 = Ux @ Vx

        Ta = []
        Tb1 = []
        Tb2 = []
        for i in range(len(TAs)):
            if np.linalg.norm(Ras[i]) < 0.1 or np.linalg.norm(Rbs[i]) < 0.1:  # ~6°
                continue
            if abs(np.linalg.norm(Ras[i]) - np.pi) < 1e-3:
                continue
            Ta.append(TAs[i][:3, :3] - np.eye(3))
            Tb1.append((R_cam2base1 @ TBs[i][:3, 3] - TAs[i][:3, 3]).reshape((3, 1)))
            Tb2.append((R_cam2base2 @ TBs[i][:3, 3] - TAs[i][:3, 3]).reshape((3, 1)))
        Ta = np.vstack(Ta)
        Tb1 = np.vstack(Tb1)
        Tb2 = np.vstack(Tb2)

        t_cam2base1, _, _, _ = np.linalg.lstsq(Ta, Tb1, rcond=None)
        t_cam2base2, _, _, _ = np.linalg.lstsq(Ta, Tb2, rcond=None)

        print("num of image pairs removed:",remove)
        X1_ = np.eye(4)
        X1_[0:3, 0:3] = R_cam2base1
        X1_[0:3, 3] = t_cam2base1.squeeze()
        se3_residual1_ = self._SE3_residual(X1_, T_base2ends, T_target2cams)

        X2_ = np.eye(4)
        X2_[0:3, 0:3] = R_cam2base2
        X2_[0:3, 3] = t_cam2base2.squeeze()
        se3_residual2_ = self._SE3_residual(X2_, T_base2ends, T_target2cams)

        return X1_, se3_residual1_, X2_, se3_residual2_

    def _SE3_residual(self, TX, T_base2ends, T_target2cams):
        """
        计算手眼标定结果的 SE(3) 残差。

        :param TX: 相机到基座的变换矩阵（4x4 浮点数数组）。
        :param T_base2ends: 机械臂末端到基座的位姿变换矩阵列表（每个元素为 4x4 浮点数数组）。
        :param T_target2cams: 相机对棋盘的位姿变换矩阵列表（每个元素为 4x4 浮点数数组）。
        :return: 平均 SE(3) 残差（浮点数）。
        """
        res = []
        for i in range(len(T_base2ends)):
            for j in range(i + 1, len(T_base2ends)):
                # 相对运动
                TA = np.linalg.inv(T_base2ends[j]) @ T_base2ends[i]
                TB = T_target2cams[j] @ np.linalg.inv(T_target2cams[i])
                res.append(np.linalg.norm(TA @ TX - TX @ TB, ord='fro'))
        return np.mean(res)


if __name__ == "__main__":
    calib_data_dir = "calib_data1920x1080"
    images_dir = os.path.join(calib_data_dir, "images")
    hand_angle_file = os.path.join(calib_data_dir, "hand_angle.txt")
    hand_eye_calib = HandEyeCalib()

    # eye-to-hand calibration
    X, se3_residual = hand_eye_calib.run_opencv_hand_to_eye(images_dir, hand_angle_file, degree_=False)
    X1, se3_residual1, X2, se3_residual2 = hand_eye_calib.run_my_hand_to_eye(images_dir, hand_angle_file, degree_=False)
    print("openCV residual:", se3_residual)
    print("my1 residual:", se3_residual1)
    print("my2 residual:", se3_residual2)
    print(f"camera to base transform matrix X = \n{X}")
    print(f"camera to base transform matrix X1 = \n{X1}")
    print(f"camera to base transform matrix X2 = \n{X2}")

    # npz 格式保存
    np.savez(os.path.join(calib_data_dir, "T_cam2base.npz"),
             residual_opencv=se3_residual, T_opencv = X,
             residual_my1=se3_residual1, T_my1=X1,
             residual_my2=se3_residual2, T_my2=X2)
    # calib_data = np.load(os.path.join(calib_data_dir, "T_cam2base.npz"))
    # T_residual,T = calib_data["residual_opencv"],calib_data["T_opencv"]
    # T1_residual,T1 = calib_data["residual_my1"],calib_data["T_my1"]
    # T2_residual,T2 = calib_data["residual_my2"],calib_data["T_my2"]

    # yaml 格式保存
    calib_data = {"residual_opencv": float(se3_residual), "T_opencv": X.tolist(),
            "residual_my1": float(se3_residual1), "T_my1": X1.tolist(),
            "residual_my2": float(se3_residual2), "T_my2": X2.tolist()}
    # with open(os.path.join(calib_data_dir, "T_cam2base.yaml"), "w") as f:
    #     yaml.dump(calib_data, f)
    # with open(os.path.join(calib_data_dir, "T_cam2base.yaml")) as f:
    #     calib_data = yaml.safe_load(f)
    # T_residual,T = np.array(calib_data["residual_opencv"]),np.array(calib_data["T_opencv"])
    # T1_residual,T1 = np.array(calib_data["residual_my1"]),np.array(calib_data["T_my1"])
    # T2_residual,T2 = np.array(calib_data["residual_my2"]),np.array(calib_data["T_my2"])
