"""
    主函数
"""
import numpy as np
import os
from robotic_arm import rm65_6FB_model
from camera import xvisio
from common import p_img2cam, p_cam2base
from spatialmath import SE3, SO3

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


calib_data_dir = "calib_data1920x1080"
calib_data = np.load(os.path.join(calib_data_dir, "T_cam2base.npz"))
candidates = [
    (calib_data["residual_opencv"], calib_data["T_opencv"]),
    (calib_data["residual_my1"], calib_data["T_my1"]),
    (calib_data["residual_my2"], calib_data["T_my2"])
]
residual_min, T_best = min(candidates, key=lambda x: x[0])
print("residual_min:", residual_min)
print(f"T_best:\n{T_best}")

T_cam2base = T_best
K_cam, dist_cam, _ = xvisio(resolution="high")

pt = (1213.30, 677.97)
z = [0.810]
P_cam = p_img2cam(pt, z, K_cam, dist_cam)
print(f"P_cam:\n{P_cam}")
P_base = p_cam2base(P_cam, T_cam2base)
print(f"P_base:\n{P_base}")

robot = rm65_6FB_model()
robot.tool = SE3(0,0,0.2038)

T_gripper2base = SE3()
T_gripper2base.R = SO3.Rx(np.pi)
T_gripper2base.t = P_base[0]
print(f"T_gripper2base:\n{T_gripper2base}")

angle = robot.ikine_LM(T_gripper2base)
print(f"angle:\n{angle}")
print(f"angle:\n{np.round(np.rad2deg(angle.q), 3)}")

robot.plot(angle.q)
plt.draw()
plt.pause(0.1)
plt.show(block=True)
