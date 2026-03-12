from robotic_arm import rm65_6FB_model
from common import *
from object_detection import process_image_fast

import matplotlib
import matplotlib.pyplot as plt
from spatialmath import SE3, SO3
from scipy.spatial.transform import Rotation

matplotlib.use("TkAgg")

resolution = "high"

# 设置摄像头和分辨率
cap = cv2.VideoCapture(0)  # 0 = 第一个摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 获取相机中目标姿态（像素坐标、旋转角度）
# results = []
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("读取失败")
#         break

#     fram, results = process_image_fast(frame)
#     cv2.imshow("camera", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == 27:  # ESC 退出
#         break

# cap.release()
# cv2.destroyAllWindows()

# if len(results) == 0:
#     exit()

# # 获取像素坐标、预设深度（或tof得到）
# pt = results[0][0]
# theta = results[0][1]

pt = (1300, 600)
theta = 30
z = 0.776

# 转换到机械臂基坐标系
P_base = p_img2base(pt, z, resolution=resolution)

# 设置机械臂和夹爪
robot = rm65_6FB_model(gripper=True)

# 设置夹爪姿态
T_gripper2base = SE3()
T_gripper2base.R = SO3.Rz(np.deg2rad(-theta)) * SO3.Rx(np.pi)
T_gripper2base.t = P_base[0]
print(f"T_gripper2base:\n{T_gripper2base}")

T_end2base = T_gripper2base * robot.tool.inv()
print(f"T_gripper2end:\n{T_end2base}")
# print(Rotation.from_matrix(T_end2base.R).as_euler('xyz', degrees=False)) # 外旋, 欧拉角
# print(T_end2base.eul("xyz")) # 内旋, 欧拉角
print(Rotation.from_matrix(T_end2base.R).as_quat())  # 四元素[x,y,z,w]
print(T_end2base.t)

# 逆运动学求解
angle = robot.ikine_LM(T_gripper2base, q0=np.zeros(6), joint_limits=True, ilimit=2000)
print(angle.success)
print(f"angle(rad):\n{angle}")
print(f"angle(deg):\n{np.round(np.rad2deg(angle.q), 3)}")
print(f"pt: {pt}\ntheta: {theta}")
# 显示结果
robot.plot(angle.q)
plt.draw()
plt.pause(0.1)
plt.show(block=False)
