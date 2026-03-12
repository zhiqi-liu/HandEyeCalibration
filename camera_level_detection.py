import cv2
import numpy as np
from camera import xvisio

# ===== 参数配置 =====
VIDEO_INDEX = 0  # 摄像头索引，0 = 第一个摄像头
BOARD_SIZE = (5, 4)  # 棋盘格内角点数量 (宽, 高)，实际棋盘格尺寸为 (6, 5) 个方格
SQUARE_SIZE = 0.05  # 棋盘格方格边长（单位：米），根据实际棋盘格尺寸调整
SUBPIX_WINDOW = (11, 11)  # 亚像素优化窗口大小，必须为奇数，通常设置为 (11, 11) 或 (5, 5)
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.1,
)  # 亚像素优化迭代条件：最大迭代次数 30，或精度达到 0.1 像素
WAITKEY_MS = 100
Z_PARALLEL_THRESHOLD = 1.0  # Z轴与相机光轴夹角小于该值（单位：度）时认为平行
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX

# ===== 生成棋盘格世界坐标点 =====
w, h = BOARD_SIZE
xx, yy = np.meshgrid(np.arange(w), np.arange(h))  # shape: (h, w)
object_points = np.column_stack(
    [
        (xx.ravel() - (w - 1) / 2) * SQUARE_SIZE,
        (yy.ravel() - (h - 1) / 2) * SQUARE_SIZE,
        np.zeros(w * h),
    ]
).astype(np.float32)

# ===== 获取相机内参和畸变参数 =====
K_cam, dist_coeffs, resolution = xvisio(resolution="high")

# ===== 设置摄像头和分辨率 =====
cap = cv2.VideoCapture(VIDEO_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")
print(f"摄像头分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

# ===== 主循环: 读取视频帧并检测棋盘格姿态 =====
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取失败")
            break

        # 检测角点
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE)

        if found:
            # 亚像素优化
            cv2.cornerSubPix(gray, corners, SUBPIX_WINDOW, (-1, -1), SUBPIX_CRITERIA)

            # 绘制棋盘角点（使用 OpenCV 内置函数）
            cv2.drawChessboardCorners(frame, BOARD_SIZE, corners, found)

            # PnP 求解位姿 target → camera
            success, rvec, tvec = cv2.solvePnP(object_points, corners, K_cam, dist_coeffs)

            if success:
                # 验证解
                proj_pts, _ = cv2.projectPoints(object_points, rvec, tvec, K_cam, dist_coeffs)
                err = np.linalg.norm(proj_pts.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)

                # 获取旋转矩阵并检查 Z 轴平行度
                R, _ = cv2.Rodrigues(rvec)
                z_board = R[:, 2]
                cosang = np.dot(z_board, [0.0, 0.0, 1.0]) / np.linalg.norm(z_board)
                z_angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

                # 输出诊断信息
                print(f"[检测成功]")
                print(f" - rvec={rvec.ravel()}")
                print(f" - Z角={z_angle:.2f}°")
                print(f" - 重投影误差: max={err.max():.3f} mean={err.mean():.3f}")

                # 在画面上显示 Z 轴状态
                if z_angle < Z_PARALLEL_THRESHOLD:
                    cv2.putText(frame, "Z轴平行", (10, 30), FONT_FACE, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Z轴夹角: {z_angle:.1f}°", (10, 30), FONT_FACE, 1, (0, 0, 255), 2)

        cv2.imshow("camera", frame)
        key = cv2.waitKey(WAITKEY_MS) & 0xFF
        if key == 27 or key == ord("q"):  # ESC 或 q 退出
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("资源已释放")
