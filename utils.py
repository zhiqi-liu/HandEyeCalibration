from roboticstoolbox import DHRobot, RevoluteDH, RevoluteMDH
import numpy as np


def ur3e_model():
    """
    构建 UR3e 六自由度机械臂的 DH 参数模型。

    本函数定义了 UR3e 机械臂的每个关节的 Denavit–Hartenberg (DH) 参数，
    并返回一个基于 `roboticstoolbox` 库的 `DHRobot` 对象。
    每个关节均为旋转关节（RevoluteDH）。

    Returns
    -------
    robot : DHRobot
        一个包含 6 个旋转关节的 UR3e 机械臂模型对象。

    Notes
    -----
    - 本模型基于 UR3e 的典型几何尺寸（以米为单位）。
    - 质量 `m`、质心位置 `r`、惯量 `I` 仅为近似值，用于动力学或仿真计算。
    - 每个关节的关节角限制（`qlim`）设置为 ±2π，即无限旋转范围。

    DH 参数说明
    ------------
    - `d` : 连杆偏移量（沿 z_{i-1}）
    - `a` : 连杆长度（沿 x_i）
    - `alpha` : 连杆扭转角（绕 x_i）
    - `offset` : 关节角偏置
    - `qlim` : 关节角限制范围
    - `m` : 连杆质量
    - `r` : 连杆质心位置向量（相对于连杆坐标系）
    - `I` : 惯性矩阵（若定义）
    """
    L1 = RevoluteDH(d=0.15185, a=0, alpha=np.pi / 2, offset=0, qlim=[-2 * np.pi, 2 * np.pi], m=1.98, r=[0, -0.02, 0])
    L2 = RevoluteDH(d=0, a=-0.24355, alpha=0, offset=0, qlim=[-2 * np.pi, 2 * np.pi], m=3.4445, r=[0.13, 0, 0.1157])
    L3 = RevoluteDH(d=0, a=-0.2132, alpha=0, offset=0, qlim=[-2 * np.pi, 2 * np.pi], m=1.437, r=[0.05, 0, 0.0238])
    L4 = RevoluteDH(d=0.13105, a=0, alpha=np.pi / 2, offset=0, qlim=[-2 * np.pi, 2 * np.pi], m=0.871, r=[0, 0, 0.01])
    L5 = RevoluteDH(d=0.08535, a=0, alpha=-np.pi / 2, offset=0, qlim=[-2 * np.pi, 2 * np.pi], m=0.805, r=[0, 0, 0.01])
    L6 = RevoluteDH(d=0.0921, a=0, alpha=0, offset=0, qlim=[-2 * np.pi, 2 * np.pi], m=0.261, r=[0, 0, -0.02],
                    I=[0, 0, 0.0001])
    robot = DHRobot([L1, L2, L3, L4, L5, L6], name='ur3e')
    return robot


def rm65_6FB_model():
    """
    构建 RM65-6FB 六自由度机械臂的 DH 参数模型。

    :return: DHRobot
        一个包含 6 个旋转关节的 RM65-6FB 机械臂模型对象。
    """
    # a(mm),alpha(deg),d(mm),offset(deg),qlim1(deg),qlim2(deg),dqlim(deg/s),m(kg),rx(mm),ry(mm),rz(mm),
    # Ixx(kg.mm^2),Ixy(kg.mm^2),Ixz(kg.mm^2),Iyy(kg.mm^2),Iyz(kg.mm^2),Izz(kg.mm^2)
    DH_ = [
        [0, 0, 240.5, 0, -178, 178, 180, 1.51, 0.491, 7.803, -10.744,
         2928.466, -32.63, -5.816, 2506.35, 47.925, 1756.017],
        [0, 90, 0, 90, -130, 130, 180, 1.653, 183.722, 0.103, -1.665,
         1711.553, -38.271, 2314.91, 70514.722, 6.507, 70036.186],
        [256, 0, 0, 90, -135, 135, 225, 0.726, 0.029, -90.105, 4.039,
         7259.884, 2.994, -0.314, 371.872, 44.451, 7228.758],
        [0, 90, 210, 0, -178, 178, 225, 0.671, 0.007, -9.486, -8.041,
         794.014, -0.821, -0.655, 596.235, -34.785, 486.228],
        [0, -90, 0, 0, -128, 128, 225, 0.647, 0.032, -83.769, 2.326,
         5375.604, 2.665, -0.304, 285.265, 14.235, 5359.769],
        [0, 90, 161.2, 0, -360, 360, 225, 0.248, -0.426, 0.237, -27.223,
         308.844, -3.781, -1.468, 304.616, 0.888, 122.62]
    ]
    L1 = RevoluteMDH(a=1e-3 * DH_[0][0],
                     alpha=np.deg2rad(DH_[0][1]),
                     d=1e-3 * DH_[0][2],
                     offset=np.deg2rad(DH_[0][3]),
                     qlim=[np.deg2rad(DH_[0][4]), np.deg2rad(DH_[0][5])],
                     m=DH_[0][7],
                     r=1e-3 * np.array([DH_[0][8], DH_[0][9], DH_[0][10]]),
                     I=1e-6 * np.array([DH_[0][11], DH_[0][14], DH_[0][16], DH_[0][12], DH_[0][15], DH_[0][13]]))
    L2 = RevoluteMDH(a=1e-3 * DH_[1][0],
                     alpha=np.deg2rad(DH_[1][1]),
                     d=1e-3 * DH_[1][2],
                     offset=np.deg2rad(DH_[1][3]),
                     qlim=[np.deg2rad(DH_[1][4]), np.deg2rad(DH_[1][5])],
                     m=DH_[1][7],
                     r=1e-3 * np.array([DH_[1][8], DH_[1][9], DH_[1][10]]),
                     I=1e-6 * np.array([DH_[1][11], DH_[1][14], DH_[1][16], DH_[1][12], DH_[1][15], DH_[1][13]]))
    L3 = RevoluteMDH(a=1e-3 * DH_[2][0],
                     alpha=np.deg2rad(DH_[2][1]),
                     d=1e-3 * DH_[2][2],
                     offset=np.deg2rad(DH_[2][3]),
                     qlim=[np.deg2rad(DH_[2][4]), np.deg2rad(DH_[2][5])],
                     m=DH_[2][7],
                     r=1e-3 * np.array([DH_[2][8], DH_[2][9], DH_[2][10]]),
                     I=1e-6 * np.array([DH_[2][11], DH_[2][14], DH_[2][16], DH_[2][12], DH_[2][15], DH_[2][13]]))
    L4 = RevoluteMDH(a=1e-3 * DH_[3][0],
                     alpha=np.deg2rad(DH_[3][1]),
                     d=1e-3 * DH_[3][2],
                     offset=np.deg2rad(DH_[3][3]),
                     qlim=[np.deg2rad(DH_[3][4]), np.deg2rad(DH_[3][5])],
                     m=DH_[3][7],
                     r=1e-3 * np.array([DH_[3][8], DH_[3][9], DH_[3][10]]),
                     I=1e-6 * np.array([DH_[3][11], DH_[3][14], DH_[3][16], DH_[3][12], DH_[3][15], DH_[3][13]]))
    L5 = RevoluteMDH(a=1e-3 * DH_[4][0],
                     alpha=np.deg2rad(DH_[4][1]),
                     d=1e-3 * DH_[4][2],
                     offset=np.deg2rad(DH_[4][3]),
                     qlim=[np.deg2rad(DH_[4][4]), np.deg2rad(DH_[4][5])],
                     m=DH_[4][7],
                     r=1e-3 * np.array([DH_[4][8], DH_[4][9], DH_[4][10]]),
                     I=1e-6 * np.array([DH_[4][11], DH_[4][14], DH_[4][16], DH_[4][12], DH_[4][15], DH_[4][13]]))
    L6 = RevoluteMDH(a=1e-3 * DH_[5][0],
                     alpha=np.deg2rad(DH_[5][1]),
                     d=1e-3 * DH_[5][2],
                     offset=np.deg2rad(DH_[5][3]),
                     qlim=[np.deg2rad(DH_[5][4]), np.deg2rad(DH_[5][5])],
                     m=DH_[5][7],
                     r=1e-3 * np.array([DH_[5][8], DH_[5][9], DH_[5][10]]),
                     I=1e-6 * np.array([DH_[5][11], DH_[5][14], DH_[5][16], DH_[5][12], DH_[5][15], DH_[5][13]]))
    robot = DHRobot([L1, L2, L3, L4, L5, L6], name='rm65_6FB')
    return robot


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
