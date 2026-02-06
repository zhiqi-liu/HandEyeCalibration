from scipy.spatial.transform import Rotation
from robotic_arm import *
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

robot = rm65_6FB_model()
angle = [np.deg2rad(0),np.deg2rad(0),np.deg2rad(0),np.deg2rad(0),np.deg2rad(0),np.deg2rad(0)]
T = robot.fkine(angle)

r = Rotation.from_matrix(T.R)
print(T)
print(r.as_euler('xyz', degrees=False))
print(T.t)

robot.plot(angle)
plt.draw()
plt.pause(0.1)
plt.show(block=True)