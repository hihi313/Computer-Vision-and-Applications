# %% import
import numpy as np
from pathlib import Path
import re
import cv2 as cv
import matplotlib.pyplot as plt

# %% Global vars

ROOT = Path("/app/HW01")
TRAJ_DIR = ROOT / "Trajectory.xyz"
IMG_DIR = ROOT / "StadiumSnap.jpg"
OUTPUT_DIR = ROOT / "outputs"

K = np.array([[1.28e+03, 0.00e+00, 9.60e+02], [0.00e+00,
             1.28e+03, 5.40e+02], [0.00e+00, 0.00e+00, 1.00e+00]])

T = np.array([[6.4278758e-01, -7.6604450e-01, 1.2365159e-08, -1.8081500e+02], [-1.9826689e-01, -1.6636568e-01, -
             9.6592581e-01, 3.4364292e-01], [7.3994207e-01, 6.2088513e-01, -2.5881904e-01, 3.8508780e+02]])

# %% Utils


def imshow(name: str, img: np.ndarray, save: bool = False, ext: str = "jpg"):
    fig = plt.figure(name)
    if img.shape[-1] == 1 or img.ndim == 2:
        plt.imshow(img, cmap='gray')
        plt.colorbar()
    else:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    if save:
        plt.imsave(f"{str(OUTPUT_DIR)}/{name}.{ext}", img)
    fig.show()

# %% Read trajectory points


rows = []
line_split_rx = re.compile("\s+")
with open(str(TRAJ_DIR)) as file:
    for line in file:
        row = np.fromstring(line.strip(), sep=" ")
        row = np.append(row, 1)
        rows.append(row)

trajectory_3D_homo = np.array(rows)
# %% Project into frame

trajectory_2D_homo = K @ T @ trajectory_3D_homo.T
trajectory_2D, z = trajectory_2D_homo[0:2, :], trajectory_2D_homo[2, :]
trajectory_2D = (trajectory_2D / z).T

# %% Draw onto image

# Order by x axis
trajectory_2D = trajectory_2D[trajectory_2D[:, 0].argsort()]

img = cv.imread(str(IMG_DIR))
h, w, c = img.shape
img_test = np.zeros((h, w, c), dtype=np.uint8)
point_color = (255, 255, 255)  # BGR
point_radius = 1

for i in range(len(trajectory_2D) - 1):
    p = trajectory_2D[i]
    pn = trajectory_2D[i + 1]
    x, y = round(p[0]), round(p[1])
    xn, yn = round(pn[0]), round(pn[1])
    if (x >= 0 and x <= w) and (y >= 0 and y <= h) and (xn >= 0 and xn <= w) and (yn >= 0 and yn <= h):
        # Solid circle point (with radius, to see clearly)
        img = cv.line(img, (x, y), (xn, yn), color=point_color,
                      thickness=point_radius)
        # For debugging
        # img_test[y][x] = point_color
        # img_test[yn][xn] = point_color

imshow("M10902117", img, True)
imshow("test", img_test, True)
# %%
