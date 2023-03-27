# %% Import
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path
from numpy import linalg as LA

# %% Global vars

ROOT = Path("/app/HW02")
OUTPUT_DIR = ROOT / "outputs"

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


# %%
img = cv.imread("ArtGallery.jpg", cv.IMREAD_GRAYSCALE)
# img1 = np.zeros_like(img)
# img2 = np.zeros_like(img)

# img1[:, :1140] = img[:, :1140]
# img2[:, 1140:] = img[:, 1140:]

# imshow("img1", img1)
# imshow("img2", img2)

# %%
p = [(256, 211),
     (677, 309),
     (248, 945),
     (676, 874)]
p = np.array(p)
p_ = [(1378, 401),
      (1676, 326),
      (1381, 775),
      (1683, 806)]
p_ = np.array(p_)

N = min(len(p), len(p_)) # #points
# %%

A = np.zeros((N*2, 9))
for i in range(0, N):
    A[i*2, :] = [0, 0, 0,
               -p[i, 0], -p[i, 1], -1,
               p_[i, 1]*p[i, 0], p_[i, 1]*p[i, 1], p_[i, 1]]
    A[i*2+1, :] = [p[i, 0], p[i, 1], 1,
               0, 0, 0,
               -p_[i, 0]*p[i, 0], -p_[i, 0]*p[i, 1], -p_[i, 0]]

# %%
U, S, V = LA.svd(A)
H = V[:, -1].reshape((3, 3))
# %%

