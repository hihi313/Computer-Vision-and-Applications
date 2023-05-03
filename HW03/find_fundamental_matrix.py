# %%

import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

# %%


def imshow(name: str, img: np.ndarray, save: bool = False, hold: bool = False, ext: str = "jpg"):
    fig = plt.figure(name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cmap = "gray" if img.shape[-1] == 1 or img.ndim == 2 else None
    plt.imshow(img, cmap=cmap)
    # plt.colorbar()
    if save:
        plt.imsave(f"{str(OUTPUT_DIR)}/{name}.{ext}", img)
    if not hold:
        fig.show()


ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
POINTS_FILE = ROOT / "points.csv"
OUTPUT_DIR = ROOT / "outputs"
IMG_L = ROOT / "L.jpg"
IMG_R = ROOT / "R.jpg"

# %%

data = np.genfromtxt(f"{POINTS_FILE}", delimiter=",", dtype=int)
data = data[1:]
pnts_l = data[:, :2]
pnts_r = data[:, 2:]
img_l = cv.imread(f"{IMG_L}", cv.IMREAD_GRAYSCALE)
img_r = cv.imread(f"{IMG_R}", cv.IMREAD_GRAYSCALE)

# %%


def get_F(pnts_l, pnts_r):
    F, mask = cv.findFundamentalMat(pnts_l, pnts_r,
                                    method=cv.FM_RANSAC, ransacReprojThreshold=1.0,
                                    confidence=0.99)
    U, S, VT = np.linalg.svd(F)
    S[-1] = 0
    F_rank2 = U @ np.diag(S) @ VT
    return F, F_rank2


F, F_rank2 = get_F(pnts_l, pnts_r)

print(f"F = \n{F}")
print(f"F_rank2 = \n{F_rank2}")

# %%


def SIFT_keypoints(img_l, img_r, debug: bool = False):
    # Using keypoint results to reference/compare
    sift = cv.xfeatures2d.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(img_l, None)
    kp_r, des_r = sift.detectAndCompute(img_r, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_l, des_r, k=2) # query, train
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    good_sort = sorted(np.array(good).squeeze(), key = lambda m: m.distance)
    # Get matched keypoints' location into 2 numpy array
    pts_l = np.zeros((len(good_sort), 2), dtype=np.uint8)
    pts_r = np.zeros((len(good_sort), 2), dtype=np.uint8)
    for i, m in enumerate(good_sort):
        pts_l[i,:] = np.round(kp_l[m.queryIdx].pt).astype(int)
        pts_r[i,:] = np.round(kp_r[m.queryIdx].pt).astype(int)

    if debug:
        # cv.drawMatchesKnn expects list of lists as matches.
        sift_matches = cv.drawMatchesKnn(
            img_l, kp_l, img_r, kp_r, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        imshow("sift_matches", sift_matches, save=True)
    return pts_l, pts_r

pts_l, pts_r = SIFT_keypoints(img_l, img_r, debug=False)
F, F_rank2 = get_F(pts_l, pts_r)
print(f"F = \n{F}")
print(f"F_rank2 = \n{F_rank2}")

# %%
