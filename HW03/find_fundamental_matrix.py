# %%
import copy
import os
from pathlib import Path
from typing import Tuple
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


ROOT = Path(os.path.dirname(os.path.abspath('__file__')))
POINTS_FILE = ROOT / "points.csv"
OUTPUT_DIR = ROOT / "outputs"
IMG_L = ROOT / "L.jpg"
IMG_R = ROOT / "R.jpg"
# %% Read images
img_l = cv.imread(f"{IMG_L}", cv.IMREAD_GRAYSCALE)
img_r = cv.imread(f"{IMG_R}", cv.IMREAD_GRAYSCALE)
h, w = img_l.shape
# %% Read data from CSV file
data = np.genfromtxt(f"{POINTS_FILE}", delimiter=",", dtype=int)
data = data[1:]
# N = len(data)
pnts_l = data[:, :2]  # N*2
pnts_r = data[:, 2:]  # N*2
# %% Compare with SIFT keypoints


def SIFT_keypoints(img_l, img_r, th_dist=0.8, debug: bool = False):
    '''
        Using SIFT keypoint to compare the result
    '''
    sift = cv.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(img_l, None)
    kp_r, des_r = sift.detectAndCompute(img_r, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_l, des_r, k=2)  # l:query, r:train
    matches = sorted(matches, key=lambda m: m[0].distance)
    # Apply ratio test
    good = []
    pnts_l = []
    pnts_r = []
    for m, n in matches:
        if m.distance < th_dist*n.distance:
            good.append([m])
            pnts_l.append(kp_l[m.queryIdx].pt)
            pnts_r.append(kp_r[m.trainIdx].pt)
    if debug:
        # cv.drawMatchesKnn expects list of lists as matches.
        sift_matches = cv.drawMatchesKnn(
            img_l, kp_l, img_r, kp_r, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        imshow("sift_matches", sift_matches, save=True)
    return np.array(pnts_l, dtype=np.float64), np.array(pnts_r, dtype=np.float64)


# pnts_l, pnts_r = SIFT_keypoints(img_l, img_r, debug=False)
# sift_n = -1
# pnts_l = pnts_l[:sift_n]
# pnts_r = pnts_r[:sift_n]
# %%


def get_F(pnts_l, pnts_r, method=cv.FM_RANSAC, th=3.0, confid=0.99):
    # Find F
    F, mask = cv.findFundamentalMat(pnts_l, pnts_r,
                                    method=method, ransacReprojThreshold=th,
                                    confidence=confid)
    # Make F rank 2
    U, S, VT = np.linalg.svd(F.astype(np.float64))
    S[-1] = 0
    F_rank2 = U @ np.diag(S) @ VT
    return F, F_rank2, mask


# F, left to right
F, F_rank2, mask = get_F(pnts_l, pnts_r)
# Ramove outlier by mask
pntsIn_l = pnts_l[mask.ravel() == 1]
pntsIn_r = pnts_r[mask.ravel() == 1]
F = F_rank2

# Compare with ground truth
F = F_rank2 = np.array([[5.675e-08, -0.0000006, -0.0017387],
                        [0.0000019, 3.741e-13, -0.0483528],
                        [0.0009205, 0.0473451, 1]], dtype=np.float64)

print(f"F = \n{F}")
print(f"F_rank2 = \n{F_rank2}")
# %%


def get_epipolar_lines(F: np.ndarray, points: np.ndarray):
    '''
        Get epipolar lines
    '''
    # Epipolat lines
    ones = np.ones((len(points), 1), dtype=np.float64)  # N*2
    pnts_1 = np.append(points, ones, axis=1)  # N*3
    lines = (F @ pnts_1.T).T  # N*3, [l_1' ... l_N']=F [x_1 ... x_N]
    return lines


# Get epipolar lines of all points
lines_l = get_epipolar_lines(F.T, pnts_r)
lines_r = get_epipolar_lines(F, pnts_l)
# %%


def get_boundary_points(lines: np.ndarray, img_width: int):
    '''
        Get intersection of line & image boundary as start/end point

        Left & right image boundary
    '''
    # Get instersect points of epipolar lines and image boundary
    N = len(lines)
    # x=0, 1x+0y+0=0
    pnts_x0 = np.cross(lines, [[1, 0, 0]]*N)
    pnts_x0 /= pnts_x0[:, -1][:, None]
    # pnts_x0 = np.rint(pnts_x0).astype(int)
    pnts_x0 = pnts_x0[:, :2]
    # x=w, 1x+0y-w=0
    pnts_xw = np.cross(lines, [[1, 0, -img_width]]*N)
    pnts_xw /= pnts_xw[:, -1][:, None]
    # pnts_xw = np.rint(pnts_xw).astype(int)
    pnts_xw = pnts_xw[:, :2]
    return pnts_x0, pnts_xw


line_l_pnts = get_boundary_points(lines_l, w)
line_r_pnts = get_boundary_points(lines_r, w)
# %%


def draw(lines: np.ndarray, points_l: np.ndarray, points_r: np.ndarray,
         points: np.ndarray, img_ref: np.ndarray,
         color: Tuple[int, int, int] = (0, 255, 0), radius: int = 3,
         thickness: int = 3):
    '''
        Draw epipolar lines & points on given image
    '''
    # lines: N*3, points: N*2
    img = cv.cvtColor(copy.copy(img_ref), cv.COLOR_GRAY2BGR)
    for l, pl, pr, p in zip(lines, points_l, points_r, points):
        # ax+by+c=0, y=-c/b
        # x0,y0 = map(int, [0, -l[2]/l[1] ])
        # aw+by+c=0, y=(-c-aw)/b
        # x1,y1 = map(int, [w, -(l[2]+l[0]*w)/l[1] ])
        cv.line(img, tuple(np.rint(pl).astype(int)), tuple(np.rint(pr).astype(int)),
                color=color, thickness=thickness)
        # cv.line(img, (x0, y0), (x1, y1), color=color, thickness=thickness)
        cv.circle(img, tuple(np.rint(p).astype(int)),
                  radius=radius, color=(0, 255, 255), thickness=thickness)
    return img


img_l_epi = draw(lines_l, line_l_pnts[0], line_l_pnts[1], pnts_l, img_l)
img_r_epi = draw(lines_r, line_r_pnts[0], line_r_pnts[1], pnts_r, img_r)

imshow("L epipolar lines", img_l_epi, save=True)
imshow("R epipolar lines", img_r_epi, save=True)
# %% Print errors


def get_norm_lines(lines: np.ndarray):
    '''
    Get normalized lines
    '''
    # compute the l2 norms of [a b] for each line
    norms = np.linalg.norm(lines[:, :2], axis=1)
    return lines / norms[:, None]


def get_point_line_dist(points: np.ndarray, lines: np.ndarray):
    '''
    Compute point-line distance
    '''
    ones = np.ones((len(points), 1), dtype=np.float64)  # N*2
    pnts_1 = np.append(points, ones, axis=1)  # N*3
    # Normaliz line
    lines_norm = get_norm_lines(lines)  # N*3
    dists = []
    for l, p in zip(lines_norm, pnts_1):
        # 1*3 x 3*1
        dists.append(np.absolute(l @ p.T))
    return np.array(dists, dtype=np.float64).squeeze()


dist_l = get_point_line_dist(pnts_l, lines_l)
dist_r = get_point_line_dist(pnts_r, lines_r)

# Print results
formatter = {'float': lambda x: f'{x:.3e}'}
print(f"F (rank = 2) = \n{np.array2string(F_rank2, formatter=formatter)}")
print(f"Left image error: \n{np.array2string(dist_l, formatter=formatter)}")
print(f"Right image error: \n{np.array2string(dist_r, formatter=formatter)}")

# %%
