# %% Import
import math
import os
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# %% Global vars
plt.ion()

ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
OUTPUT_DIR = ROOT / "outputs"

# %% Utils


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


def grayscale_denoise(*imgs: Tuple[np.ndarray],
                      kern_morph_size: int = 11,
                      kern_med_size: int = 11,
                      debug: bool = False):
    results = []
    kern_morph = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE,
                                          ksize=(kern_morph_size, kern_morph_size))
    for img in imgs:
        # Convert to luminance image, because i think it will have more constrast if the input image has high contrast
        img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        img_l, a, b = cv.split(img_lab)
        if debug:
            imshow("img_l", img_l, hold=True)
        # Median to remove noise
        img_med = cv.medianBlur(img_l, kern_med_size)
        if debug:
            imshow("img_med", img_med, hold=True)
        # Morphological open & closing to remove noise & smooth local pixel value
        img_open = cv.morphologyEx(img_med, cv.MORPH_OPEN, kern_morph)
        img_close = cv.morphologyEx(img_open, cv.MORPH_CLOSE, kern_morph)
        if debug:
            imshow("img_close", img_close, hold=True)
        # End processing
        results.append(img_close)
    plt.show()
    return results


def high_boost(*imgs: Tuple[np.ndarray], alpha: float = 2.0, kernel_size: int = 7, debug: bool = False):
    results = []
    for img in imgs:
        # LPF
        img_lp = cv.GaussianBlur(img, ksize=(
            kernel_size, kernel_size), sigmaX=0, sigmaY=0, borderType=cv.BORDER_CONSTANT)
        # HPF =
        img_hp = cv.addWeighted(img, alpha, img_lp, -1, 0)
        if debug:
            imshow("img_hp", img_hp, hold=True)
        results.append(img_hp)
    plt.show()
    return results


def binarize(*imgs: Tuple[np.ndarray], contrast: float = 2.0, inv_bw: bool = False):
    results = []
    inv = cv.THRESH_BINARY_INV if inv_bw else cv.THRESH_BINARY
    for img in imgs:
        # Increase contrast
        img_contrast = cv.convertScaleAbs(img, alpha=contrast)
        # Otsu threshold
        threshold, img_bw = cv.threshold(
            img_contrast, 0, 255, cv.THRESH_OTSU + inv)
        results.append(img_bw)
    return results


def max_contour(*imgs: Tuple[np.ndarray], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3, debug: bool = False, draw_n: int = 3):
    results = []
    for i, img in enumerate(imgs):
        contours, hierarchy = cv.findContours(
            img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Sort contour by max->min area
        contours = sorted(contours, reverse=True,
                          key=lambda cnt: cv.contourArea(cnt))
        h, w = img.shape[:2]
        if debug:
            # Draw n max contour
            n = draw_n if draw_n <= len(contours) else len(contours)
            for j in range(n):
                img_contour_j = np.zeros((h, w, 3), dtype=np.uint8)
                cv.drawContours(img_contour_j, contours, j,
                                color=color, thickness=thickness)
                imshow(f"img{i}_contour_{j}", img_contour_j, hold=True)
        # Save max contours
        results.append(contours[0])
    plt.show()
    return results


def max_hull(*imgs: Tuple[np.ndarray], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3, debug: bool = False, draw_n: int = 3):
    results = []
    contours = max_contour(
        *imgs, color=color, thickness=thickness, debug=False, draw_n=draw_n)
    for i, cnt in enumerate(contours):
        # hull of max contour
        hull = cv.convexHull(cnt)
        if debug:
            h, w = imgs[i].shape[:2]
            img_hull = np.zeros((h, w, 3), dtype=np.uint8)
            cv.drawContours(img_hull, [hull],
                            0, color=color, thickness=thickness)
            imshow(f"img{i}_hull", img_hull, hold=True)
        results.append(hull)
    plt.show
    return results


def draw_contours(size: Tuple, *contours: Tuple[np.ndarray], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3):
    h, w = size[:2]
    for i, cnt in enumerate(contours):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv.drawContours(img, [cnt], 0, color=color, thickness=thickness)
        imshow(f"contour {i}", img, hold=True)
    plt.show()


def approx_quadrilateral(*hulls, epsilon_weight: float = 2e-2, debug: bool = False,
                         img_size: Tuple = None, color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 3, font_color: Tuple[int, int, int] = (0, 0, 255),
                         font_thickness: int = 3):
    results = []
    # Approximate the convex quadrilateral
    for i, hull in enumerate(hulls):
        epsilon = epsilon_weight * cv.arcLength(hull, True)
        corners = cv.approxPolyDP(hull, epsilon, True).squeeze()
        # Get 4 farest point from center
        corners = sort_by_distance(corners)[:4]
        corners = order_quadrilateral_corners(corners)
        if debug and img_size is not None:
            h, w = img_size[:2]
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for j, c in enumerate(corners):
                cv.putText(img, f"{j}", tuple(c),
                           fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                           color=font_color, thickness=font_thickness, lineType=cv.LINE_AA)
                cv.circle(img, tuple(c), radius=3,
                          color=color, thickness=thickness)
                imshow(f"hull {i} corners", img, hold=True)
        results.append(corners)
    plt.show()
    return results


def sort_by_distance(points: np.ndarray, ref_point: np.ndarray = None):
    if ref_point is None:
        ref_point = np.mean(points, axis=0)
    assert ref_point.shape == points[0].shape, "Points' shapes not match"
    return np.array(sorted(points, key=lambda p: np.linalg.norm(p - ref_point)))


def order_quadrilateral_corners(corners: np.ndarray):
    # Rearrange quadrilateral corners to order: top-left, top-right, bottom-right, bottom-left
    assert len(corners) == 4, "Quadrilateral's #corners != 4"
    s = corners.sum(axis=1)
    # Top-left: smallest sum.
    tl = corners[np.argmin(s)]
    # Bottom-right: largest sum.
    br = corners[np.argmax(s)]

    d = np.diff(corners, axis=1)
    # Top-right: smallest difference.
    tr = corners[np.argmin(d)]
    # Bottom-left: largest difference.
    bl = corners[np.argmax(d)]
    return np.array([tl, tr, br, bl])


def inch_to_numPix(width: float, height: float, ppi: int = 96, precision=1e-3):
    w = width * round(ppi)
    h = height * round(ppi)
    w_r = round(w)
    h_r = round(h)
    # using epsilon not !=, because arithmetic limit/precision of computer
    rounded = (abs(w - w_r) > precision) or (abs(h - h_r) > precision)
    return w_r, h_r, rounded

# Angles of given points


def angle(points, origin=None):
    points = points.squeeze()
    if origin is None:
        # Calculate moments of contours
        M = cv.moments(points)
        # Calculate centroid of object
        origin = np.array((M['m10']/M['m00'], M['m01']/M['m00']))
    # Compute the vectors from origin to each point
    vectors = points - origin
    # Compute the angles of each vector with respect to the origin
    angles = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return angles % 360, origin


def points_at_angle(contour: np.ndarray, ang: float, origin: np.ndarray = None,
                    epsilon: float = 15, minN: int = 1, topN: float = 0.1):
    ang = ang % 360
    points = contour.squeeze()
    angles, origin = angle(points, origin)
    # Find the indices of the points that satisfy the condition
    indices = np.where(np.abs(angles - ang) < epsilon)
    # Select the points using the indices
    pntAng = points[indices]
    # If not enough point
    if len(pntAng) < minN and minN >= 1:
        N = max(round(len(points) * topN), round(minN))
        # Select top N closest deg point
        pntAng = points[np.argsort(np.abs(angles - ang))][:N]
    return pntAng, origin


def extend1(*points):
    # Should all be N-D point
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    points = points.squeeze()
    ones = np.ones((points.shape[0], 1))
    points_ones = np.concatenate((points, ones), axis=1)
    return points_ones


def draw_points(size: Tuple[int, int], points: np.ndarray, count: int = 10,
                radius: int = 3, color: Tuple[int, int, int] = (0, 255, 0),
                thickness: int = 3):
    h, w = size[:2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, p in enumerate(points):
        xy = (round(p[0]), round(p[1]))
        cv.circle(img, xy, radius=radius, color=color, thickness=thickness)
        if (count is not None) and (count > 0) and (i % count == 0):
            cv.putText(img, f"{i}", xy,
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                       color=(0, 255, 255), thickness=thickness, lineType=cv.LINE_AA)

    imshow("points", img)


def shift_contour(contour, from_pnt, to_pnt):
    shift = to_pnt - from_pnt
    return contour.squeeze() + np.rint(shift).astype(np.int8)


# %% Read input images
imgL = cv.imread(f"{ROOT / 'left view 03.jpg'}", cv.IMREAD_REDUCED_COLOR_4)
imgR = cv.imread(f"{ROOT / 'right view 03.jpg'}", cv.IMREAD_REDUCED_COLOR_4)
height, width, channel = imgL.shape

imshow("imgL", imgL)
imshow("imgR", imgR)

# %%
imgL_open_close, imgR_open_close = grayscale_denoise(imgL, imgR, kern_med_size=15)

imshow("imgL_open_close", imgL_open_close, save=True)
imshow("imgR_open_close", imgR_open_close, save=True)

# %%

cannyL = cv.Canny(imgL_open_close, 0, 200)
cannyR = cv.Canny(imgR_open_close, 0, 200)

dilate_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
cannyL = cv.dilate(cannyL, dilate_k)
cannyR = cv.dilate(cannyR, dilate_k)

imshow("cannyL", cannyL)
imshow("cannyR", cannyR)
# %%
h, w = imgL.shape[:2]
tmp = np.zeros((h, w, 3), dtype=np.uint8)

contoursL, _ = cv.findContours(cannyL, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contoursR, _ = cv.findContours(cannyR, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# %%
def hull(contours: Tuple[np.ndarray], debug_img_size: Tuple[int, int]=None, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3):
    results = []
    for i, cnt in enumerate(contours):
        # hull of max contour
        hull = cv.convexHull(cnt)
        if debug_img_size is not None:
            h, w = debug_img_size[:2]
            img_hull = np.zeros((h, w, 3), dtype=np.uint8)
            cv.drawContours(img_hull, [hull],
                            0, color=color, thickness=thickness)
            imshow(f"hull{i}", img_hull, hold=True)
        results.append(hull)
    plt.show
    return results

hullL = hull(contoursL, (h, w))

# %%
contoursL = sorted(contoursL, reverse=True, key=lambda cnt: cv.contourArea(cnt))[0]
contoursR = sorted(contoursR, reverse=True, key=lambda cnt: cv.contourArea(cnt))[0]

cv.drawContours(tmp, [contoursL], 0, color=(0, 255, 0), thickness=3)

draw_contours((h, w), contoursL, contoursR)
# %%
imgL_hb, imgR_hb = high_boost(imgL_open_close, imgR_open_close, alpha=2)

imshow("imgL_hp", imgL_hb, save=True)
imshow("imgR_hp", imgR_hb, save=True)
# %% Binarize image, make sure the A4 paper is white
imgL_bw, imgR_bw = binarize(imgL_hb, imgR_hb, contrast=2, inv_bw=True)

imshow("imgL_bw", imgL_bw, save=True)
imshow("imgR_bw", imgR_bw, save=True)
# %%
L_hull, R_hull = max_hull(imgL_bw, imgR_bw, debug=True, draw_n=1)
# draw_contours(imgL_bw.shape, L_hull, R_hull)
# %%
corners_L, corners_R = approx_quadrilateral(
    L_hull, R_hull, debug=True, img_size=imgL_bw.shape)
# %% Get corners of destination A4 image
a4_w, a4_h = 8.3, 11.7
dst_width, dst_height, rounded = inch_to_numPix(a4_w, a4_h, 100)
dst_corners = np.array([[0, 0],
                        [dst_width, 0],
                        [dst_width, dst_height],
                        [0, dst_height]])
# %% Get contour of feet
# src -> dst, Use least-square
H_L, mask_L = cv.findHomography(corners_L, dst_corners, 0)
H_R, mask_R = cv.findHomography(corners_R, dst_corners, 0)


imgL_pers = cv.warpPerspective(imgL_bw, H_L, (dst_width, dst_height),
                               flags=cv.INTER_CUBIC,
                               borderMode=cv.BORDER_CONSTANT)
imgR_pers = cv.warpPerspective(imgR_bw, H_R, (dst_width, dst_height),
                               flags=cv.INTER_CUBIC,
                               borderMode=cv.BORDER_CONSTANT)
# Invert to make feet white(content), in order to get content's contour
imgL_pers = cv.bitwise_not(imgL_pers)
imgR_pers = cv.bitwise_not(imgR_pers)


imshow("imgL_pers", imgL_pers, save=True)
imshow("imgR_pers", imgR_pers, save=True)
# %%
footL, footR = max_contour(imgL_pers, imgR_pers, debug=True, draw_n=1)
# %%  Using convex hull has better result, less points & eleminate concave points
footL_hull = cv.convexHull(footL)
footR_hull = cv.convexHull(footR)

topAngle = 270
topPntsL, originL = points_at_angle(footL_hull, topAngle, epsilon=15)
topPntsR, originR = points_at_angle(footR_hull, topAngle, epsilon=5)
topL = sorted(topPntsL, key=lambda p: p[1])[0]
topR = sorted(topPntsR, key=lambda p: p[1])[0]

bottomAngle = 90
bottomPntsL, _ = points_at_angle(
    footL_hull, bottomAngle, epsilon=5, origin=originL)
bottomPntsR, _ = points_at_angle(
    footR_hull, bottomAngle, epsilon=5, origin=originR)
bottomL = sorted(bottomPntsL, reverse=True, key=lambda p: p[1])[0]
bottomR = sorted(bottomPntsR, reverse=True, key=lambda p: p[1])[0]

# %% Show results
h, w = imgR_pers.shape[:2]
for i, (cnt, o, ts, t, bs, b) in enumerate([(footL, originL, topPntsL, topL, bottomPntsL, bottomL),
                                            (footR, originR, topPntsR, topR, bottomPntsR, bottomR)]):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv.drawContours(img, [cnt], 0, (0, 255, 0), 2)
    cv.circle(img, (round(o[0]), round(o[1])),
              radius=3, color=(0, 255, 255), thickness=3)
    for pnts, pnt in [(ts, t),
                      (bs, b)]:
        for p in pnts:
            cv.circle(img, tuple(p), radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img, tuple(pnt), radius=3,
                  color=(255, 255, 255), thickness=3)
    imshow(f"contour {i} extreme point", img, save=True, hold=True)
plt.show()

# %%
tl, tr = dst_corners[0], dst_corners[1]
tl_1, tr_1, topL_1, bottomL_1, topR_1, bottomR_1 = extend1(tl, tr, topL, bottomL, topR, bottomR)
footL_1 = extend1(footL)
footR_1 = extend1(footR)
lineL = np.cross(topL_1, bottomL_1)
lineR = np.cross(topR_1, bottomR_1)
# %%
# %%

def get_x(line, y):
    a, b, c = line[0], line[1], line[2]
    return (-c - b * y) / a

def fill_half_img(img_size,line, left=True):
    h, w = img_size[:2]
    img = np.zeros((h, w), dtype=np.uint8)
    pnt_x = 0 if left else w 
    top = (pnt_x, 0)
    bottom  = (pnt_x, h)
    top_split = (get_x(line, 0), 0)
    bottom_split = (get_x(line, h), h)
    poly = [top, top_split, bottom_split, bottom] if left else [top_split, top, bottom, bottom_split]
    poly = np.array(poly, dtype=np.int32).reshape((-1,1,2))
    cv.fillPoly(img,[poly],(255,255, 255))
    return img

halfL_mask = fill_half_img((dst_height, dst_width), lineL, left=True)
halfR_mask = fill_half_img((dst_height, dst_width), lineR, left=False)
imshow("halfL_mask", halfL_mask)
imshow("halfR_mask", halfR_mask)

# %%

imgL_half =  cv.bitwise_and(cv.bitwise_not(imgL_pers), halfL_mask)
imgR_half =  cv.bitwise_and(cv.bitwise_not(imgR_pers), halfR_mask)
imshow("imgL_half", imgL_half)
imshow("imgR_half", imgR_half)
# %%

def shift_img(from_pnt, to_pnt, img):
    h, w = img.shape[:2]
    shift = to_pnt - from_pnt
    M = np.float32([[1,0,shift[0]],
                    [0,1,shift[1]]])
    return cv.warpAffine(img,M,(w,h))

imgR_half_shift = shift_img(topR, topL, imgR_half)
# %%

img_whole = cv.bitwise_not(cv.add(imgL_half, imgR_half_shift))
imshow("img_whole", img_whole)

# %%

contours, hierarchy = cv.findContours(img_whole, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
foot_cnt = sorted(contours, reverse=True, key=lambda cnt: cv.contourArea(cnt))[0]

draw_contours((dst_height, dst_width), foot_cnt)

# %% bounding rectangle

bound_rect = cv.minAreaRect(foot_cnt) # (x, y), (w, h), ang


# %% Draw foot & bounding box

# Draw foot
img = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
cv.drawContours(img, [foot_cnt], -1, color=(255, 255, 255), thickness=cv.FILLED)
# Draw bounding box
box = cv.boxPoints(bound_rect)
box = np.int0(box) # bottom-left, top-left, top-right, bottom-right
cv.drawContours(img, [box], -1, color=(0, 255, 0), thickness=3)
# Draw width & height

def pix2len(n_pix, ref_n_pix, ref_len, inch2cm=True):
    results = []
    for (p, rp, rl) in zip(n_pix, ref_n_pix, ref_len):
        l = p / rp * rl
        if inch2cm:
            l *= 2.54
        results.append(l)
    return results

box_wh = bound_rect[1]
foot_h, foot_w = pix2len([box_wh[1], box_wh[0]], [dst_height, dst_width], [a4_h, a4_w], inch2cm=True)
cv.putText(img, f"width={foot_w:.2f}cm, height={foot_h:.2f}cm", tuple(box[0] + (-50, 50)) ,
            fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
            color=(0, 255, 255), thickness=3, lineType=cv.LINE_AA)


imshow(f"test", img)



# # %%
# # epsilon: numerically tolerance
# def select_same_side(points, line, ref_pnt, epsilon: float = 1e-12):
#     points = points.squeeze()
#     # Check all points' last element are 1
#     tmp = np.append(points, [ref_pnt], axis=0)[:, -1]
#     assert np.all(np.absolute(tmp - 1) < epsilon), "all point[-1] != 1"
#     sign = np.sign(ref_pnt.dot(line))
#     points_sign = np.sign(points.dot(line))
#     same_side_pnts = points[(points_sign == sign) | (points_sign == 0)]
#     return same_side_pnts[:, :-1]

# footR_half = select_same_side(footR_1, lineR, tr)
# footL_half = select_same_side(footL_1, lineL, tl)

# # %%

# draw_points((dst_height, dst_width), footR_half, count = 100)


# # %%
# draw_points((dst_height, dst_width), footL_half, count = 100)

# # %%

# footL = footL.squeeze()
# footL_half = footL[footL[:, 0] < topL[0]]

# footR = footR.squeeze()
# footR_half = footR[footR[:, 0] > topR[0]]
# # %%

# footR_half_shift = shift_contour(footR_half, topR, topL)

# # %%


# # %%
# # footL_sim = cv.estimateAffinePartial2D(	np.array([topL, bottomL]), np.array([topR, bottomR]), method=cv.RANSAC)

# # %%
# foot_whole = np.append(footL_half, footR_half_shift, axis=0)
# angles_whole, origin_whole = angle(foot_whole)
# foot_whole = foot_whole[np.argsort(angles_whole)]

# foot_whole = shift_contour(foot_whole, origin_whole,
#                            (dst_width/2, dst_height/2))

# draw_points((dst_height, dst_width), foot_whole, count=100)

# img_whole = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
# cv.drawContours(img_whole, [foot_whole], 0, color=(
#     255, 255, 255), thickness=cv.FILLED)
# imshow("foot whole", img_whole, save=True)

# %%
