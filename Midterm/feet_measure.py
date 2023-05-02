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

# %% Read input images
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


imgL = cv.imread(f"{ROOT / 'left view 04.jpg'}", cv.IMREAD_REDUCED_COLOR_4)
imgR = cv.imread(f"{ROOT / 'right view 04.jpg'}", cv.IMREAD_REDUCED_COLOR_4)
height, width, channel = imgL.shape


imshow("imgL", imgL)
imshow("imgR", imgR)

# %%


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


imgL_open_close, imgR_open_close = grayscale_denoise(
    imgL, imgR, kern_med_size=15)

imshow("imgL_open_close", imgL_open_close, save=True)
imshow("imgR_open_close", imgR_open_close, save=True)

# %%


def get_edges(img, canny_th1=0, canny_th2=200, dilate_k_size=5):
    canny = cv.Canny(img, canny_th1, canny_th2)
    dilate_k = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (dilate_k_size, dilate_k_size))
    return cv.dilate(canny, dilate_k)


cannyL = get_edges(imgL_open_close)
cannyR = get_edges(imgR_open_close)

imshow("cannyL", cannyL)
imshow("cannyR", cannyR)
# %%


def draw_contours(contours, size: Tuple[int, int], all: bool = False, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3):
    h, w = size[:2]
    h = round(h)
    w = round(w)
    if all:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv.drawContours(img, contours, -1, color=color, thickness=thickness)
        imshow(f"contour", img, hold=True)
    else:
        for i, cnt in enumerate(contours):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv.drawContours(img, [cnt], 0, color=color, thickness=thickness)
            imshow(f"contour {i}", img, hold=True)
    plt.show()


contoursL, _ = cv.findContours(cannyL, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contoursR, _ = cv.findContours(cannyR, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

draw_contours(contoursL, (height, width))
draw_contours(contoursR, (height, width))
# %%


def hull(contours, debug_img_size: Tuple[int, int] = None, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3):
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


hullL = hull(contoursL, (height, width))
hullR = hull(contoursR, (height, width))
# %% Get 4 corner hull


def get_quadrilateral(hulls, arcLen_weight: float = 2e-2, debug_img_size: Tuple[int, int] = None, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3):
    results = []
    for i, hull in enumerate(hulls):
        epsilon = arcLen_weight * cv.arcLength(hull, True)
        corners = cv.approxPolyDP(hull, epsilon, True).squeeze()
        if len(corners) == 4:
            results.append((hull, corners))
        if debug_img_size is not None:
            h, w = debug_img_size[:2]
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv.drawContours(img, [hull], 0, color=color, thickness=thickness)
            imshow(f"quadrilateral{i}", img, hold=True)
        results.append(hull)
    plt.show
    return results


quadrilateralL = get_quadrilateral(hullL, debug_img_size=(height, width))
quadrilateralR = get_quadrilateral(hullR, debug_img_size=(height, width))

# %% Get 2nd largest quadrilateral
paperL, corners_L = sorted(
    quadrilateralL, reverse=True, key=lambda cnt: cv.contourArea(cnt[0]))[1]
paperR, corners_R = sorted(
    quadrilateralR, reverse=True, key=lambda cnt: cv.contourArea(cnt[0]))[1]

draw_contours([paperL], (height, width))
draw_contours([paperR], (height, width))

# %%


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


corners_L = order_quadrilateral_corners(corners_L)
corners_R = order_quadrilateral_corners(corners_R)

print(corners_L)
print(corners_R)
# %% Get corners of destination A4 image


def inch_to_numPix(width: float, height: float, ppi: int = 96, precision=1e-3):
    w = width * round(ppi)
    h = height * round(ppi)
    w_r = round(w)
    h_r = round(h)
    # using epsilon not !=, because arithmetic limit/precision of computer
    rounded = (abs(w - w_r) > precision) or (abs(h - h_r) > precision)
    return w_r, h_r, rounded


a4_w, a4_h = 8.3, 11.7
dst_width, dst_height, rounded = inch_to_numPix(a4_w, a4_h, 100)
dst_corners = np.array([[0, 0],
                        [dst_width, 0],
                        [dst_width, dst_height],
                        [0, dst_height]])
# %% Homography to get paper
# src -> dst, Use least-square
H_L, mask_L = cv.findHomography(corners_L, dst_corners, 0)
H_R, mask_R = cv.findHomography(corners_R, dst_corners, 0)


imgL_pers = cv.warpPerspective(cannyL, H_L, (dst_width, dst_height),
                               flags=cv.INTER_CUBIC,
                               borderMode=cv.BORDER_CONSTANT)
imgR_pers = cv.warpPerspective(cannyR, H_R, (dst_width, dst_height),
                               flags=cv.INTER_CUBIC,
                               borderMode=cv.BORDER_CONSTANT)

imshow("imgL_pers", imgL_pers, save=True)
imshow("imgR_pers", imgR_pers, save=True)
# %% Get foot & its contour by not-foot region


def foot_contour(foot_edge_img, pad_width=3, debug_img_size=None):
    # foot black edge, white bg
    th, b_edge = cv.threshold(foot_edge_img, 0, 255,
                              cv.THRESH_OTSU + cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(b_edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hulls = hull(contours, debug_img_size=debug_img_size)
    # Get non-foot area's contour, corresponding to largest hull
    cnt, h = sorted(zip(contours, hulls), reverse=True,
                    key=lambda z: cv.contourArea(z[1]))[0]
    # Build img of foot
    foot_bw = np.zeros((dst_height, dst_width), dtype=np.uint8)
    # All white image
    foot_bw.fill(255)
    # Fill non-foot area to black
    cv.drawContours(foot_bw, [cnt], 0, (0, 0, 0), thickness=cv.FILLED)
    # Padd boarder with black
    foot_bw[:, 0:pad_width] = foot_bw[:, -pad_width:] = 0
    foot_bw[0:pad_width, :] = foot_bw[-pad_width:, :] = 0
    # Find the foot contour
    foot_cnts, _ = cv.findContours(
        foot_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    foot_cnt = sorted(foot_cnts, reverse=True,
                      key=lambda cnt: cv.contourArea(cnt))[0]
    return foot_cnt, foot_bw


footL_cnt, footL = foot_contour(imgL_pers)
footR_cnt, footR = foot_contour(imgR_pers)

imshow("footL", footL)
imshow("footR", footR)
draw_contours([footL_cnt], (dst_height, dst_width))
draw_contours([footR_cnt], (dst_height, dst_width))

# %% Find top & bottom points


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


def top_bottom_points(contour, epsilon=(5, 5), topAngle=270, bottomAngle=90,
                      debug_img_size=None):
    topPnts, origin = points_at_angle(contour, topAngle, epsilon=epsilon[0])
    top = sorted(topPnts, key=lambda p: p[1])[0]

    bottomPnts, _ = points_at_angle(
        contour, bottomAngle, epsilon=epsilon[1], origin=origin)
    bottom = sorted(bottomPnts, reverse=True, key=lambda p: p[1])[0]
    if debug_img_size is not None:
        h, w = debug_img_size[:2]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw contour
        cv.drawContours(img, [contour], 0, (0, 255, 0), 2)
        # Draw centroid
        cv.circle(img, (round(origin[0]), round(origin[1])),
                  radius=3, color=(0, 255, 255), thickness=3)
        for pnts, pnt in [(topPnts, top), (bottomPnts, bottom)]:
            # Draw point @ angle
            for p in pnts:
                cv.circle(img, tuple(p), radius=3,
                          color=(0, 0, 255), thickness=3)
            # Draw extreme point
            cv.circle(img, tuple(pnt), radius=3,
                      color=(255, 255, 255), thickness=3)
        imshow(f"contour extreme point", img)
    return top, bottom, origin


topL, bottomL, originL = top_bottom_points(footL_cnt,
                                           topAngle=270 - 10,
                                           bottomAngle=90 + 10,
                                           epsilon=(20, 20),
                                           debug_img_size=(dst_height, dst_width))
# %%
topR, bottomR, originR = top_bottom_points(footR_cnt,
                                           topAngle=270 + 10,
                                           bottomAngle=90 - 10,
                                           epsilon=(20, 20),
                                           debug_img_size=(dst_height, dst_width))
# %% Make splitting line


def get_line(pnt1, pnt2):
    pnt1 = np.array(pnt1).squeeze()
    pnt2 = np.array(pnt2).squeeze()
    pnt1 = np.append(pnt1, 1)
    pnt2 = np.append(pnt2, 1)
    return np.cross(pnt1, pnt2)


lineL = get_line(topL, bottomL)
lineR = get_line(topR, bottomR)
# %% Make mask


def get_x(line, y):
    a, b, c = line[0], line[1], line[2]
    return (-c - b * y) / a


def fill_half_img(img_size, line, left=True):
    h, w = img_size[:2]
    img = np.zeros((h, w), dtype=np.uint8)
    pnt_x = 0 if left else w
    top = (pnt_x, 0)
    bottom = (pnt_x, h)
    top_split = (get_x(line, 0), 0)
    bottom_split = (get_x(line, h), h)
    poly = [top, top_split, bottom_split, bottom] if left else [
        top_split, top, bottom, bottom_split]
    poly = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv.fillPoly(img, [poly], (255, 255, 255))
    return img


footL_mask = fill_half_img((dst_height, dst_width), lineL, left=True)
footR_mask = fill_half_img((dst_height, dst_width), lineR, left=False)
imshow("halfL_mask", footL_mask)
imshow("halfR_mask", footR_mask)
# %% Cut foot by not-foot region
footL_half = cv.bitwise_and(cv.bitwise_not(footL), footL_mask)
footR_half = cv.bitwise_and(cv.bitwise_not(footR), footR_mask)

imshow("imgL_half", footL_half)
imshow("imgR_half", footR_half)
# %% Combine 2 half feet


def shift_img(from_pnt, to_pnt, img):
    h, w = img.shape[:2]
    shift = to_pnt - from_pnt
    M = np.float32([[1, 0, shift[0]],
                    [0, 1, shift[1]]])
    return cv.warpAffine(img, M, (w, h))


foot_whole = cv.bitwise_not(
    cv.add(footL_half, shift_img(topR, topL, footR_half)))
imshow("img_whole", foot_whole)

# %%

contours, hierarchy = cv.findContours(
    foot_whole, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
foot_cnt = sorted(contours, reverse=True,
                  key=lambda cnt: cv.contourArea(cnt))[0]

draw_contours([foot_cnt], (dst_height, dst_width))

# %% bounding rectangle

bound_rect = cv.minAreaRect(foot_cnt)  # (x, y), (w, h), ang


# %% Draw foot & bounding box

# Draw foot
img = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
cv.drawContours(img, [foot_cnt], -1, color=(255,
                255, 255), thickness=cv.FILLED)
# Draw bounding box
box = cv.boxPoints(bound_rect)
box = np.int0(box)  # bottom-left, top-left, top-right, bottom-right
cv.drawContours(img, [box], -1, color=(0, 255, 0), thickness=3)


def pix2len(n_pix, ref_n_pix, ref_len, inch2cm=True):
    results = []
    for (p, rp, rl) in zip(n_pix, ref_n_pix, ref_len):
        l = p / rp * rl
        if inch2cm:
            l *= 2.54
        results.append(l)
    return results


# Draw width & height
box_wh = bound_rect[1]
foot_h, foot_w = pix2len([box_wh[1], box_wh[0]], [dst_height, dst_width], [
                         a4_h, a4_w], inch2cm=True)
cv.putText(img, f"width={foot_w:.2f}cm, height={foot_h:.2f}cm", tuple(box[0] + (-50, 50)),
           fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
           color=(0, 255, 255), thickness=3, lineType=cv.LINE_AA)
imshow(f"M10902117", img, save=True)

# %%
