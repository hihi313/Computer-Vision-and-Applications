# %% Import
import math
import os
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# %% Global vars


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
# TODO?: show multi images


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


def max_hull(*imgs: Tuple[np.ndarray], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3, debug: bool = False, draw_n: int = 3):
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
        # hull of max contour
        hull = cv.convexHull(contours[0])
        if debug:
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
    plt.show()
    return corners


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


# %%


imgL = cv.imread(f"{ROOT / 'left view 03.jpg'}", cv.IMREAD_REDUCED_COLOR_4)
imgR = cv.imread(f"{ROOT / 'right view 03.jpg'}", cv.IMREAD_REDUCED_COLOR_4)
height, width, channel = imgL.shape

imshow("imgL", imgL)
imshow("imgR", imgR)
# %%

imgL_open_close, imgR_open_close = grayscale_denoise(imgL, imgR)

imshow("imgL_open_close", imgL_open_close, save=True)
imshow("imgR_open_close", imgR_open_close, save=True)

# %%
imgL_hb, imgR_hb = high_boost(imgL_open_close, imgR_open_close, alpha=2)

imshow("imgL_hp", imgL_hb, save=True)
imshow("imgR_hp", imgR_hb, save=True)


# %% Binarize image, make sure the A4 paper is white

imgL_bw, imgR_bw = binarize(imgL_hb, imgR_hb, inv_bw=True)

imshow("imgL_bw", imgL_bw, save=True)
imshow("imgR_bw", imgR_bw, save=True)

# %%

L_hull, R_hull = max_hull(imgL_bw, imgR_bw, debug=True)

draw_contours(imgL_bw.shape, L_hull, R_hull)

# %%

corners = approx_quadrilateral(
    L_hull, R_hull, debug=True, img_size=imgL_bw.shape)

# %%


def inch_to_numPix(width: float, height: float, ppi: int = 96, epsilon=1e-3):
    w = width * round(ppi)
    h = height * round(ppi)
    w_r = round(w)
    h_r = round(h)
    # using epsilon not !=, because arithmetic limit/precision of computer
    rounded = (abs(w - w_r) > epsilon) or (abs(h - h_r) > epsilon)
    return w_r, h_r, rounded


a4_w, a4_h = 8.3, 11.7

dst_width, dst_height, rounded = inch_to_numPix(a4_w, a4_h, 100)

dst_corners = np.array([[0, 0],
                        [dst_width, 0],
                        [dst_width, dst_height],
                        [0, dst_height]])

# %%
# src -> dst, Use least-square
H, mask = cv.findHomography(corners, dst_corners, 0)


imgR_pers = cv.warpPerspective(imgR_bw, H, (dst_width, dst_height),
                               flags=cv.INTER_CUBIC,
                               borderMode=cv.BORDER_CONSTANT)

imshow("imgR_pers", imgR_pers, save=True)
# %%

# LSD
# # %%

# lsd = cv.line_descriptor.LSDDetector.createLSDDetector()
# klsd1 = lsd.detect(imgL_clahe, 2, 2)
# klsd2 = lsd.detect(imgR_clahe, 2, 2)

# bd = cv.line_descriptor.BinaryDescriptor.createBinaryDescriptor()
# _, lsd_des1 = bd.compute(imgL_clahe, klsd1)
# _, lsd_des2 = bd.compute(imgR_clahe, klsd2)

# # %%
# lsd_des = [lsd_des1, lsd_des2]
# octave0 = ([], [])
# des = ([], [])
# for i, klsd in enumerate([klsd1, klsd2]):
#     for j in range(len(klsd)):
#         if klsd[j].octave == 0:
#             octave0[i].append(klsd[j])
#             des[i].append(lsd_des[i][j])

# # %%
# bdm = cv.line_descriptor.BinaryDescriptorMatcher()
# lsd_matches = bdm.match(lsd_des1, lsd_des2)

# lsd_matches = sorted(lsd_matches, key = lambda x:x.distance)
# # %%
# good_matches = lsd_matches[:4]
# lsd_mask = np.ones_like(good_matches)
# img_lsd_match = None
# img_lsd_match = cv.line_descriptor.drawLineMatches(	imgL_l, octave0[0], imgR_l, octave0[1], good_matches, img_lsd_match, 3, -1, lsd_mask, cv.DrawMatchesFlags_DEFAULT)

# imshow("img_lsd_match", img_lsd_match, save=True)

# Keypoint
# # %%

# sift = cv.xfeatures2d.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(imgL_clahe,None)
# kp2, des2 = sift.detectAndCompute(imgR_clahe,None)
# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# # cv.drawMatchesKnn expects list of lists as matches.
# tmp = cv.drawMatchesKnn(imgL_clahe,kp1,imgR_clahe,kp2,good,None,flags=cv.DrawMatchesFlags_DEFAULT)

# imshow("tmp", tmp, save=True)
# %%
