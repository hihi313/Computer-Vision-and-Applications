# %% Import
import copy
import imghdr
import os
from pathlib import Path
from typing import List, Tuple
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# %% Global vars
ROOT = Path("/app/Final")  # Path(os.path.dirname(os.path.abspath('__file__')))
OUTPUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT / "SidebySide"
CALI_FILE = IMG_DIR / "CalibrationData.txt"

K_l = np.array([[1496.880651, 0.000000, 605.175810],
                [0.000000, 1490.679493, 338.418796],
                [0.000000, 0.000000, 1.000000]],
               dtype=np.float64)
K_r = np.array([[1484.936861, 0.000000, 625.964760],
                [0.000000, 1480.722847, 357.750205],
                [0.000000, 0.000000, 1.000000]],
               dtype=np.float64)
RT_l = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64)
RT_r = np.array([[0.893946, 0.004543, 0.448151, -186.807456, ],
                 [0.013206, 0.999247, -0.036473, 3.343985, ],
                 [-0.447979, 0.038523, 0.893214, 45.030463, ]],
                dtype=np.float64)
F = np.array([[0.000000191234, 0.000003409602, -0.001899934537],
              [0.000003427498, -0.000000298416, -0.023839273818],
              [-0.000612047140, 0.019636148869, 1.000000000000]],
             dtype=np.float64)

# %% Utils


def imshow(name: str, img: np.ndarray, save: bool = False, hold: bool = False, ext: str = "png"):
    gray = img.ndim == 2  # img.shape[-1] == 1 or
    is_float = img.dtype.kind == 'f'

    img = img if is_float else cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cmap = "gray" if gray else None

    fig = plt.figure(name)
    plt.imshow(img, cmap=cmap)
    if gray:
        plt.colorbar()
    if save:
        plt.imsave(f"{str(OUTPUT_DIR)}/{name}.{ext}", img)
    if not hold:
        fig.show()

# %% Load images


def load_images(dir: Path):
    files = os.listdir(f"{dir}")
    files.sort()
    images = []
    for f in files:
        f = f"{dir / f}"
        # Check it's image
        if imghdr.what(f) is not None:
            img = cv.imread(f)
            if img is not None:
                images.append(img)
    return images
# %% Split images


def split_images(images: List[np.ndarray], to_gray: bool = True):
    w = images[0].shape[1]
    half = w//2
    result = []
    for img in images:
        L, R = img[:, :half], img[:, half:]
        if to_gray:
            L = cv.cvtColor(L, cv.COLOR_BGR2GRAY)
            R = cv.cvtColor(R, cv.COLOR_BGR2GRAY)
        result.append((L, R))
    return result
# %% Get brightest point


def get_max_px_row(img: np.ndarray):
    h, w = img.shape[:2]
    blur = cv.GaussianBlur(img, (1, 7), sigmaX=0,
                           borderType=cv.BORDER_REPLICATE)
    # imshow("blur", blur)
    ys = [i for i in range(h)]
    # Get max column index of each row, currently only using max()
    max_xs = np.argmax(blur, axis=1)  # max Illumination's x of each y
    # Compose 2D index
    result = np.array([max_xs, ys]).T
    return result
# %% Draw points


def draw_point(image: np.ndarray, point: np.ndarray,
               radius: int = 5, thickness: int = 1, alpha: float = 0.5,
               color: Tuple[int, int, int] = (0, 255, 0)):
    if image.ndim or image.shape[2] < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    img_black = copy.copy(image)  # np.zeros_like(image)
    # Sift right/match point x coordinate
    # Draw points
    for i in range(len(point)):
        # left point
        cv.circle(img_black, point[i], radius, color, thickness)
    # Overlay image
    return cv.addWeighted(img_black, alpha, image, (1-alpha), 0)
# %% Get epipolar lines


def homogeneous(points: np.ndarray):
    """
        points: N*2
    """
    N, s = points.shape[:2]
    assert points.ndim == 2 and N > 0 and s == 2
    return np.append(points, np.ones((N, 1)), axis=1)  # N*3, last col is 1s


def epilines(F: np.ndarray, points: np.ndarray, is_left: bool,
             normalize: bool = False):
    assert F.shape == (3, 3)
    # # To 3*N homogeneous points
    # pnts = homogeneous(points).T
    # epilines = (F @ pnts).T  # N*3
    # # normlaize the lines, by first 2 element's norm
    # if normalize:
    #     epilines /= np.linalg.norm(epilines[:, :2],
    #                                ord=2, axis=1, keepdims=True)
    whichImg = 1 if is_left else 2
    points = points.reshape(-1, 1, 2)
    # N*1*3
    lines = cv.computeCorrespondEpilines(points, whichImg, F)
    return lines.squeeze()


# images = load_images(IMG_DIR)
# image_pairs = split_images(images)
# # A test image pair
# idx = 100
# L_img = image_pairs[idx][0]
# R_img = image_pairs[idx][1]
# imshow("L_img", L_img)
# imshow("L_img", R_img)
# # 1. Brightest pixel each row
# # Get top N brightest point as match candidate?
# L_px = get_max_px_row(L_img)
# R_px = get_max_px_row(R_img)
# # Display to check
# imshow("L max points", draw_point(L_img, L_px, radius=1), True)
# imshow("R max points", draw_point(R_img, R_px, radius=1), True)
# # 2. Get R points near L epilines
# L_px = L_px.astype(np.float64)
# R_epilines = epilines(F, L_px, True)

# %% Get patch score image


def get_patch(image: np.ndarray, anchor: Tuple[int, int], patch_size: int):
    """
        anchor: (x, y), usually at center of the patch
        patch is square
        margin left = margin top >= margin right = margin bottom
    """
    n_row, n_col = image.shape[:2]
    # anchor must within image
    assert (0 <= anchor[1] and anchor[1] <= n_row) and (
        0 <= anchor[0] and anchor[0] <= n_col)
    assert patch_size <= min(n_row, n_col) and isinstance(patch_size, int)

    ml = mt = mr = mb = patch_size // 2
    # even size patch, anchor is at bottom right corner of the center
    if patch_size % 2 == 0:
        mr -= 1
        mb -= 1
    top_row = anchor[1] - mt
    bottom_row = anchor[1] + mb
    left_col = anchor[0] - ml
    right_col = anchor[0] + mr

    # boundary handle, add area to another available region, anchor won't at center any more
    row_idx_limit = n_row - 1
    col_idx_limit = n_col - 1
    if top_row < 0:  # vertical boundary
        bottom_row += -top_row  # add the out of range area
        top_row = 0
    elif bottom_row > row_idx_limit:
        top_row -= (bottom_row - row_idx_limit)
        bottom_row = row_idx_limit
    # horizontal boundary
    if left_col < 0:
        right_col += -left_col
        left_col = 0
    elif right_col > col_idx_limit:
        left_col -= (right_col - col_idx_limit)
        bottom_row = col_idx_limit
    return image[round(top_row):round(bottom_row+1), round(left_col):round(right_col+1)]


def match_score_img(points, img, img_another,
                    patch_size=8, score_method=cv.TM_SQDIFF_NORMED):
    # {pts: score_img}
    score_imgs = []
    for p in points:
        patch = get_patch(img, p, patch_size)
        score_img = cv.matchTemplate(img_another, patch, score_method)
        score_imgs.append(score_img)
    return np.array(score_imgs)


# score_method = cv.TM_SQDIFF_NORMED
# L_pts_sImgs = match_score_img(L_px, L_img, R_img, score_method=score_method)
# %% Get max/min score along epiline on score image


def get_match_on_score_img(epilines, pts_scoreImgs, score_method):
    assert len(epilines) == len(pts_scoreImgs)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    take_min = True if score_method in [
        cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED] else False
    matches = []
    for line, sImg in zip(epilines, pts_scoreImgs):
        # ax+by+c=0 -> y = -(c+ax) / b
        a, b, c = line
        # Draw ROI, the epilines
        roi_mask = np.zeros_like(sImg, dtype=np.uint8)
        cv.line(roi_mask, (0, round(-c/b)), (roi_mask.shape[1], round(-(a*roi_mask.shape[1]+c)/b)),
                (255, 255, 255), 3, lineType=cv.LINE_4)
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(sImg, mask=roi_mask)
        matches.append(minLoc if take_min else maxLoc)
    return np.array(matches)


# L_matches = get_match_on_score_img(
#     R_epilines, L_pts_sImgs, score_method=score_method)
# L_matches = L_matches.astype(np.float64)
# %% Draw match


def draw_match(image_l: np.ndarray, point_l: np.ndarray,
               image_r: np.ndarray, match_pnts: np.ndarray,
               mask=None,
               radius: int = 2, thickness: int = 1,
               match_color: Tuple[int, int, int] = (0, 255, 0),
               unmatch_color: Tuple[int, int, int] = (0, 0, 255)):
    N = len(point_l)
    assert N == len(match_pnts)
    h, w = image_l.shape[:2]
    h_r, w_r = image_r.shape[:2]
    assert h == h_r

    # concate image
    image = np.concatenate((image_l, image_r), axis=1)
    if image.ndim or image.shape[2] < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    img_black = copy.copy(image)  # np.zeros_like(image) + 255
    # Sift right/match point x coordinate
    match = [(round(m[0]+w), round(m[1])) if m is not None else None
             for m in match_pnts]
    # Draw points
    for i in range(N):
        color = unmatch_color if mask is not None and not mask[i] else match_color
        # left point
        cv.circle(img_black, point_l[i].round().astype(np.int32),
                  radius, color, thickness)
        # right point
        cv.circle(img_black, match[i],
                  radius, color, thickness)
        # line
        cv.line(img_black, point_l[i].round().astype(np.int32), match[i],
                color, thickness, lineType=cv.LINE_AA)
    # Overlay image
    alpha = 0.5
    return cv.addWeighted(img_black, alpha, image, (1-alpha), 0)


# imshow("match", draw_match(L_img, L_px, R_img, L_matches), True)
# %% Correct/adjust match

# Test correctMatches
# L_px_before, L_matches_before = np.copy(L_px), np.copy(L_matches)

# L_px, L_matches = cv.correctMatches(F, L_px_before.reshape(
#     1, -1, 2), L_matches_before.reshape(1, -1, 2))
# L_px = L_px.reshape(-1, 2)
# L_matches = L_matches.reshape(-1, 2)

# imshow("correct match", draw_match(L_img, L_px, R_img, L_matches), True)
# %% Test correctMatches

# print(L_px != L_px_before)
# print(L_matches != L_matches_before)

# %% Get 3D points by triangulation


def projection_matrix(K, Rt, inverse=True):
    # Get projection matrix
    R, t = Rt[:, :-1], Rt[:, -1].reshape(-1, 1)
    R_cw = R.T
    t_cw = -R.T @ t
    RT = np.hstack([R_cw, t_cw]) if inverse else np.hstack([R, t])
    P = K @ RT
    result = (P, R_cw, t_cw) if inverse else (P, R, t)
    return result

def check_triangulate(P_r, pts_raw, dist_th):
    assert pts_raw.shape[-1] == 4
    # Result check like: https://github.com/opencv/opencv/blob/a74fe2ec01d9218d06cb7675af633fc3f409a6a2/modules/calib3d/src/five-point.cpp#L516
    mask = pts_raw[:, 2] * pts_raw[:, 3] > 0
    # N*4, Normalize
    pts_3d = pts_raw / pts_raw[:, 3].reshape(-1, 1)
    mask = (pts_3d[:, 2]< dist_th) & mask
    # N*3. points after projection
    pts = (P_r @ pts_3d.T).T
    mask = (pts[:, 2] > 0) & (pts[:, 2] < dist_th) & mask
    return pts_3d[:, :3], mask


def get_3D_points(P_l: np.ndarray, points_l: np.ndarray,
                  P_r: np.ndarray, points_r: np.ndarray,
                  z_th: float = 1000.0):
    """
        P: 3*4
        points: N*2
    """
    # All input must be float type to make the cv.triangulatePoints works correctly
    assert np.issubdtype(P_l.dtype, np.floating) and np.issubdtype(points_l.dtype, np.floating) and np.issubdtype(
        P_r.dtype, np.floating) and np.issubdtype(points_r.dtype, np.floating)
    points_3d_tri = cv.triangulatePoints(
        P_l, P_r, points_l.T, points_r.T).T  # 4*N -> N*4
    return check_triangulate(P_r, points_3d_tri, z_th)


# P_l, R_l, t_l = projection_matrix(K_l, RT_l, False)
# P_r, R_r, t_r = projection_matrix(K_r, RT_r, False)

# points_3d, mask = get_3D_points(P_l, L_px, P_r, L_matches)
# %% Get 3d points DLT


def get_3D_points_DLT(P_l: np.ndarray, points_l: np.ndarray,
                      P_r: np.ndarray, points_r: np.ndarray,
                      z_th: float = 1000.0):
    p1_l = P_l[0, :].reshape(1, -1)  # 1*4
    p2_l = P_l[1, :].reshape(1, -1)
    p3_l = P_l[2, :].reshape(1, -1)
    p1_r = P_r[0, :].reshape(1, -1)
    p2_r = P_r[1, :].reshape(1, -1)
    p3_r = P_r[2, :].reshape(1, -1)
    ul = points_l[:, 0].reshape(-1, 1)  # N*1
    vl = points_l[:, 1].reshape(-1, 1)
    ur = points_r[:, 0].reshape(-1, 1)
    vr = points_r[:, 1].reshape(-1, 1)
    a1 = ul * p3_l - p1_l
    a2 = vl * p3_l - p2_l
    a3 = ur * p3_r - p1_r
    a4 = vr * p3_r - p2_r
    # N*4*4
    A = np.stack((a1, a2, a3, a4), axis=1)
    U, S, VH = np.linalg.svd(A)
    # N*4*4
    V = np.transpose(VH.conj(), axes=(0, 2, 1))
    # last column, N*4
    x = V[:, :, -1]
    return check_triangulate(P_r, x, z_th)


# points_3d_dlt = get_3D_points_DLT(P_l, L_px, P_r, L_matches)


# %% Get 3d points inliers that reprojection error < th
def inlier_3d(points_3d: np.ndarray,
              K_l: np.ndarray, R_l: np.ndarray, t_l, points_l: np.ndarray,
              K_r: np.ndarray, R_r: np.ndarray, t_r, points_r: np.ndarray,
              mask=None, outlier_th: float = 2) -> np.ndarray:
    N = len(points_3d)
    assert N == len(points_l) and N == len(points_r)

    # Apply mask to remove outlier (w/ nan, inf, ...)
    # if mask is not None:
    #     assert len(mask) == N
    #     points_3d = points_3d[mask]
    #     points_l = points_l[mask]
    #     points_r = points_r[mask]

    # r_l, _ = cv.Rodrigues(R_l.T)
    # r_r, _ = cv.Rodrigues(R_r.T)

    # L, result is N*1*2
    # est_l, _=cv.projectPoints(points_3d, r_l, t_l, K_l, None)
    # est_l = est_l.squeeze()
    est_l = K_l @ (R_l @ points_3d.T + t_l)
    est_l = est_l.T
    est_l = est_l[:, :-1] / est_l[:, -1].reshape(-1, 1)
    # R
    # est_r, _=cv.projectPoints(points_3d, r_r, t_r, K_r, None)
    # est_r = est_r.squeeze()
    est_r = K_r @ (R_r @ points_3d.T + t_r)
    est_r = est_r.T
    est_r = est_r[:, :-1] / est_r[:, -1].reshape(-1, 1)

    # Reprojection distance/error
    err_l = np.linalg.norm(est_l - points_l, axis=-1, ord=2)
    err_r = np.linalg.norm(est_r - points_r, axis=-1, ord=2)
    err = (err_l + err_r) / 2.0
    # reprojection error should < TH
    inli_mask = err < outlier_th
    return inli_mask, err


# inlier_mask_dlt = inlier_3d(points_3d_dlt,
#                              K_l, R_l, t_l, L_px,
#                              K_r, R_r, t_r, L_matches, None)
# p3d_inline, inlier_mask, err = inlier_3d(points_3d,
#                                          K_l, R_l, t_l, L_px,
#                                          K_r, R_r, t_r, L_matches, mask=mask)
# np.savetxt(f"{OUTPUT_DIR / 'test.xyz'}", p3d_inline, delimiter=' ')
# %% Draw 3d outlier

# imshow("3d outlier", draw_match(
#     L_img, L_px[mask], R_img, L_matches[mask], inlier_mask), True)
# imshow("3d outlier dlt", draw_match(L_img, L_px,
#        R_img, L_matches, inlier_mask_dlt), True)

# %%
if __name__ == "__main__":
    images = load_images(IMG_DIR)
    image_pairs = split_images(images)
    result_3d = np.empty((0, 3))
    result_3d_dlt = np.empty((0, 3))
    P_l, R_l, t_l = projection_matrix(K_l, RT_l, False)
    P_r, R_r, t_r = projection_matrix(K_r, RT_r, False)
    z_th = 1000.0
    repro_th = 3
    for L_img, R_img in tqdm(image_pairs):
        # A image pair
        # 1. Brightest pixel each row
        L_px = get_max_px_row(L_img).astype(np.float64)
        # R_px = get_max_px_row(R_img)
        # 2. Get R points near L epilines
        R_epilines = epilines(F, L_px, True)
        # 3. Get L points' score image
        score_method = cv.TM_SQDIFF_NORMED
        L_pts_sImgs = match_score_img(
            L_px, L_img, R_img, score_method=score_method)
        # 4. Search match point along epiline on score image
        L_matches = get_match_on_score_img(
            R_epilines, L_pts_sImgs, score_method=score_method).astype(np.float64)
        # 4.1 Correct match
        L_px, L_matches = cv.correctMatches(
            F, L_px.reshape(1, -1, 2), L_matches.reshape(1, -1, 2))
        L_px = L_px.reshape(-1, 2)
        L_matches = L_matches.reshape(-1, 2)
        # 5. triangulate
        points_3d, mask = get_3D_points(P_l, L_px, P_r, L_matches, z_th=z_th)
        points_3d_dlt, mask_dlt = get_3D_points_DLT(
            P_l, L_px, P_r, L_matches, z_th=z_th)
        # 6. Get inlier if reprojection err < th
        # inlier mask
        inlier_mask, err = inlier_3d(points_3d,
                                     K_l, R_l, t_l, L_px,
                                     K_r, R_r, t_r, L_matches,
                                     outlier_th=repro_th)
        inlier_mask_dlt, err_dlt = inlier_3d(points_3d_dlt,
                                             K_l, R_l, t_l, L_px,
                                             K_r, R_r, t_r, L_matches,
                                             outlier_th=repro_th)
        # 7. get inliers
        p3d_inline = points_3d[mask & inlier_mask]
        p3d_inline_dlt = points_3d_dlt[mask_dlt & inlier_mask_dlt]
        # 8. save
        result_3d = np.append(result_3d, p3d_inline, axis=0)
        result_3d_dlt = np.append(result_3d_dlt, p3d_inline_dlt, axis=0)
        np.savetxt(f"{OUTPUT_DIR / 'ckpt.xyz'}", result_3d, delimiter=' ')
        np.savetxt(f"{OUTPUT_DIR / 'ckpt DLT.xyz'}",
                   result_3d_dlt, delimiter=' ')
    np.savetxt(f"{OUTPUT_DIR / 'result_3d.xyz'}", result_3d, delimiter=' ')
    np.savetxt(f"{OUTPUT_DIR / 'result_3d DLT.xyz'}",
               result_3d_dlt, delimiter=' ')
