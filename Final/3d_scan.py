# %% Import
import copy
import imghdr
import os
from pathlib import Path
from typing import List, Tuple
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# %% Global vars
ROOT = Path("/app/Final") #Path(os.path.dirname(os.path.abspath('__file__')))
OUTPUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT/ "SidebySide"
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
    gray = img.ndim == 2 #img.shape[-1] == 1 or
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

images = load_images(IMG_DIR)

# %% Split images

def split_images(imges: List[np.ndarray], to_gray: bool = True):
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

image_pairs = split_images(images)

# %% Testing images
idx = 33
imgL = image_pairs[idx][0]
imgR = image_pairs[idx][1]
imshow("L", imgL)
imshow("R", imgR)
# %%

def get_max_px_row(img:np.ndarray):
    h, w = img.shape[:2]
    blur = cv.GaussianBlur(img, (1, 7), sigmaX=0, borderType=cv.BORDER_REPLICATE)
    # imshow("blur", blur)
    ys = [i for i in range(h)]
    # Get max column index of each row, currently only using max()
    max_xs = np.argmax(blur, axis=1) # max Illumination's x of each y
    # Compose 2D index
    result = np.array([max_xs, ys]).T
    return result

L_px = get_max_px_row(imgL)
R_px = get_max_px_row(imgR) # Get top N brightest point as match candidate?
# %% draw point

def draw_point(image:np.ndarray, point:np.ndarray, 
               radius: int = 5, thickness: int = 1, alpha: float = 0.5,
               color: Tuple[int, int, int] = (0, 255, 0)):
    if image.ndim or image.shape[2] < 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    img_black = copy.copy(image)#np.zeros_like(image)
    # Sift right/match point x coordinate
    # Draw points
    for i in range(len(point)):
        # left point
        cv.circle(img_black, point[i], radius, color, thickness)
    # Overlay image
    return cv.addWeighted(img_black, alpha, image, (1-alpha), 0)

imgL_max_pts = draw_point(imgL, L_px, radius=1)
imgR_max_pts = draw_point(imgR, R_px, radius=1)

imshow("imgL_max_pts", imgL_max_pts, True)
imshow("imgR_max_pts", imgR_max_pts, True)
# %% Get epipolar lines

def homogeneous(points: np.ndarray):
    """
        points: N*2
    """
    N, s = points.shape[:2]
    assert points.ndim == 2 and N > 0 and s == 2
    return np.append(points, np.ones((N, 1)), axis=1) # N*3, last col is 1s

def get_epilines(F: np.ndarray, points: np.ndarray, normalize: bool = False):
    assert F.shape == (3, 3)
    # To 3*N homogeneous points
    pnts = homogeneous(points).T
    epilines = (F @ pnts).T # N*3
    # normlaize the lines, by first 2 element's norm
    if normalize:
        epilines /= np.linalg.norm(epilines[:, :2], ord=2, axis=1, keepdims=True)
    return epilines

L_epilines = get_epilines(F, L_px, True)
# %% Get points near epilines

def get_points_near_epilines(epilines:np.ndarray, candidate_points:np.ndarray,
                             distance_th: float = None, num_candidate: int = 3):
    """
        epilines: N*3
        candidate_points: candidate pixel locations, N_c*2
    """
    N = len(epilines)
    N_c = len(candidate_points)
    assert N > 0 and N_c > 0

    candi_pnts_homo = homogeneous(candidate_points)
    # N*3 * 3*N_c = N*N_c, N_c = #candidate points
    dist = abs(epilines @ candi_pnts_homo.T)

    # N*N_C*2, element is (candidate_point, distance)
    epi_candi = []
    for i in range(N):
        candis = []
        for j in range(N_c):
            candis.append((candidate_points[j], dist[i][j]))
        epi_candi.append(candis)

    # Sort by distance for each row
    epi_close = [sorted(candis, key=lambda c: c[1])
                 for candis in epi_candi]
    # Filter by condition provided & remove distance info
    if distance_th is not None:
        # For each row, check elements' distance < th
        epi_close = [[candi[0]
                      for candi in closes if candi[1] < distance_th]
                     for closes in epi_close]
    elif num_candidate is not None:
        # top N candidates/points close to epiline
        epi_close = [[candi[0]
                      for candi in closes[:num_candidate]]
                     for closes in epi_close]

    return epi_close

L_px_close = get_points_near_epilines(L_epilines, R_px, num_candidate=10)
# %% Patch scores of (candidate) points

def get_patch(image: np.ndarray, anchor: Tuple[int, int], patch_size: int):
    """
        anchor: (x, y), usually at center of the patch
        patch is square
        margin left = margin top >= margin right = margin bottom
    """
    n_row, n_col = image.shape[:2]
    # anchor must within image
    assert (0 <= anchor[1] or anchor[1] <= n_row) and (0 <= anchor[0] or anchor[0] <= n_col)
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
    if top_row < 0:# vertical boundary
        bottom_row += -top_row# add the out of range area
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
    return image[top_row:bottom_row+1, left_col:right_col+1]

def get_patch_score(image_l:np.ndarray, points_l:np.ndarray, 
                    image_r:np.ndarray, candidates_r:np.ndarray, 
                    patch_size:int = 8, score_method: int = cv.TM_SQDIFF_NORMED):
    """
        For each left point, get patch scores of its right (candidate) points
        points_l: N*2
        candidates_r: N*?, ? is #candidates of the left point. 
            (candidates on right image). May be empty
    """
    N = len(points_l)
    assert N == len(candidates_r)

    pts_scores = []
    # patchs' dict to reuse patches on right image
    patDic_r = {}
    for pt_l, pts_r in zip(points_l, candidates_r):
        patch_l = get_patch(image_l, pt_l, patch_size)
        scores = []
        for pt_r in pts_r:
            ptr = tuple(pt_r) # convert to tuple key
            patch_r = None
            if ptr in patDic_r:
                patch_r = patDic_r[ptr]
            else:
                patch_r = get_patch(image_r, pt_r, patch_size)
                patDic_r[ptr] = patch_r
            score = cv.matchTemplate(patch_l, patch_r, score_method)
            scores.append((pt_r, score))
        pts_scores.append(scores)
    return pts_scores

score_method = cv.TM_SQDIFF_NORMED
# N*?*2, ? = #candidates, 2= (candidate point, score)
L_px_scores = get_patch_score(imgL, L_px, imgR, L_px_close, score_method=score_method)
# %% Get match  by score

def get_match(px_scores: List[List[Tuple[Tuple[int, int], float]]], 
              score_method:int, score_th: float=None):
    matches = []
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    take_max = False if score_method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED] else True
    for scores in px_scores:
        # get matched point based on min/max score
        match = []
        if len(scores) > 0:
            # Sort by score
            match = sorted(scores, key=lambda l: l[1], reverse=take_max)
            # Filter by score_th if not provided
            if score_th is not None:
                if take_max:
                    match = [p_s for p_s in match
                             if p_s[1] > score_th]
                else:
                    match = [p_s for p_s in match
                             if p_s[1] < score_th]
        matches.append(match)
    return matches

L_px_matches = get_match(L_px_scores, score_method=score_method)
# Take first match's point only
L_px_matches = [match[0][0] if len(match) > 0 else match
                for match in L_px_matches]

print(len([m  for m in L_px_matches if len(m) > 0]))
# %% Draw match

def draw_match(image_l:np.ndarray, point_l:np.ndarray, 
               image_r:np.ndarray, match_pnts: np.ndarray,
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
    img_black = copy.copy(image) #np.zeros_like(image) + 255
    # Sift right/match point x coordinate
    match = [(m[0]+w, m[1]) if m is not None else None
             for m in match_pnts]
    # Draw points
    for i in range(N):
        if match[i] is not None:
            # left point
            cv.circle(img_black, point_l[i], radius, match_color, thickness)
            # right point
            cv.circle(img_black, match[i], radius, match_color, thickness)
            # line
            cv.line(img_black, point_l[i], match[i], match_color, thickness, lineType=cv.LINE_AA)
        else:
            # left point
            cv.circle(img_black, point_l[i], radius, unmatch_color, thickness)
    # Overlay image
    alpha = 0.5
    return cv.addWeighted(img_black, alpha, image, (1-alpha), 0)

imgL_max_pts = draw_match(imgL, L_px, imgR, L_px_matches)
imshow("tmp", imgL_max_pts, True)
# %%
print("sdf")
# %%
if __name__ == "__main__":
    plt.show()