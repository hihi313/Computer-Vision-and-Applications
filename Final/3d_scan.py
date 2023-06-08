# %% Import
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
    rows = [i for i in range(h)]
    # Get max column index of each row, currently only using max()
    max_col = np.argmax(blur, axis=1)
    # Compose 2D index
    result = np.array([rows, 
                       max_col]).T
    return result

L_px = get_max_px_row(imgL)
R_px = get_max_px_row(imgR) # Get top N brightest point as match candidate?
# %% Get epipolar lines

def homogeneous(points: np.ndarray):
    """
        points: N*2
    """
    N, s = points.shape[:2]
    assert points.ndim == 2 and N > 0 and s == 2
    return np.append(points, np.ones((N, 1)), axis=1)

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
# %% Match point


def get_patch(image: np.ndarray, anchor: Tuple[int, int], patch_size: int):
    """
        anchor: (x, y), usually at center of the patch
        patch is square
        margin left = margin top >= margin right = margin bottom
    """
    n_row, n_col = image.shape[:2]
    # anchor must within image
    assert (0 <= anchor[1] or anchor[1] <= n_row) and (0 <= anchor[0] or anchor[0] <= n_col)
    assert patch_size <= min(n_row, n_col)

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


def match_point(image_l:np.ndarray, points:np.ndarray, 
                image_r:np.ndarray, candidate_points:np.ndarray, epilines:np.ndarray,
                patch_size:int = 8):
    """
        Search matching point along *normalized* epiline(on right image)
    """
    assert isinstance(patch_size, int)
    N = len(points)
    assert N > 0 and N == len(epilines)

    candi_pnts_homo = homogeneous(candidate_points)
    # N*3 * 3*N_c = N*N_c, N_c = #candidate points
    dist = abs(epilines @ candi_pnts_homo.T)

    # C closest candidate point (at least 1 candidate for a point)
    C = 5 # TODO: to param
    # Sort by distance of each epiline, N*C, element is index of candidate points
    dist_idx = np.argsort(dist, axis=1)[:, :C] 
    # N*C*2. Conver index to point to compare
    candi_close = [candidate_points[dist_idx[i, :]] for i in range(N)]

    # Compare point's patch
    scores = [] # score of all epiline's closest points
    for i in range(N):
        score_pnts = [] # score of an epiline's closest points
        for j in range(C):
            patch_l = get_patch(image_l, candi_close[i][j], patch_size)
            patch_r = get_patch(image_r, candi_close[i][j], patch_size)
            score = cv.matchTemplate(patch_l, patch_r, cv.TM_CCOEFF_NORMED)
            score_pnts.append((score, j))
        scores.append(score_pnts)
    print("sdf")

match_point(imgL, L_px, imgR, R_px, L_epilines)
# %%
