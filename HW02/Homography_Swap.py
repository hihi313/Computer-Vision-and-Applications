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
        if save:
            plt.imsave(f"{str(OUTPUT_DIR)}/{name}.{ext}", img)
    else:
        tmp = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(tmp)
        if save:
            plt.imsave(f"{str(OUTPUT_DIR)}/{name}.{ext}", tmp)
    fig.show()


def get_poly_mask(ref_img: np.ndarray, points: np.ndarray, inv: bool = False, dtype: np.dtype = np.uint8, line_type: int = cv.LINE_AA):
    mask = np.zeros_like(ref_img, dtype=dtype)
    cv.fillPoly(mask, [points.reshape((-1, 1, 2))],
                (255, 255, 255), lineType=line_type)
    imshow("mask", mask)
    if inv:
        inv_mask = cv.bitwise_not(mask)
        imshow("inv mask", inv_mask)
        return mask, inv_mask
    return mask, None


# %%
img = cv.imread("ArtGallery.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
N_row, N_col, N_channel = img.shape

# %%
pL = [(256, 211),
      (677, 309),
      (676, 874),
      (248, 945)]
pL = np.array(pL)

pR = [(1378, 401),
      (1676, 326),
      (1683, 806),
      (1381, 775)]
pR = np.array(pR)

# points
N = min(len(pL), len(pR))

maskR, inv_maskR = get_poly_mask(img_gray, pR, inv=True)
maskL, inv_maskL = get_poly_mask(img_gray, pL, inv=True)

# %%

# Find homography
H, mask = cv.findHomography(pL, pR, 0)  # Use least-square

# Left to right
frameR = cv.warpPerspective(img, H, (N_col, N_row),
                            flags=cv.INTER_CUBIC,
                            borderMode=cv.BORDER_CONSTANT)
frameR = cv.bitwise_and(frameR, frameR, mask=maskR)
imshow("frameR", frameR, save=True)

# Right to left
frameL = cv.warpPerspective(img, LA.inv(H), (N_col, N_row),
                            flags=cv.INTER_CUBIC,
                            borderMode=cv.BORDER_CONSTANT)
frameL = cv.bitwise_and(frameL, frameL, mask=maskL)
imshow("frameL", frameL, save=True)
# %%

# Fill left
img_bgL = cv.bitwise_and(img, img, mask=inv_maskL)
resultL = cv.add(img_bgL, frameL)
imshow("resultL", resultL)

# Fill right
img_bgR = cv.bitwise_and(resultL, resultL, mask=inv_maskR)
result = cv.add(img_bgR, frameR)
imshow("M10902117", result, save=True)

# %%
