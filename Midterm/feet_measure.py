# %% Import
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path
from numpy import linalg as LA

# %% Global vars

ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
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