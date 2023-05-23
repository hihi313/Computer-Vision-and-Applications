# %% Import
import os
import math
from pathlib import Path
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
# %% Global vars
ROOT = Path("/app/HW04")#Path(os.path.dirname(os.path.abspath('__file__')))
OUTPUT_DIR = ROOT / "outputs"
POINTS_FILE = ROOT / "points.csv"

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


# %% Read data from CSV file
data = np.genfromtxt(f"{POINTS_FILE}", delimiter=",", dtype=int)
# Test
# data = np.genfromtxt(f"{ROOT / 'points test.csv'}", delimiter=",", dtype=int)
data = data[1:]
squares = [data[i:i+4, :] for i in range(0, len(data), 4)]
# %% Test with OpenCV
# objectPnts = []
# imgPnts = []
# for sq in squares:
#     objP = []
#     imgP = []
#     for pnt in sq:
#         objP.append((pnt[2], pnt[3], 0))
#         imgP.append((pnt[0], pnt[1]))
#     objectPnts.append(np.array(objP, dtype=np.float32))
#     imgPnts.append(np.array(imgP, dtype=np.float32))

# reprojErr, K, distCoef, rVec, tVec = cv.calibrateCamera(objectPnts, imgPnts, (1080, 1920), None, None)
# %%
Hs = [cv.findHomography(sq[:, 2:], sq[:, :2], 0)[0]
      for sq in squares]  # Use least-square
# %% Test data
Hs[0] = np.array([[379.199677, 111.830818, 152.000000],
                  [-64.140297, 350.826263, 149.000000],
                  [0.102074, 0.210233, 1.000000]])
Hs[1] = np.array([[168.271439, 125.763161, 596.000000],
                  [81.960945, 320.478027, 84.000000],
                  [-0.148918, 0.211012, 1.000000]])
Hs[2] = np.array([[228.969971, -209.572891, 490.000000],
                  [41.616714, 105.178200, 387.000000],
                  [-0.078244, -0.182428, 1.000000]])
# %%


def get_omega_coefs(Hs: np.ndarray):
    assert len(Hs) >= 3
    coef_mat = []
    for h in Hs:
        h = h.T
        # coef = [[h[1, 1]*h[1, 2], h[1, 1]*h[2, 2]+h[2, 1]*h[1, 2], h[1, 1]*h[3, 2]+h[3, 1]*h[1, 2], h[2, 1]*h[2, 2], h[2, 1]*h[3, 2]+h[3, 1]*h[2, 2], h[3, 1]*h[3, 2]],
        #         [h[1, 1]**2-h[1, 2]**2, 2*(h[1, 1]*h[2, 1]-h[1, 2]*h[2, 2]), 2*(h[1, 1]*h[3, 1]-h[1, 2]*h[3, 2]), h[2, 1]**2-h[2, 2]**2, 2*(h[2, 1]*h[3, 1] - h[2, 2]*h[3, 2]), h[3, 1]**2-h[3, 2]**2]]
        coef = [[h[0, 0]*h[1, 0], h[0, 0]*h[1, 1]+h[0, 1]*h[1, 0], h[0, 0]*h[1, 2]+h[0, 2]*h[1, 0], h[0, 1]*h[1, 1], h[0, 1]*h[1, 2]+h[0, 2]*h[1, 1], h[0, 2]*h[1, 2]],
                [h[0, 0]**2-h[1, 0]**2, 1*(h[0, 0]*h[0, 1]-h[1, 0]*h[1, 1]), 1*(h[0, 0]*h[0, 2]-h[1, 0]*h[1, 2]), h[0, 1]**2-h[1, 1]**2, 1*(h[0, 1]*h[0, 2]-h[1, 1]*h[1, 2]), h[0, 2]**2-h[1, 2]**2]]
        coef_mat.append(coef)
    return np.vstack(coef_mat)

def to_matrix(elements:np.ndarray, type:str="triangle"):
    '''
        To upper triangular/symmetric matrix
    '''
    if not isinstance(elements, np.ndarray):
        elements = np.array(elements)
    elements = elements.flatten()
    sqrt_b24ac = math.sqrt(1.0+8.0*len(elements))
    assert (sqrt_b24ac - 1) % 2 == 0, f"{len(elements)} elements can't form a symmetric matrix"
    N = int((-1 + sqrt_b24ac) / 2)
    sym = np.zeros((N, N))
    sym[np.triu_indices(3)] = elements
    if type == "symmetric":
        sym += sym.T - np.diag(sym.diagonal())
    return sym

def solve_omega(H: np.ndarray, method: str="svd"):
    if method != "svd":
        pass
    else:
        # Default is SVD
        U, S, VH = np.linalg.svd(H.astype(np.float64))
        V = VH.conj().T
        w = V[:, -1] # last column & normalize by last element
    return to_matrix(w, "symmetric")


def solve_K(w: np.ndarray, skew: bool = False):
    w_ = np.linalg.inv(w)
    w_ /= w_[-1][-1]
    # Test
    # w_ = 1e6*np.array([[1.5347, 0.1894, 0.0005315349], [0.1894, 1.3784, 0.000409567], [0.0005, 0.0004, 0.0000]])
    c = w_[0, 2]
    e = w_[1, 2]
    d = math.sqrt(w_[1, 1] - e**2)
    b = (w_[0, 1] - c*e) / d if skew else 0
    # Test
    # b = -25.7338
    a = math.sqrt(w_[0, 0] - b**2 - c**2)
    print(abs(w_[0, 1]-c*e))
    return to_matrix([a, b, c, d, e, 1], "triangle")

coef = get_omega_coefs(Hs)
w = solve_omega(coef)
K = solve_K(w, skew=True)
# %%

# def get_symmetric()
print("Symmetric matrix:\n", sym)
# %%
