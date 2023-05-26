# %% Import
import os
import math
from pathlib import Path
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
# %% Global vars
ROOT = Path("/app/HW04")  # Path(os.path.dirname(os.path.abspath('__file__')))
OUTPUT_DIR = ROOT / "outputs"
POINTS_FILE = ROOT / "points.csv"

# %% Utils


def imshow(name: str, img: np.ndarray, save: bool = False, hold: bool = False, ext: str = "png"):
    fig = plt.figure(name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cmap = "gray" if img.shape[-1] == 1 or img.ndim == 2 else None
    plt.imshow(img, cmap=cmap)
    # plt.colorbar()
    if save:
        plt.imsave(f"{str(OUTPUT_DIR)}/{name}.{ext}", img)
    if not hold:
        fig.show()

# %% Auto detect accurate corner
# img = cv.imread(f"{ROOT / 'imgcalibration.png'}")
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # find Harris corners
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray,2,3,0.04)
# dst = cv.dilate(dst,None)
# ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
# dst = np.uint8(dst)
# # find centroids
# ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
# # define the criteria to stop and refine the corners
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# # Now draw them
# # res = np.hstack((centroids,corners))
# res = np.int0(corners)
# tmp = np.zeros_like(img)
# img[res[:,1],res[:,0]] = [0,0,255]


# for x, y in res:
#     img = cv.putText(img, f"({x},{y})", (x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX,
#                     fontScale=0.4, color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)
# imshow("tmp", img, save=True, ext="bmp")
# %% Read data from CSV file
data = np.genfromtxt(f"{POINTS_FILE}",
                     delimiter=",",
                     dtype=int,
                     comments='#',
                     skip_header=1,
                     usecols=tuple(range(4)))
squares = [data[i:i+4, :] for i in range(0, len(data), 4)]
# %% Test with OpenCV
objectPnts = []
imgPnts = []
for sq in squares:
    objP = []
    imgP = []
    for pnt in sq:
        objP.append((pnt[2], pnt[3], 0))
        imgP.append((pnt[0], pnt[1]))
    objectPnts.append(np.array(objP, dtype=np.float32))
    imgPnts.append(np.array(imgP, dtype=np.float32))

reprojErr, K, distCoef, rVec, tVec = cv.calibrateCamera(
    objectPnts, imgPnts, (1080, 1920), None, None)
R, jacob = cv.Rodrigues(rVec[0])
t = tVec[0]
print(f"OpenCV's K=\n{K}")
print(f"Opencv's R|t=\n{np.hstack((R,t))}")
# %%
Hs = [cv.findHomography(sq[:, 2:], sq[:, :2], 0)[0]
      for sq in squares]  # Use least-square
# %% Test with handout data
# Hs[0] = np.array([[379.199677, 111.830818, 152.000000],
#                   [-64.140297, 350.826263, 149.000000],
#                   [0.102074, 0.210233, 1.000000]])
# Hs[1] = np.array([[168.271439, 125.763161, 596.000000],
#                   [81.960945, 320.478027, 84.000000],
#                   [-0.148918, 0.211012, 1.000000]])
# Hs[2] = np.array([[228.969971, -209.572891, 490.000000],
#                   [41.616714, 105.178200, 387.000000],
#                   [-0.078244, -0.182428, 1.000000]])
# %%


def get_omega_coefs(Hs: np.ndarray):
    assert len(Hs) >= 3
    coef_mat = []
    for h in Hs:
        h = h.T
        # coef = [[h[1, 1]*h[1, 2], h[1, 1]*h[2, 2]+h[2, 1]*h[1, 2], h[1, 1]*h[3, 2]+h[3, 1]*h[1, 2], h[2, 1]*h[2, 2], h[2, 1]*h[3, 2]+h[3, 1]*h[2, 2], h[3, 1]*h[3, 2]],
        #         [h[1, 1]**2-h[1, 2]**2, 2*(h[1, 1]*h[2, 1]-h[1, 2]*h[2, 2]), 2*(h[1, 1]*h[3, 1]-h[1, 2]*h[3, 2]), h[2, 1]**2-h[2, 2]**2, 2*(h[2, 1]*h[3, 1] - h[2, 2]*h[3, 2]), h[3, 1]**2-h[3, 2]**2]]
        coef = [[h[0, 0]*h[1, 0], h[0, 0]*h[1, 1]+h[0, 1]*h[1, 0], h[0, 0]*h[1, 2]+h[0, 2]*h[1, 0], h[0, 1]*h[1, 1], h[0, 1]*h[1, 2]+h[0, 2]*h[1, 1], h[0, 2]*h[1, 2]],
                [h[0, 0]**2-h[1, 0]**2, 2*(h[0, 0]*h[0, 1]-h[1, 0]*h[1, 1]), 2*(h[0, 0]*h[0, 2]-h[1, 0]*h[1, 2]), h[0, 1]**2-h[1, 1]**2, 2*(h[0, 1]*h[0, 2]-h[1, 1]*h[1, 2]), h[0, 2]**2-h[1, 2]**2]]
        coef_mat.append(coef)
    return np.vstack(coef_mat)


def to_matrix(elements: np.ndarray, type: str = "triangle"):
    '''
        To upper triangular/symmetric matrix
    '''
    if not isinstance(elements, np.ndarray):
        elements = np.array(elements)
    elements = elements.flatten()
    sqrt_b24ac = math.sqrt(1.0+8.0*len(elements))
    assert (sqrt_b24ac -
            1) % 2 == 0, f"{len(elements)} elements can't form a symmetric matrix"
    N = int((-1 + sqrt_b24ac) / 2)
    sym = np.zeros((N, N))
    sym[np.triu_indices(3)] = elements
    if type == "symmetric":
        sym += sym.T - np.diag(sym.diagonal())
    return sym


def solve_omega(H: np.ndarray, method: str = "svd"):
    if method != "svd":
        pass
    else:
        # Default is SVD
        U, S, VH = np.linalg.svd(H.astype(np.float64))
        V = VH.conj().T
        w = V[:, -1]  # last column & normalize by last element
    return to_matrix(w, "symmetric")


def solve_K(w: np.ndarray, skew: bool = False):
    w_ = np.linalg.inv(w)
    w_ /= w_[-1][-1]
    c = w_[0, 2]
    e = w_[1, 2]
    d = math.sqrt(w_[1, 1] - e**2)
    b = (w_[0, 1] - c*e) / d if skew else 0
    a = math.sqrt(w_[0, 0] - b**2 - c**2)
    print(abs(w_[0, 1]-c*e))
    return to_matrix([a, b, c, d, e, 1], "triangle")


coef = get_omega_coefs(Hs)
w = solve_omega(coef)
K = solve_K(w, skew=False)

print(f"K=\n{K}")
# %%


def get_Rt(H: np.ndarray, K: np.ndarray):
    K_inv = np.linalg.inv(K)
    r_ = K_inv @ H  # K^{-1}*H
    h1, h2, h3 = [H[:, i] for i in range(len(H[0]))]
    r1, r2, t_ = [r_[:, i]
                  for i in range(len(r_[0]))]  # K^{-1}*H, Get column vectors
    lambda_scale = np.mean(
        (1/np.linalg.norm(K_inv @ h1, ord=2), 1/np.linalg.norm(K_inv @ h2, ord=2)))
    r1 *= lambda_scale
    r2 *= lambda_scale
    r3 = np.cross(r1, r2)
    t = lambda_scale * t_
    return np.column_stack((r1, r2, r3)), t

# First Homography is in world coordinate
R, t = get_Rt(Hs[0], K)

print(f"R|t=\n{np.hstack((R, t[:, np.newaxis]))}")
