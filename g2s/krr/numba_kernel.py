import numpy as np
import numba


@numba.njit(fastmath=True)
def laplacian_kernel(feat1, feat2, sigma):
    a_size = feat1.shape[0]
    b_size = feat2.shape[0]
    l1_norm = np.ones((a_size, b_size))
    for i in range(a_size):
        for j in range(b_size):
            l1_norm[i, j] = np.sum(np.abs(feat1[i] - feat2[j]))
    return np.exp(-1 * (l1_norm / sigma))


@numba.njit(fastmath=True)
def gaussian_kernel(feat1, feat2, sigma):
    a_size = feat1.shape[0]
    b_size = feat2.shape[0]
    l2_norm = np.ones((a_size, b_size))
    for i in range(a_size):
        for j in range(b_size):
            l2_norm[i, j] = np.sum(np.square(feat1[i] - feat2[j]))
    return np.exp(-(l2_norm / (2*(sigma**2))))
