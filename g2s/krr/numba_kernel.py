import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True)
def laplacian_kernel(feat1, feat2, sigma):
    a_size = feat1.shape[0]
    b_size = feat2.shape[0]
    l1_norm = np.ones((a_size, b_size))
    for i in prange(a_size):
        for j in range(b_size):
            l1_norm[i, j] = np.sum(np.abs(feat1[i] - feat2[j]))
    return np.exp(-1 * (l1_norm / sigma))


@njit(fastmath=True, parallel=True)
def gaussian_kernel(feat1, feat2, sigma):
    a_size = feat1.shape[0]
    b_size = feat2.shape[0]
    l2_norm = np.ones((a_size, b_size))
    for i in prange(a_size):
        for j in range(b_size):
            l2_norm[i, j] = np.sum(np.square(feat1[i] - feat2[j]))
    return np.exp(-(l2_norm / (2*(sigma**2))))


if __name__ == '__main__':
    x_train = np.random.random((10000, 100))
    gk = gaussian_kernel(x_train, x_train, 32)
    lk = laplacian_kernel(x_train, x_train, 32)
