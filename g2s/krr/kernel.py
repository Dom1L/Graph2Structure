import numpy as np


def gaussian_kernel(feat1, feat2, sigma):
    l2_norm = np.square(feat1[None, :] - feat2[:, None])
    return np.exp(-(l2_norm.sum(axis=2) / (2*(sigma**2))))


def laplacian_kernel(feat1, feat2, sigma):
    l1_norm = np.abs(feat1[None, :] - feat2[:, None])
    return np.exp(-(l1_norm.sum(axis=2) / sigma))
