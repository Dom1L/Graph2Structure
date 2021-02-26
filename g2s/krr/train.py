import numpy as np
from scipy.linalg import cho_solve, cho_factor
from tqdm import trange


def train_multikernel(kernel, labels, verbose=True):
    alphas = []
    inv_kernel = np.linalg.inv(kernel)
    for i in trange(labels.shape[1], disable=not verbose):
        alphas.append(np.dot(inv_kernel, labels[:, i]))
    return np.array(alphas)


def train_cholesky(kernel, labels, verbose=True):
    alphas = []
    for i in trange(labels.shape[1], disable=not verbose):
        alphas.append(cho_solve(cho_factor(kernel), labels[:, i]))
    return np.array(alphas)


def predict_distances(kernel, alphas):
    distances = []
    for i in range(alphas.shape[0]):
        distances.append(np.dot(kernel, alphas[i]))
    return np.array(distances).swapaxes(1, 0)
