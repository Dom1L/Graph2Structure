import numpy as np
from scipy.linalg import cho_solve, cho_factor
from tqdm import trange


def train_multikernel(kernel, labels, verbose=True):
    """
    Multi kernel approach to determine alpha coefficients.

    Only a single kernel inversion is performed here!

    Parameters
    ----------
    kernel: np.array, shape(n_poins, n_points)
        Kernel matrix to train on.
    labels: np.array, shape(n_poins, n_points)
        Training labels.
    verbose: bool
        Verbosity for the tqdm progress bar.

    Returns
    -------
    alphas: np.array, shape(n_distances, n_points)
        Kernel Ridge Regression alpha coefficients.
    """
    alphas = []
    inv_kernel = np.linalg.inv(kernel)
    for i in trange(labels.shape[1], disable=not verbose):
        alphas.append(np.dot(inv_kernel, labels[:, i]))
    return np.array(alphas)


def train_cholesky(kernel, labels, verbose=True):
    """
    Cholesky decomposition approach to determine alpha coefficients.

    Parameters
    ----------
    kernel: np.array, shape(n_poins, n_points)
        Kernel matrix to train on.
    labels: np.array, shape(n_poins, n_points)
        Training labels.
    verbose: bool
        Verbosity for the tqdm progress bar.

    Returns
    -------
    alphas: np.array, shape(n_distances, n_points)
        Kernel Ridge Regression alpha coefficients.
    """
    alphas = []
    cho_kernel = cho_factor(kernel)
    for i in trange(labels.shape[1], disable=not verbose):
        alphas.append(cho_solve(cho_kernel, labels[:, i]))
    return np.array(alphas)


def predict_distances(kernel, alphas):
    """
    Predicts all distances given a kernel and alpha coefficients.

    Parameters
    ----------
    kernel: np.array, shape(n_poins, n_points)
        Kernel matrix to use for prediction.
    alphas: np.array, shape(n_distances, n_points)
        Kernel Ridge Regression alpha coefficients.

    Returns
    -------
    distances: np.array, shape(n_molecules, n_distances)
        Predicted vectorized distance matrix.

    """
    distances = []
    for i in range(alphas.shape[0]):
        distances.append(np.dot(kernel, alphas[i]))
    return np.array(distances).swapaxes(1, 0)
