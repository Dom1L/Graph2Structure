import numpy as np
from scipy.spatial.distance import squareform
from g2s.utils import calculate_distances, read_xyz
from scipy.optimize import minimize

from numba import jit


def lsbuild(boundary_matrix, _lambda=1.0, _tau=1e-5):
    n_atoms = boundary_matrix.shape[0]
    coords = np.zeros((n_atoms, 3))
    distance_matrix = init_dmat(boundary_matrix=boundary_matrix, seed=42)
    base_ids = np.arange(4)
    base_coords = embed_base(distance_matrix[base_ids][:, base_ids])
    n_base_atoms = base_coords.shape[0]
    base_row, base_col = np.triu_indices(n_base_atoms, k=1)
    res = minimize(opt_func, base_coords, method='BFGS',
                   args=(distance_matrix[(base_col, base_row)],
                         distance_matrix[(base_row, base_col)],
                         _lambda, _tau,
                         n_base_atoms, base_row, base_col, _lambda * (distance_matrix[(base_col, base_row)] - distance_matrix[(base_row, base_col)])))
    coords[base_ids, :] = res.x.reshape(n_base_atoms, 3)

    remaining = np.arange(n_base_atoms, n_atoms)
    _base_ids = [*base_ids]
    for i in remaining:
        new_coords = get_undetermined(coords[base_ids, :], distance_matrix, i, base_ids)
        _base_ids.append(i)

        n_base_atoms = len(_base_ids)
        base_ids = np.array(_base_ids)

        coords[i, :] = new_coords
        base_d = distance_matrix[base_ids][:, base_ids]
        base_row, base_col = np.triu_indices(n_base_atoms, k=1)
        res = minimize(opt_func, coords[base_ids, :], method='BFGS',
                       args=(base_d[(base_col, base_row)],
                             base_d[(base_row, base_col)],
                             _lambda, _tau,
                             n_base_atoms, base_row, base_col, _lambda * (base_d[(base_col, base_row)] - base_d[(base_row, base_col)])))
        coords[base_ids, :] = res.x.reshape(n_base_atoms, 3)
    return coords


def init_dmat(boundary_matrix, seed=42):
    n_atoms = boundary_matrix.shape[0]
    row, col = np.triu_indices(n_atoms, k=1)
    np.random.seed(seed)
    t = np.random.uniform(0, 1, size=(len(row)))
    return squareform(t * boundary_matrix[col, row] + (1 - t) * boundary_matrix[(row, col)])


def embed_base(distance_matrix):
    u, s, vh = np.linalg.svd(distance_matrix)
    eigen_val_idx = np.argsort(-s)
    e_tilde = np.zeros((3, 3))
    e_tilde[np.eye(3).astype(bool)] = s[eigen_val_idx[:3]]
    u_tilde = u[:, eigen_val_idx[:3]]
    return np.dot(u_tilde, np.sqrt(e_tilde))


@jit(nopython=True, fastmath=True)
def opt_func(x, dist_l, dist_u, _lambda, _tau, n_atoms, row, col, cofactor):
    x = x.reshape(n_atoms, 3)
    _coord_dist = np.expand_dims(x, 0) - np.expand_dims(x, 1)
    # coord_dist = np.sum(_coord_dist[row, col] ** 2, axis=1)

    coord_dist = []
    for r, c in zip(row, col):
        coord_dist.append(np.sum(_coord_dist[r, c] ** 2))
    coord_dist = np.array(coord_dist)
    b = np.sqrt((_lambda ** 2) * (dist_l - np.sqrt(coord_dist + (_tau ** 2))) ** 2 + _tau ** 2)
    c = np.sqrt((_lambda ** 2) * (dist_u - np.sqrt(coord_dist + (_tau ** 2))) ** 2 + _tau ** 2)
    return np.sum(cofactor+b+c)


# # @jit(nopython=True)
# def omega_eq(x, c, _lambda, _tau):
#     return


def get_undetermined(base_coords, distance_matrix, unknown_idx, base_ids):
    a = -2 * np.array([base_coords[base_ids[i + 1]] -
                       base_coords[base_ids[i]] for i in range(0, len(base_coords) - 1)])

    b = np.array([((distance_matrix[base_ids[i + 1], unknown_idx] ** 2 -
                    distance_matrix[i, unknown_idx] ** 2) -
                   (np.linalg.norm(base_coords[i + 1]) -
                    np.linalg.norm(base_coords[i])))
                  for i in range(0, len(base_coords) - 1)])

    return np.linalg.lstsq(a, b, rcond=None)[0]


if __name__ == '__main__':
    test_mol = '/Users/c0uch1/Downloads/octane.xyz'
    test_nc, test_coords = read_xyz(test_mol)
    test_distances = calculate_distances(test_coords)

    __coords = lsbuild(test_distances)
    print(np.mean(np.abs(calculate_distances(__coords) - test_distances)))
