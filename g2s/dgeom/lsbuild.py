import numpy as np
from scipy.spatial.distance import squareform
from g2s.utils import calculate_distances, write_xyz, read_xyz
from scipy.optimize import minimize

adjacency = np.array([[0, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0]])

methane_nuclear_charges = np.array([6, 1, 1, 1, 1])

methane_coords = np.array([[-0.0126981359, 1.0858041578, 0.0080009958],
                           [0.002150416, -0.0060313176, 0.0019761204],
                           [1.0117308433, 1.4637511618, 0.0002765748],
                           [-0.540815069, 1.4475266138, -0.8766437152],
                           [-0.5238136345, 1.4379326443, 0.9063972942]])

distances = calculate_distances(methane_coords)


def lsbuild(adjacency, distances):
    n_atoms = distances.shape[0]
    triu = np.triu_indices(n_atoms, k=1)
    tril = np.tril_indices(n_atoms, k=-1)

    if check_symmetric(distances):
        D = distances
    else:
        t = np.random.uniform(0, 1, size=(len(triu[0])))
        D = squareform(t*distances[tril] + (1-t)*distances[triu])
    base = np.arange(4)
    u, s, vh = np.linalg.svd(D)
    eigen_val_idx = np.argsort(-s)
    e_tilde = np.zeros((3, 3))
    e_tilde[np.eye(3).astype(bool)] = s[eigen_val_idx[:3]]
    u_tilde = u[:, eigen_val_idx[:3]]
    x = np.dot(u_tilde, np.sqrt(e_tilde))
    _lambda = 1.0
    _tau = 1e-5

    res = minimize(opt_func, x, method='L-BFGS-B', args=(distances[tril], distances[triu], _lambda, _tau))
    print(res.fun)
    recoords = res.x.reshape(5, 3)
    pass

def opt_func(x, dist_l, dist_u, _lambda, _tau):
    triu = np.triu_indices(8, k=1)
    x = x.reshape(8, 3)
    coord_dist = x[None, :] - x[:, None]
    coord_dist = np.sum(coord_dist[triu]**2, axis=1)
    return np.sum(_lambda*(dist_l-dist_u) + omega_eq(coord_dist, dist_l, _lambda, _tau) + omega_eq(coord_dist, dist_u, _lambda, _tau))


def omega_eq(x, c, _lambda, _tau):
    return np.sqrt((_lambda**2)*(c - np.sqrt(x+(_tau**2)))**2 + _tau**2)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)



def get_undetermined(base_coords, distance_matrix, base_idxs, unknown_idx):
    a = np.array([base_coords[i] - base_coords[0] for i in range(1, len(base_idxs))])
    # b = [(distance_matrix[i, unknown_idx]**2 - distance_matrix[i, 0]**2 + distance_matrix[0, unknown_idx]**2)/2 for i in range(1, len(base_coords))]
    b = np.array([((distance_matrix[i, unknown_idx]**2 - distance_matrix[0, unknown_idx]**2) - (np.linalg.norm(base_coords[i]) - np.linalg.norm(base_coords[0]))) for i in range(1, len(base_coords))])
    # b = np.array([((distance_matrix[i, unknown_idx]**2 - distance_matrix[0, unknown_idx]**2) - distance_matrix[i, 0]**2)/-2 for i in range(1, len(base_idxs))])

    new_coords = np.linalg.lstsq(a, b)[0]
    idxs = np.array([*base_idxs, unknown_idx])
    c = np.array([*base_coords, new_coords])
    d = distance_matrix[idxs][:, idxs]
    n, m = triu
    res = minimize(opt_func, coords.flatten(), method='L-BFGS-B', args=(d[(m,n)], d[triu], _lambda, _tau))

    print(np.abs(d[tril]-calculate_distances(res.x.reshape(8,3))[tril]).mean())
