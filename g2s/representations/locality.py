import numpy as np


def get_local_environment(index, representation, nuclear_charges,
                          distances=None, sorting='norm_row', n_neighs=3):
    row, col = np.triu_indices(n_neighs+1, k=1)
    neighbour_sorting_idx = np.argsort(representation[index])[1:n_neighs+1]
    local_idxs = np.array([index, *neighbour_sorting_idx])
    atomic_representation = representation[local_idxs][:, local_idxs]
    local_nuclear_charges = nuclear_charges[local_idxs]
    if distances is not None:
        local_distances = distances[local_idxs][:, local_idxs]

    if sorting == 'norm_row':
        row_norm_idx = np.argsort(np.linalg.norm(atomic_representation, axis=1))
        atomic_representation = atomic_representation[row_norm_idx][:, row_norm_idx]
        local_nuclear_charges = local_nuclear_charges[row_norm_idx]
        local_idxs = local_idxs[row_norm_idx]

        if distances is not None:
            local_distances = local_distances[row_norm_idx][:, row_norm_idx]
    if distances is not None:
        return atomic_representation[row, col], local_nuclear_charges, local_idxs, local_distances[row, col]
    else:
        return atomic_representation[row, col], local_nuclear_charges, local_idxs


def get_unique_environments(local_environments, local_idxs, local_distances=None):
    uq_env = {}
    for i, r in enumerate(local_environments):
        if tuple(r) not in uq_env.keys():
            uq_env[tuple(r)] = {'local_idxs': [],
                                'local_dist': []}
        uq_env[tuple(r)]['local_idxs'].append(local_idxs[i])
        if local_distances is not None:
            uq_env[tuple(r)]['local_dist'].append(local_distances[i])

    if local_distances is not None:

        for k in uq_env.keys():
            uq_env[k]['avg_dist'] = np.array(uq_env[k]).mean(axis=0)
    return uq_env


def construct_sparse_dm(local_distances, local_idxs, n_atoms, n_neighs=4):
    row, col = np.triu_indices(n_neighs, k=1)
    _r, _c = np.triu_indices(n_atoms, k=1)

    sparse_dm = np.zeros((n_atoms, n_atoms))
    d = {(r, c): [] for r, c in zip(_r, _c)}
    for ld, li in zip(local_distances, local_idxs):
        for i in range(len(row)):
            atom_i = li[row[i]]
            atom_j = li[col[i]]
            dist = ld[i]
            pair = (atom_i, atom_j) if atom_i <= atom_j else (atom_j, atom_i)
            d[pair].append(dist)
    for p in d.keys():
        if d[p]:
            atom_i, atom_j = p
            sparse_dm[atom_i, atom_j] = np.mean(d[p])
            sparse_dm[atom_j, atom_i] = np.mean(d[p])

    return sparse_dm