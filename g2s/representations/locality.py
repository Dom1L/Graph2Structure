import numpy as np
import igraph


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


def get_full_local_knn_environment(idx, adjacency_matrix, representation, nuclear_charges, distances=None, depth=2, sorting='norm_row'):
    graph = igraph.Graph()
    graph = graph.Adjacency(list(adjacency_matrix.astype(float)))
    # Get all K=1 neighbors around a query atom
    neighbor_idxs = graph.neighbors(idx, mode='OUT')
    new_neighs = neighbor_idxs
    d = 1
    # if k > 1 then scan further neighbors
    while d < depth:
        kn = []
        for n in new_neighs:
            kn.append(graph.neighbors(n, mode='OUT'))

        # Filter non-unique indices
        kn = [item for sublist in kn for item in sublist]
        kn = list(filter((idx).__ne__, kn))
        kn = np.unique(kn)

        neighbor_idxs.extend(kn)
        new_neighs = kn
        d += 1
    neighbor_idxs = np.unique(neighbor_idxs)

    local_idxs = np.array([idx, *neighbor_idxs])
    row, col = np.triu_indices(len(local_idxs), k=1)

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


def get_local_knn_environment(idx, adjacency_matrix, representation, nuclear_charges, distances=None, depth=2, sorting='norm_row'):
    graph = igraph.Graph()
    graph = graph.Adjacency(list(adjacency_matrix.astype(float)))
    # Get all K=1 neighbors around a query atom
    neighbor_idxs = graph.neighbors(idx, mode='OUT')
    new_neighs = neighbor_idxs
    d = 1
    # if k > 1 then scan further neighbors
    while d < depth:
        kn = []
        for n in new_neighs:
            kn.append(graph.neighbors(n, mode='OUT'))

        # Filter non-unique indices
        kn = [item for sublist in kn for item in sublist]
        kn = list(filter((idx).__ne__, kn))
        kn = np.unique(kn)

        neighbor_idxs.extend(kn)
        new_neighs = kn
        d += 1
    neighbor_idxs = np.unique(neighbor_idxs)

    # row, col = np.triu_indices(len(local_idxs), k=1)

    atomic_representation = representation[idx][neighbor_idxs]
    local_nuclear_charges = nuclear_charges[neighbor_idxs]
    if distances is not None:
        local_distances = distances[idx][neighbor_idxs]

    if sorting == 'norm_row':
        row_norm_idx = np.argsort(np.linalg.norm(atomic_representation, axis=1))
        atomic_representation = atomic_representation[row_norm_idx]
        local_nuclear_charges = local_nuclear_charges[row_norm_idx]
        neighbor_idxs = neighbor_idxs[row_norm_idx]

        if distances is not None:
            local_distances = local_distances[row_norm_idx]

    local_nuclear_charges = np.array([nuclear_charges[idx], *local_nuclear_charges])
    local_idxs = np.array([idx, *neighbor_idxs])

    if distances is not None:
        return atomic_representation, local_nuclear_charges, local_idxs, local_distances
    else:
        return atomic_representation, local_nuclear_charges, local_idxs


def get_unique_environments(local_environments, local_idxs, local_distances=None):
    uq_env = {}
    for j in range(len(local_environments)):
        for i, r in enumerate(local_environments[j]):
            if tuple(r) not in uq_env.keys():
                uq_env[tuple(r)] = {'local_idxs': [],
                                    'local_dist': []}
            uq_env[tuple(r)]['local_idxs'].append(local_idxs[j][i])
            if local_distances is not None:
                uq_env[tuple(r)]['local_dist'].append(local_distances[j][i])

    if local_distances is not None:

        for k in uq_env.keys():
            uq_env[k]['avg_dist'] = np.array(uq_env[k]['local_dist']).mean(axis=0)
    return uq_env


def construct_sparse_dm(local_distances, local_idxs, n_atoms):
    _r, _c = np.triu_indices(n_atoms, k=1)

    sparse_dm = np.zeros((n_atoms, n_atoms))
    d = {(r, c): [] for r, c in zip(_r, _c)}
    for ld, li in zip(local_distances, local_idxs):
        row, col = np.triu_indices(local_idxs, k=1)
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