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


def get_local_knn_environment(idx, adjacency_matrix, representation, nuclear_charges,
                              distances=None, depth=2, min_neighs=5,
                              sorting='norm_row'):
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

    neighbor_idxs = list(np.unique(neighbor_idxs))

    while len(neighbor_idxs) < min_neighs:
        kn = []
        for n in new_neighs:
            kn.append(graph.neighbors(n, mode='OUT'))

        # Filter non-unique indices
        kn = [item for sublist in kn for item in sublist]
        kn = list(filter((idx).__ne__, kn))
        kn = np.unique([*new_neighs, *kn])

        neighbor_idxs.extend(kn)
        new_neighs = kn
        neighbor_idxs = list(np.unique(neighbor_idxs))

    neighbor_idxs = np.array(neighbor_idxs)
    atomic_representation = representation[idx][neighbor_idxs]
    local_nuclear_charges = nuclear_charges[neighbor_idxs]
    if distances is not None:
        local_distances = distances[idx][neighbor_idxs]

    if sorting == 'norm_row':
        row_norm_idx = np.argsort(atomic_representation)
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
            uq_env[k]['lower_bound'] = np.array(uq_env[k]['local_dist']).min(axis=0)
            uq_env[k]['upper_bound'] = np.array(uq_env[k]['local_dist']).max(axis=0)
            uq_env[k]['stddev_dist'] = np.array(uq_env[k]['local_dist']).std(axis=0)
    return uq_env


def construct_sparse_matrix(upper_bounds, lower_bounds, local_idxs, n_atoms, tight_bounds=False):
    if lower_bounds is None:
        lower_bounds = upper_bounds

    _r, _c = np.triu_indices(n_atoms, k=1)

    sparse_dm = np.zeros((n_atoms, n_atoms))
    d_upper = {(r, c): [] for r, c in zip(_r, _c)}
    d_lower = {(r, c): [] for r, c in zip(_r, _c)}

    for i in range(len(upper_bounds)):
        upb, lowb, li = upper_bounds[i], lower_bounds[i], local_idxs[i]
        atom_i = li[0]

        for j, atom_j in enumerate(li[1:]):
            pair = (atom_i, atom_j) if atom_i <= atom_j else (atom_j, atom_i)
            d_upper[pair].append(upb[j])
            d_lower[pair].append(lowb[j])

    for atom_pair in d_upper.keys():
        if d_upper[atom_pair]:
            atom_i, atom_j = atom_pair
            upb, lowb = d_upper[atom_pair], d_lower[atom_pair]

            if tight_bounds:
                mean_bound = (np.mean(upb) + np.mean(lowb))/2
                sparse_dm[atom_i, atom_j] = mean_bound
                sparse_dm[atom_j, atom_i] = mean_bound
            else:
                sparse_dm[atom_i, atom_j] = np.max(upb)
                sparse_dm[atom_j, atom_i] = np.min(lowb)

    return sparse_dm


def get_sparse_bounds_matrix(uq_ha_env, mol_data, boundaries=True, tight_envs=True):
    heavy_atom_local_rep = mol_data['ha_lr']
    heavy_atom_idxs = mol_data['heavy_atom_idxs']
    ha_upper_bounds = []
    ha_lower_bounds = []
    for i in range(len(heavy_atom_local_rep)):
        if boundaries:
            upper_bound = uq_ha_env[tuple(heavy_atom_local_rep[i])]['upper_bound']
            lower_bound = uq_ha_env[tuple(heavy_atom_local_rep[i])]['lower_bound']
        else:
            upper_bound = lower_bound = uq_ha_env[tuple(heavy_atom_local_rep[i])]['avg_dist']
        ha_upper_bounds.append(upper_bound)
        ha_lower_bounds.append(lower_bound)

    ha_sparse_bounds = construct_sparse_matrix(ha_upper_bounds, ha_lower_bounds, mol_data['ha_lidxs'],
                                               n_atoms=len(heavy_atom_idxs),
                                               tight_bounds=tight_envs)
    return ha_sparse_bounds, mol_data['nuclear_charges'][heavy_atom_idxs]
