import igraph
import numpy as np

from .representations import generate_bond_length


def local_bondlength(adjacency_matrix, nuclear_charges, distances=None):
    for a in adjacency_matrix:
        if np.sum(a) == 0:
            raise NotImplementedError('Non Bonded Hydrogen Found!')
    n_heavy_atoms = len(np.where(nuclear_charges != 1)[0])
    graph = igraph.Graph().Adjacency(list(adjacency_matrix.astype(float)))
    bond_h_idx = get_hydrogen_neighbours(graph, nuclear_charges)
    bond_length_matrix = generate_bond_length(adjacency_matrix, nuclear_charges)

    local_h_repr = []
    heavy_hydrogen_mapping = []
    hydrogen_heavy_distances = []
    for heavy_atom_index, hydrogen_indices in bond_h_idx:
        # Get local atomic representation of the atom bound to hydrogen
        atomic_representation = bond_length_matrix[hydrogen_indices[0], :n_heavy_atoms]

        # Sort by bond length distances to get the closest neighbours
        # Results in a representation vector of size 4
        neighbour_sorting_idx = np.argsort(atomic_representation)[:4]
        local_h_repr.append(atomic_representation[neighbour_sorting_idx])

        if distances is not None:
            h_distances, hydrogen_mapping = map_closest_distance(heavy_atom_index, hydrogen_indices,
                                                                 distances, neighbour_sorting_idx)
            heavy_hydrogen_mapping.append(hydrogen_mapping)
            hydrogen_heavy_distances.append(h_distances)
        else:
            heavy_hydrogen_mapping.append((heavy_atom_index, hydrogen_indices, neighbour_sorting_idx[1:]))

    if distances is not None:
        return np.array(local_h_repr), np.array(heavy_hydrogen_mapping), np.array(hydrogen_heavy_distances)
    else:
        return np.array(local_h_repr), np.array(heavy_hydrogen_mapping)


def get_hydrogen_neighbours(graph, nuclear_charges):
    heavy_atom_idx = np.where(nuclear_charges != 1)[0]

    bond_h_idx = []
    for idx in heavy_atom_idx:
        neighs = graph.neighbors(idx, mode='IN')
        h_ids = []
        for n in neighs:
            if nuclear_charges[n] == 1:
                h_ids.append(n)
        if h_ids:
            bond_h_idx.append((idx, np.array(h_ids)))
    return bond_h_idx


def map_closest_distance(heavy_atom_index, hydrogen_indices, distances, neighbour_sorting_idx):
    shortest_h_distances = distances[hydrogen_indices][:, neighbour_sorting_idx].sum(axis=1)
    closest_h_idx = hydrogen_indices[np.argmin(shortest_h_distances)]
    h_distances = [distances[heavy_atom_index, closest_h_idx]]
    for nbh in neighbour_sorting_idx[1:]:
        h_distances.append(distances[nbh, closest_h_idx])
    if len(hydrogen_indices) > 1:
        h_distances.append(distances[hydrogen_indices[0], hydrogen_indices[1]])
    else:
        h_distances.append(0.)

    hydrogen_mapping = (heavy_atom_index, np.array(hydrogen_indices)[np.argsort(shortest_h_distances)],
                        neighbour_sorting_idx[1:])
    return np.array(h_distances), hydrogen_mapping
