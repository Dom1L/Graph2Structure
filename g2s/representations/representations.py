import igraph
import numpy as np

from ..constants import periodic_table, atom_radii


def generate_bond_hop(adjacency_matrix, nuclear_charges):
    n_atoms = len(nuclear_charges)
    graph = igraph.Graph().Adjacency(list(adjacency_matrix.astype(float)))
    bond_length_matrix = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        bond_length_matrix[i] = [len(j) - 1 for j in graph.get_shortest_paths(i)]
    return bond_length_matrix


def generate_bond_length(adjacency_matrix, nuclear_charges):
    n_atoms = len(nuclear_charges)
    graph = igraph.Graph().Adjacency(list(adjacency_matrix.astype(float)))
    bond_length_matrix = np.zeros((n_atoms, n_atoms))
    weights = get_weights(graph, adjacency_matrix, nuclear_charges)
    for i in range(n_atoms):
        bond_length_matrix[i] = graph.shortest_paths(i, weights=weights)[0]
    return bond_length_matrix


def generate_graph_coulomb_matrix(adjacency_matrix, nuclear_charges):
    bond_length_matrix = generate_bond_length(adjacency_matrix, nuclear_charges)
    with np.errstate(divide='ignore'):  # ignore the dividing by 0 warning
        inv_dist = 1 / bond_length_matrix
    # First, calculate the off diagonals
    zizj = nuclear_charges[None, :] * nuclear_charges[:, None]
    np.fill_diagonal(inv_dist, 0)  # to get rid of nasty NaNs
    graph_cm = zizj * inv_dist
    # Second, calculate self interaction
    np.fill_diagonal(graph_cm, 0.5 * nuclear_charges ** 2.4)
    return graph_cm


def get_weights(graph, adjacency, nuclear_charges):
    weights = []
    for edge in graph.get_edgelist():
        bond_order = adjacency[edge] - 1  # correcting for 0 indexing in python
        el1 = periodic_table[int(nuclear_charges[edge[0]])]
        el2 = periodic_table[int(nuclear_charges[edge[1]])]
        atom_radius1 = atom_radii[el1][bond_order]
        atom_radius2 = atom_radii[el2][bond_order]
        weights.append(atom_radius1 + atom_radius2)
    return weights
