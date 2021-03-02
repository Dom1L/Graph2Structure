import igraph
import numpy as np

from ..constants import periodic_table, atom_radii


def generate_bond_hop(adjacency_matrix, nuclear_charges):
    """
    Calculates bond hop matrix.

    A bond hop matrix counts the amount of bonds to "hop" on the shortest
    path between two atoms.

    Parameters
    ----------
    adjacency_matrix: np.array, shape(n_atoms, n_atoms)
        Bond order matrix of the system.
    nuclear_charges: np.array, shape(n_atoms)
        Nuclear charges.

    Returns
    -------
    bond_hop_matrix: np.array, shape(n_atoms, n_atoms)
        Bond hop matrix of the system.
    """
    n_atoms = len(nuclear_charges)
    graph = igraph.Graph().Adjacency(list(adjacency_matrix.astype(float)))
    bond_hop_matrix = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        bond_hop_matrix[i] = [len(j) - 1 for j in graph.get_shortest_paths(i)]
    return bond_hop_matrix


def generate_bond_length(adjacency_matrix, nuclear_charges):
    """
    Calculates bond length matrix.

    A bond length matrix sums the bond lengths on the shortest path between two atoms.

    Parameters
    ----------
    adjacency_matrix: np.array, shape(n_atoms, n_atoms)
        Bond order matrix of the system.
    nuclear_charges: np.array, shape(n_atoms)
        Nuclear charges.

    Returns
    -------
    bond_length_matrix: np.array, shape(n_atoms, n_atoms)
        Bond length matrix of the system.
    """
    n_atoms = len(nuclear_charges)
    graph = igraph.Graph().Adjacency(list(adjacency_matrix.astype(float)))
    bond_length_matrix = np.zeros((n_atoms, n_atoms))
    weights = get_weights(graph, adjacency_matrix, nuclear_charges)
    for i in range(n_atoms):
        bond_length_matrix[i] = graph.shortest_paths(i, weights=weights)[0]
    return bond_length_matrix


def generate_graph_coulomb_matrix(adjacency_matrix, nuclear_charges):
    """
    Calculates graph coulomb matrix.

    Has the same form as the regular coulomb matrix, but used bond lengths instead of
    atomic distances.

    Parameters
    ----------
    adjacency_matrix: np.array, shape(n_atoms, n_atoms)
        Bond order matrix of the system.
    nuclear_charges: np.array, shape(n_atoms)
        Nuclear charges.

    Returns
    -------
    graph_cm: np.array, shape(n_atoms, n_atoms)
        Graph coulomb matrix of the system.
    """
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
    """
    Computes bond lengths of the graph.
    These bond lengths can be applied to each
    edge in a graph and thereby allows the computation
    of shortest paths and path lengths between two atoms.

    Parameters
    ----------
    graph: igraph.Graph
    adjacency: np.array, shape(n_atoms, n_atoms)
        Bond order matrix of the system.
    nuclear_charges: np.array, shape(n_atoms)
        Nuclear charges.


    Returns
    -------
    weights: list, shape(n_edges)
        Contains bond lengths of each edge.

    """
    weights = []
    for edge in graph.get_edgelist():
        bond_order = adjacency[edge] - 1  # correcting for 0 indexing in python
        el1 = periodic_table[int(nuclear_charges[edge[0]])]
        el2 = periodic_table[int(nuclear_charges[edge[1]])]
        atom_radius1 = atom_radii[el1][bond_order]
        atom_radius2 = atom_radii[el2][bond_order]
        weights.append(atom_radius1 + atom_radius2)
    return weights
