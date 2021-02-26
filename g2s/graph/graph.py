import numpy as np

from ..representations.representations import generate_bond_hop, generate_bond_length, generate_graph_coulomb_matrix


class GraphCompound(object):
    """ The ``Graph Compound`` class is used to store data from"""

    def __init__(self, adjacency_matrix, nuclear_charge, distances=None):
        empty_array = np.asarray([], dtype=float)

        self.molid = float("nan")
        self.name = None

        # Information about the compound
        self.natoms = len(nuclear_charge)
        self.adjacency_matrix = np.array(adjacency_matrix).astype(int)
        self.nuclear_charges = np.array(nuclear_charge).astype(int)
        self.distances = np.array(distances)

        # Representations:
        self.representation = empty_array

    def filter_atoms(self, atom_filter='heavy'):
        if atom_filter == 'heavy':
            nonh_idx = np.where(self.nuclear_charges != 1)[0]
            # Does not apply to very small molecules like H2O
            if len(nonh_idx) >= 4:
                self.adjacency_matrix = self.adjacency_matrix[nonh_idx][:, nonh_idx]
                self.nuclear_charges = self.nuclear_charges[nonh_idx]
                if self.distances is not None:
                    self.distances = self.distances[nonh_idx][:, nonh_idx]

    def generate_bond_order(self, size=9, sorting="row-norm"):
        self.representation = self.adjacency_matrix
        if sorting == 'norm_row':
            self.sort_norm()
        self.zero_padding(size)
        self.representation = self.representation[np.triu_indices(self.representation.shape[1], k=1)]

    def generate_bond_hop(self, size=9, sorting="row-norm"):
        self.representation = generate_bond_hop(self.adjacency_matrix, self.nuclear_charges)
        if sorting == 'norm_row':
            self.sort_norm()
        self.zero_padding(size)
        self.representation = self.representation[np.triu_indices(self.representation.shape[1], k=1)]

    def generate_bond_length(self, size=9, sorting="row-norm"):
        self.representation = generate_bond_length(self.adjacency_matrix, self.nuclear_charges)
        if sorting == 'norm_row':
            self.sort_norm()
        self.zero_padding(size)
        self.representation = self.representation[np.triu_indices(self.representation.shape[1], k=1)]

    def generate_graph_coulomb_matrix(self, size=9, sorting="row-norm"):
        self.representation = generate_graph_coulomb_matrix(self.adjacency_matrix, self.nuclear_charges)
        if sorting == 'norm_row':
            self.sort_norm()
        self.zero_padding(size)
        self.representation = self.representation[np.triu_indices(self.representation.shape[1], k=0)]

    def sort_norm(self):
        idx_list = np.argsort(np.linalg.norm(self.representation, axis=1))
        self.representation = self.representation[idx_list][:, idx_list]
        self.nuclear_charges = self.nuclear_charges[idx_list]
        if self.distances is not None:
            self.distances = self.distances[idx_list][:, idx_list]

    def zero_padding(self, size):
        padded_representation = np.zeros((size, size))
        n_atoms = self.representation.shape[0]
        padded_representation[:n_atoms, :n_atoms] = self.representation
        self.representation = padded_representation
        if self.distances is not None:
            padded_distances = np.zeros((size, size))
            padded_distances[:n_atoms, :n_atoms] = self.distances
            self.distances = padded_distances[np.triu_indices(self.representation.shape[1], k=1)]
