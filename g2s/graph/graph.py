import numpy as np

from ..representations.representations import generate_bond_hop, generate_bond_length, generate_graph_coulomb_matrix
from ..representations.hydrogens import local_bondlength


class GraphCompound(object):
    """
    Interface to filter atoms, calculate representations, do zero padding and sorting.
    """

    def __init__(self, adjacency_matrix, nuclear_charge, distances=None):
        """

        Parameters
        ----------
        adjacency_matrix: np.array, shape(n_atoms, n_atoms)
            Bond order matrix of the system.
        nuclear_charge: np.array, shape(n_atoms)
            Nuclear charges.
        distances: np.array, shape(n_atoms, n_atoms)
            Interatomic distance matrix.
        """
        empty_array = np.asarray([], dtype=float)

        self.molid = float("nan")
        self.name = None

        # Information about the compound
        self.natoms = len(nuclear_charge)
        self.adjacency_matrix = np.array(adjacency_matrix).astype(int)
        self.nuclear_charges = np.array(nuclear_charge).astype(int)
        self.distances = np.array(distances)

        # Sort atoms such that heavy atoms are always at the end
        self.resort_atoms()

        # Representations:
        self.representation = empty_array
        self.sorting_idxs = None
        self.filtered = False

        # Stores the full adjacency matrix etc. in case of filtering
        self.full_adjacency_matrix = None
        self.full_nuclear_charges = None
        self.full_distances = None

        # Hydrogen representation, distances and mapping
        self.hydrogen_representations = np.zeros((1, 4))
        self.heavy_hydrogen_mapping = empty_array
        self.hydrogen_heavy_distances = np.zeros((1, 5))

    def resort_atoms(self):
        """
        Resorts the order of atoms such that all hydrogens are at the end.
        This is done to avoid conflicts during the mapping process.

        """
        sort_idx = np.argsort(-1 * self.nuclear_charges)
        self.adjacency_matrix = self.adjacency_matrix[sort_idx][:, sort_idx]
        self.nuclear_charges = self.nuclear_charges[sort_idx]
        self.distances = self.distances[sort_idx][:, sort_idx]

    def filter_atoms(self, atom_filter='heavy'):
        """
        Filter specific atoms in the molecules.

        Heavy atom filtering is not applied to molecules
        with less than 4 heavy atoms.

        In case filtering is applied, a copy of the full
        distance matrix, adjacency matrix etc. is saved in
        self.full_adjacency_matrix etc.

        Parameters
        ----------
        atom_filter: str, (default='heavy')
            Which atoms to filter.

        """
        if atom_filter == 'heavy':
            nonh_idx = np.where(self.nuclear_charges != 1)[0]
            # Does not apply to very small molecules like H2O
            if len(nonh_idx) >= 4:
                self.full_adjacency_matrix = self.adjacency_matrix.copy()
                self.full_nuclear_charges = self.nuclear_charges.copy()
                self.adjacency_matrix = self.adjacency_matrix[nonh_idx][:, nonh_idx]
                self.nuclear_charges = self.nuclear_charges[nonh_idx]
                self.filtered = True
                if self.distances is not None:
                    self.full_distances = self.distances.copy()
                    self.distances = self.distances[nonh_idx][:, nonh_idx]

    def generate_bond_order(self, size=9, sorting="row-norm"):
        """
        Calculate bond order representation.

        Parameters
        ----------
        size: int
            Padding size. Set to largest number of atoms in the dataset.
        sorting: str
            Sorting method to use. Currently only row-norm implemented.
        """
        self.representation = self.adjacency_matrix
        if sorting == 'norm_row':
            self.sort_norm()
        self.zero_padding(size)
        self.representation = self.representation[np.triu_indices(self.representation.shape[1], k=1)]

    def generate_bond_hop(self, size=9, sorting="row-norm"):
        """
        Calculate bond hop representation.

        Parameters
        ----------
        size: int
            Padding size. Set to largest number of atoms in the dataset.
        sorting: str
            Sorting method to use. Currently only row-norm implemented.
        """
        self.representation = generate_bond_hop(self.adjacency_matrix, self.nuclear_charges)
        if sorting == 'norm_row':
            self.sort_norm()
        self.zero_padding(size)
        self.representation = self.representation[np.triu_indices(self.representation.shape[1], k=1)]

    def generate_bond_length(self, size=9, sorting="row-norm"):
        """
        Calculate bond length representation.

        Parameters
        ----------
        size: int
            Padding size. Set to largest number of atoms in the dataset.
        sorting: str
            Sorting method to use. Currently only row-norm implemented.
        """
        self.representation = generate_bond_length(self.adjacency_matrix, self.nuclear_charges)
        if sorting == 'norm_row':
            self.sort_norm()
        self.zero_padding(size)
        self.representation = self.representation[np.triu_indices(self.representation.shape[1], k=1)]

    def generate_graph_coulomb_matrix(self, size=9, sorting="row-norm"):
        """
        Calculate graph-coulomb representation.

        Parameters
        ----------
        size: int
            Padding size. Set to largest number of atoms in the dataset.
        sorting: str
            Sorting method to use. Currently only row-norm implemented.
        """
        self.representation = generate_graph_coulomb_matrix(self.adjacency_matrix, self.nuclear_charges)
        if sorting == 'norm_row':
            self.sort_norm()
        self.zero_padding(size)
        self.representation = self.representation[np.triu_indices(self.representation.shape[1], k=0)]

    def generate_local_hydrogen_matrix(self):
        """
        Generates a local bond length representation for hydrogen atoms.
        Only 4 heavy atoms are included. Representation will have size 5.
        In case the molecule has less than 4 heavy atoms, no representation will be computed.

        On top of the representation, also computes mapping of each heavy atom to hydrogens.
        """
        n_heavy_atoms = len(np.where(self.nuclear_charges != 1)[0])
        # Does not apply to very small molecules like H2O
        if n_heavy_atoms < 4:
            return

        # Always use the full matrix!
        if self.filtered:
            adjacency, nuclear_charges, distances = self.full_adjacency_matrix, self.full_nuclear_charges, self.full_distances
        else:
            adjacency, nuclear_charges, distances = self.adjacency_matrix, self.nuclear_charges, self.distances

        if distances is None:
            local_h_repr, heavy_hydrogen_mapping = local_bondlength(adjacency, nuclear_charges, distances)
            hydrogen_heavy_distances = None
        else:
            local_h_repr, heavy_hydrogen_mapping, hydrogen_heavy_distances = local_bondlength(adjacency, nuclear_charges, distances)

        self.hydrogen_representations = local_h_repr
        self.heavy_hydrogen_mapping = heavy_hydrogen_mapping
        self.hydrogen_heavy_distances = hydrogen_heavy_distances

    def sort_norm(self):
        """
        Resort representation, nuclear charges, adjacency matrix and distance matrix using the
        row norm of the representation.
        Ensure permutational invariance!
        """
        idx_list = np.argsort(np.linalg.norm(self.representation, axis=1))
        self.representation = self.representation[idx_list][:, idx_list]
        self.nuclear_charges = self.nuclear_charges[idx_list]
        self.adjacency_matrix = self.adjacency_matrix[idx_list][:, idx_list]
        self.sorting_idxs = idx_list
        if self.filtered:
            self.full_adjacency_matrix = self.full_adjacency_matrix[idx_list][:, idx_list]
            self.full_nuclear_charges = self.full_nuclear_charges[idx_list]
        if self.distances is not None:
            self.distances = self.distances[idx_list][:, idx_list]
            if self.filtered:
                self.full_distances = self.full_distances[idx_list][:, idx_list]

    def zero_padding(self, size):
        """
        Adds zeros to the represenation and distance matrix to make it
        usable with machine learning algorithms.

        Parameters
        ----------
        size: int
            Size of the total matrix.

        """
        padded_representation = np.zeros((size, size))
        n_atoms = self.representation.shape[0]
        padded_representation[:n_atoms, :n_atoms] = self.representation
        self.representation = padded_representation
        if self.distances is not None:
            padded_distances = np.zeros((size, size))
            padded_distances[:n_atoms, :n_atoms] = self.distances
            self.distances = padded_distances[np.triu_indices(self.representation.shape[1], k=1)]
