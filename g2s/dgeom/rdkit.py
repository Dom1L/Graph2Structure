from rdkit import Chem
from rdkit.Geometry import Point3D

from ..constants import periodic_table


def graph_to_rdkit(elements, adjacency_matrix):
    """
    Converts a bond order matrix to an RDkit molecule.
    Blatantly adapted from https://stackoverflow.com/questions/51195392/smiles-from-graph

    Parameters
    ----------
    elements: np.array, shape(n_atoms)
        Elements of the molecule (not nuclear charges!).
    adjacency_matrix: np.array, shape(n_atoms, n_atoms)
        Bond order matrix.

    Returns
    -------
    mol: rdkit.mol

    """

    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(elements)):
        a = Chem.Atom(elements[i])
        mol_idx = mol.AddAtom(a)
        node_to_idx[i] = mol_idx

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):

            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    #Chem.SanitizeMol(mol)
    return mol


def embed_hydrogens(adjacency_matrix, nuclear_charges, heavy_atom_coords):
    """
    Computes hydrogen positions using RDkits ETKDG algorithm.
    During the embedding, heavy atom coordinates are fixed.

    The passed adjacency matrix and nuclear charges must already contain
    hydrogens!!

    Parameters
    ----------
    adjacency_matrix: np.array, shape(n_atoms, n_atoms)
        Bond order matrix.
    nuclear_charges: np.array, shape(n_atoms)
        Nuclear charges of the system.
    heavy_atom_coords: np.array, shape(n_atoms, 3)
        Cartesian coordinates.

    Returns
    -------
    embedded_coords: np.array, shape(n_atoms, 3)
        Heavy atom + embedded hydrogen coordinates.
    embedded_nuclear_charges: np.array, shape(n_atoms)
        Full list of nuclear charges including hydrogens.

    """
    elements = [periodic_table[nc] for nc in nuclear_charges]
    mol = graph_to_rdkit(elements, adjacency_matrix)

    # Generate some 2D coords, otherwise GetConformer is empty
    Chem.rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()

    # Set Coordinates
    for i in range(heavy_atom_coords.shape[0]):
        x, y, z = heavy_atom_coords[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))

    # Coord map fixes indices/coords during embedding
    coord_map = {i: mol.GetConformer().GetAtomPosition(i) for i in range(len(heavy_atom_coords))}
    mol_h = mol  #Chem.AddHs(mol)

    Chem.AllChem.EmbedMolecule(mol_h, coordMap=coord_map, useRandomCoords=True)

    embedded_coords = mol_h.GetConformer().GetPositions()
    embedded_nuclear_charges = [periodic_table.index(atom.GetSymbol()) for atom in mol_h.GetAtoms()]

    return embedded_coords, embedded_nuclear_charges



