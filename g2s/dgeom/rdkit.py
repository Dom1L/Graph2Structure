from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.AllChem import EmbedMolecule
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from ..constants import periodic_table
import numpy as np

def graph_to_rdkit(elements, adjacency_matrix, num_heavy_atoms):
    """
    Converts a bond order matrix to an RDkit molecule.
    Blatantly adapted from https://stackoverflow.com/questions/51195392/smiles-from-graph

    Parameters
    ----------
    elements: np.array, shape(n_atoms)
        Elements of the molecule (not nuclear charges!).
    adjacency_matrix: np.array, shape(n_atoms, n_atoms)
        Bond order matrix.
    num_heavy_atoms: integer
        Number of heavy (non-hydrogen) atoms.

    Returns
    -------
    mol: rdkit.mol

    """

    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(num_heavy_atoms):
        a = Chem.Atom(elements[i])
        mol_idx = mol.AddAtom(a)
        node_to_idx[i] = mol_idx

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):

            # only traverse half the matrix
            if (iy <= ix) or (iy>=num_heavy_atoms):
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
    Chem.SanitizeMol(mol)
    return mol


def embed_hydrogens(adjacency_matrix, nuclear_charges, heavy_atom_coords, seed=1, maxAttempts=128, useExpTorsionAnglePrefs=False, useBasicKnowledge=False):
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
    embedded_adjacency_matrix: np.array, shape(n_atoms, n_atoms)
        Adjacency matrix corresponding to heavy and embedded hydrogen atoms.
    """
    num_heavy_atoms=heavy_atom_coords.shape[0]
    elements = [periodic_table[nc] for nc in nuclear_charges]
    mol = graph_to_rdkit(elements, adjacency_matrix, num_heavy_atoms)

    # Generate some 2D coords, otherwise GetConformer is empty
    Compute2DCoords(mol)
    conf = mol.GetConformer()

    # Set Coordinates
    for i in range(num_heavy_atoms):
        x, y, z = heavy_atom_coords[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))

    # Coord map fixes indices/coords during embedding
    coord_map = {i: mol.GetConformer().GetAtomPosition(i) for i in range(num_heavy_atoms)}

    mol_h = Chem.AddHs(mol)

    EmbedMolecule(mol_h, coordMap=coord_map, useRandomCoords=True, ignoreSmoothingFailures=True, randomSeed=seed, maxAttempts=maxAttempts, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge)

    embedded_coords = mol_h.GetConformer().GetPositions()
    embedded_nuclear_charges = [atom.GetAtomicNum() for atom in mol_h.GetAtoms()]
    embedded_adjacency_matrix=GetAdjacencyMatrix(mol_h)
    return embedded_coords, embedded_nuclear_charges, embedded_adjacency_matrix



