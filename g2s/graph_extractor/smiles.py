try:
    from rdkit import Chem
    from rdkit.Chem.rdmolops import GetAdjacencyMatrix
except ModuleNotFoundError:
    raise ModuleNotFoundError('Install RDKit if you want to use this feature!')

from ..constants import periodic_table


def smiles_graph(smiles, add_hydrogen=False, get_chirality=False):
    """
    Converts SMILES to a bond order matrix and nuclear charges.

    Parameters
    ----------
    smiles: str
        SMILES string
    add_hydrogen: bool, (default=False)
        Adds hydrogens to molecule.
    get_chirality: bool, (default=False)
        Determines chiral centers in the molecule

    Returns
    -------
    bond_order_matrix: np.array, shape(n_atoms, n_atoms)
        Bond order matrix.
    nuclear_charges: np.array, shape(n_atoms, n_atoms)
        Nuclear charges.
    chiral_centers: list
        [(0, 'S')]
    """
    mol = Chem.MolFromSmiles(smiles)

    if add_hydrogen:
        mol = Chem.AddHs(mol)

    bond_order_matrix, nuclear_charges = get_bo(mol)

    if get_chirality is False:
        return bond_order_matrix, nuclear_charges
    else:
        chiral_centers = rdkit_chirality(mol)
        return bond_order_matrix, nuclear_charges, chiral_centers


def get_bo(mol):
    """
    Gets bond order matrix and nuclear charges from RDkit.

    Parameters
    ----------
    mol: rdkit.mol

    Returns
    -------
    bond_order_matrix: np.array, shape(n_atoms, n_atoms)
        Bond order matrix.
    nuclear_charges: np.array, shape(n_atoms, n_atoms)
        Nuclear charges.
    """
    nuclear_charges = [periodic_table.index(atom.GetSymbol()) for atom in mol.GetAtoms()]
    bond_order_matrix = GetAdjacencyMatrix(mol, useBO=True)
    return bond_order_matrix, nuclear_charges


def rdkit_chirality(mol):
    """
    Determines chiral centers in molecule.

    Parameters
    ----------
    mol: rdkit.mol

    Returns
    -------
    chiral_centers: list
        [(0, 'S')]
    """
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)
    return Chem.FindMolChiralCenters(mol)
