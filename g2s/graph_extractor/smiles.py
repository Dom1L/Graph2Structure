try:
    from rdkit import Chem
    from rdkit.Chem.rdmolops import GetAdjacencyMatrix
except ModuleNotFoundError:
    raise ModuleNotFoundError('Install RDKit if you want to use this feature!')

from ..constants import periodic_table


def smiles_graph(smiles, add_hydrogen=False, get_chirality=False):
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
    nuclear_charges = [periodic_table.index(atom.GetSymbol()) for atom in mol.GetAtoms()]
    bond_order_matrix = GetAdjacencyMatrix(mol, useBO=True)
    return bond_order_matrix, nuclear_charges


def rdkit_chirality(mol):
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)
    return Chem.FindMolChiralCenters(mol)
