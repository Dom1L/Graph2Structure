from rdkit import Chem

try:
    from xyz2mol import AC2BO, xyz2AC, read_xyz_file, BO2mol, chiral_stereo_check, AC2mol
except ImportError:
    raise ImportError('Install xyz2mol software in order to use this parser. '
                      'Visit: https://github.com/jensengroup/xyz2mol')


def xyz2mol_graph(filepath, get_chirality=False):
    nuclear_charges, charge, coords = read_xyz_file(filepath)
    _adj_matrix, mol = xyz2AC(nuclear_charges, coords, charge)
    bond_order_matrix, atomic_valence_electrons = AC2BO(_adj_matrix, nuclear_charges, charge,
                                                 allow_charged_fragments=True,
                                                 use_graph=True)
    if get_chirality is False:
        return bond_order_matrix, nuclear_charges, coords
    else:
        chiral_mol = AC2mol(mol, _adj_matrix, nuclear_charges, charge,
                            allow_charged_fragments=True,
                            use_graph=True)
        chiral_stereo_check(chiral_mol)
        chiral_centers = Chem.FindMolChiralCenters(chiral_mol)
        return bond_order_matrix, nuclear_charges, coords, chiral_centers
