import os
import subprocess
import tempfile

try:
    from rdkit import Chem
    from .smiles import get_bo, rdkit_chirality
except ModuleNotFoundError:
    raise ModuleNotFoundError('Install RDKit if you want to use this feature!')


def run_xtb(cmd):
    with open('xtb.log', 'w') as logfile:
        subprocess.run(cmd.split(), stdout=logfile, stderr=logfile)


def xtb_opt(filepath, outpath=None, opt=True, ncores=1):

    if outpath is None:
        outpath = tempfile.mkdtemp()

    if opt:
        cmd = f'xtb {filepath} --opt --parallel {ncores} --wbo'
    else:
        cmd = f'xtb {filepath} --parallel {ncores} --wbo'
    os.chdir(outpath)
    run_xtb(cmd)
    return outpath


def xtb_graph(filepath, outpath=None, ncores=1, opt=False, get_chirality=False):
    xtb_out = xtb_opt(filepath, outpath=outpath, opt=opt, ncores=ncores)

    mol = Chem.SDMolSupplier(f'{xtb_out}/xtbtopo.mol')[0]
    coords = mol.GetConformer().GetPositions()
    bond_order_matrix, nuclear_charges = get_bo(mol)

    if get_chirality is False:
        return bond_order_matrix, nuclear_charges, coords
    else:
        chiral_centers = rdkit_chirality(mol)
        return bond_order_matrix, nuclear_charges, coords, chiral_centers
