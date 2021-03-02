import os
import subprocess
import tempfile

import numpy as np

from ..utils import read_xyz
from ..constants import periodic_table


def run_xtb(cmd):
    with open('xtb.log', 'w') as logfile:
        subprocess.run(cmd.split(), stdout=logfile, stderr=logfile)


def xtb_opt(filepath, outpath=None, opt=True, ncores=1):
    """
    Determines xTB input command.
    Can do singlepoint or optimization

    Parameters
    ----------
    filepath: str
        Path to xyz file.
    outpath: str
        Output directory. If None, a temporary directory will be created.
    opt: bool, default=True
        Whether to run geometry optimization.
    ncores: int
        Amount of cores to use

    Returns
    -------
    outpath: str
        Output directory.
    """
    if outpath is None:
        outpath = tempfile.mkdtemp()

    if opt:
        cmd = f'xtb {filepath} --opt --parallel {ncores} --wbo'
    else:
        cmd = f'xtb {filepath} --parallel {ncores} --wbo'
    os.chdir(outpath)
    run_xtb(cmd)
    return outpath


def xtb_graph(filepath, outpath=None, ncores=1, opt=False):
    """
    Runs an xTB calculation to determine Wiberg Bond Orders, which
    can be used to derive a bond order matrix.

    Use with care!

    Parameters
    ----------
    filepath: str
        Path to xyz file.
    outpath: str
        Output directory. If None, a temporary directory will be created.
    opt: bool, default=True
        Whether to run geometry optimization.
    ncores: int
        Amount of cores to use

    Returns
    -------
    bond_order_matrix: np.array, shape(n_atoms, n_atoms)
        Bond order matrix.
    nuclear_charges: np.array, shape(n_atoms)
        Nuclear charges.
    coords: np.array, shape(n_atoms, 3)
        Cartesian coordinates of the system.
    """
    xtb_out = xtb_opt(filepath, outpath=outpath, opt=opt, ncores=ncores)

    elements, coords = read_xyz(filepath)
    nuclear_charges = np.array([periodic_table.index(e) for e in elements])
    # mol = Chem.MolFromMolFile(f'{xtb_out}/xtbtopo.mol', sanitize=False, removeHs=False, strictParsing=False)
    # coords = mol.GetConformer().GetPositions()
    bond_order_matrix = get_bo(f'{xtb_out}/wbo', natoms=len(elements))

    return bond_order_matrix, nuclear_charges, coords


def get_bo(wbo_file, natoms):
    """
    Parser for xTB wbo files.

    Parameters
    ----------
    wbo_file: str
        xTB wbo file.
    natoms: int
        Number of atoms in the system to determine the
        size of the bond order matrix.

    Returns
    -------
    bo_mat: np.array, shape(n_atoms, n_atoms)
        Bond order matrix.

    """
    with open(wbo_file, 'r') as infile:
        lines = infile.readlines()

    bo_mat = np.zeros((natoms, natoms))
    for line in lines:
        i, j, wbo = line.split()
        i = int(i) - 1
        j = int(j) - 1

        bo_mat[i, j] = round(float(wbo))
        bo_mat[j, i] = round(float(wbo))
    return bo_mat.astype(int)
