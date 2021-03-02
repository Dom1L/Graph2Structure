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
    xtb_out = xtb_opt(filepath, outpath=outpath, opt=opt, ncores=ncores)

    elements, coords = read_xyz(filepath)
    nuclear_charges = np.array([periodic_table.index(e) for e in elements])
    # mol = Chem.MolFromMolFile(f'{xtb_out}/xtbtopo.mol', sanitize=False, removeHs=False, strictParsing=False)
    # coords = mol.GetConformer().GetPositions()
    bond_order_matrix = get_bo(f'{xtb_out}/wbo', natoms=len(elements))

    return bond_order_matrix, nuclear_charges, coords


def get_bo(wbo_file, natoms):
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
