import os
import subprocess

import numpy as np
from tqdm import tqdm

from ..utils import vector_to_square


class DGSOL:
    def __init__(self, distances, nuclear_charges, vectorized_input=True):
        self.nuclear_charges = nuclear_charges
        self.distances = vector_to_square(distances) if vectorized_input else distances
        self.coords = None
        self.c_errors = None

    def gen_cerror_overview(self):
        print('Error Type, Min, Mean, Max')
        print(f'minError: {np.min(self.c_errors[:, 1])}, {np.mean(self.c_errors[:, 1])}, {np.max(self.c_errors[:, 1])}')
        print(f'avgError: {np.min(self.c_errors[:, 2])}, {np.mean(self.c_errors[:, 2])}, {np.max(self.c_errors[:, 2])}')
        print(f'maxError: {np.min(self.c_errors[:, 2])}, {np.mean(self.c_errors[:, 3])}, {np.max(self.c_errors[:, 3])}')

    def to_scientific_notation(self, number):
        a, b = '{:.17E}'.format(number).split('E')
        num = '{:.12f}E{:+03d}'.format(float(a) / 10, int(b) + 1)
        return num[1:]

    def write_dgsol_input(self, distances, outpath):
        n, m = np.triu_indices(distances.shape[1], k=1)
        with open(f'{outpath}/dgsol.input', 'w') as outfile:
            for i, j in zip(n, m):
                outfile.write(
                    f'{i + 1:9.0f}{j + 1:10.0f}   {self.to_scientific_notation(distances[i, j])}   '
                    f'{self.to_scientific_notation(distances[i, j])}\n')

    def parse_dgsol_coords(self, path, n_solutions, n_atoms):
        with open(f'{path}/dgsol.output') as outfile:
            lines = outfile.readlines()

        coords = []
        for line in lines:
            if not line.startswith('\n') and len(line) > 30:
                coords.append([float(n) for n in line.split()])
        coords = np.array(coords).reshape((n_solutions, n_atoms, 3))
        return coords

    def solve_distance_geometry(self, outpath, n_solutions=10):
        construction_errors = []
        mol_coordinates = []
        mol_ids = np.arange(self.distances.shape[0])
        for i, ids in tqdm(enumerate(mol_ids), total=len(mol_ids)):
            out = f'{outpath}/{ids:04}'
            os.makedirs(out, exist_ok=True)
            self.write_dgsol_input(distances=self.distances[i], outpath=out)
            self.run_dgsol(out, n_solutions=n_solutions)
            errors = self.parse_dgsol_errors(out)
            lowest_errors_idx = np.argsort(errors[:, 2])
            construction_errors.append(errors[lowest_errors_idx[0]])
            coords = self.parse_dgsol_coords(out, n_solutions, n_atoms=len(self.nuclear_charges[i]))
            mol_coordinates.append(coords[lowest_errors_idx[0]])
        self.coords = np.array(mol_coordinates)
        self.c_errors = np.array(construction_errors)

    def run_dgsol(self, outpath, n_solutions=10):
        cmd = f'dgsol -s{n_solutions} {outpath}/dgsol.input {outpath}/dgsol.output {outpath}/dgsol.summary'
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if error is not None:
            raise UserWarning(f'{outpath} produced the following error: {error}')

    def parse_dgsol_errors(self, outpath):
        """
        There are 4 types of errors in the dgsol output:

        f_err      The value of the merit function
        derr_min      The smallest error in the distances
        derr_avg      The average error in the distances
        derr_max      The largest error in the distances
        """
        with open(f'{outpath}/dgsol.summary', 'r') as input:
            lines = input.readlines()

        errors = []
        # skip the header lines
        for line in lines[5:]:
            errors.append(line.split()[2:])   # the first two entries are n_atoms and n_distances
        return np.array(errors).astype('float32')
