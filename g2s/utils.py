import numpy as np
from scipy.spatial.distance import squareform, pdist

from .constants import periodic_table


def write_xyz(outname, coords, elements, element_input='nuclear_charges'):
    if element_input == 'nuclear_charges':
        elements = [periodic_table[int(nc)] for nc in elements]
    with open(f'{outname}', 'w') as outfile:
        outfile.write(f'{len(elements)}\n')
        outfile.write('\n')
        for xyz, nc in zip(coords, elements):
            outfile.write(f'{nc}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\n')


def combine_heavy_hydrogen_coords(heavy_coords, hydrogen_coords, heavy_nuclear_charges):
    return np.vstack((heavy_coords, hydrogen_coords)), np.array([*heavy_nuclear_charges, *[1]*len(hydrogen_coords)])


def vector_to_square(vectorized_distances):
    return np.array([squareform(vec_dist) for vec_dist in vectorized_distances])


def calculate_distances(coordinates):
    return squareform(pdist(coordinates, lambda a, b: np.sqrt(np.sum((a - b) ** 2))))


def filter_nonzero_distances(padded_distances, nuclear_charges):
    distances = []

    for i in range(len(nuclear_charges)):
        n_atoms = len(nuclear_charges[i])

        distmat = vector_to_square([padded_distances[i]])[0]
        distances.append(distmat[:n_atoms][:, :n_atoms])
    return np.array(distances, dtype=object)
