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
            outfile.write(f'{elements}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\n')


def vector_to_square(vectorized_distances):
    return np.array([squareform(vec_dist) for vec_dist in vectorized_distances])


def calculate_distances(coordinates):
    return squareform(pdist(coordinates, lambda a, b: np.sqrt(np.sum((a - b) ** 2))))


def filter_nonzero_distances(padded_distances):
    distances = [padded_distances[i][np.where(padded_distances[i] != 0.0)[0]] for i in range(len(padded_distances))]
    return np.array(distances)
