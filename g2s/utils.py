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


def read_xyz(filename, return_string=False):
    with open(filename, 'r') as infile:
        lines = infile.readlines()
    if return_string:
        return lines[2:]
    else:
        elements = []
        coords = []
        for line in lines[2:]:
            atom, x, y, z = line.split()
            elements.append(atom)
            coords.append([x, y, z])
        return elements, np.array(coords).astype('float')


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


def mae(prediction, reference):
    return np.mean(np.abs(prediction - reference))


def weighted_mean_mae(prediction, reference):
    """
    Weighted mean MAE to calculate the true MAE for distance matrices of unequal sizes.
    Since for the KRR prediction part, a zero padded distance matrix is required
    (technically not, but makes life easier),
    blindly calculating the MAE of the G2S prediction will be overly optimistic
    since zeros are predicted almost perfectly.

    To account for this, a weighted error has to be calculated.
    This function loops over every distance matrix entry, filters zeros and calculates the true MAE for each entry.
    The final MAE is calculated by weighting each individual MAE depending on the amount of non zero entries of
    each distance matrix entry.

    For a distance matrix without zeros, the weighted mean MAE equals the mean MAE.

    """
    _mae = []
    n_samples = []
    for i in range(reference.shape[1]):
        non_zero_idx = np.where(reference[:, i] != 0.0)[0]
        if non_zero_idx.size == 0:
            continue
        non_zero_pred = prediction[:, i][non_zero_idx]
        non_zero_ref = reference[:, i][non_zero_idx]
        _mae.append(mae(non_zero_pred, non_zero_ref))
        n_samples.append(len(non_zero_idx))
    _mae = np.array(_mae)
    n_samples = np.array(n_samples)

    # Total amount datapoints
    n_total = len(_mae)
    weights = (n_samples / n_total)
    return np.sum(weights * _mae) / np.sum(weights)

