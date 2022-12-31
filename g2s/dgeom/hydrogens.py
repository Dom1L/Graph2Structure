import numpy as np

try:
    import quadpy
    scheme = quadpy.sphere.lebedev_131()
except:
    pass

def lebedev_sphere_opt(bond, points_distances, origin):
    """
    Draws a Lebedev sphere around a central atom.
    Assumes that the bonded atom is in the origin

    Parameters
    ----------
    bond: float
        Bond length of heavy atom to hydrogen bond.
    points_distances: tuple
        Contains positions (coordinates) of atoms and distances of atoms to the hydrogen in question.
    origin: np.array
        Coordinates of the central heavy atom.

    Returns
    -------
    candidate_position: np.array
        Coordinates of the hydrogen position with the lowest distance error to given points.

    """
    candidate_positions = scheme.points * bond
    deltas = np.sum(
        [(np.linalg.norm(candidate_positions - (pos - origin), axis=1) - dist) ** 2 for pos, dist in points_distances],
        axis=0)
    return candidate_positions[np.argmin(deltas)] + origin


def get_hydrogen_positions(positions, distances, n_hydrogens):
    """
    Calculates hydrogen positions around a given central atom.
    Can return a maximum number of 3 hydrogen positions.

    Parameters
    ----------
    positions: np.array, shape(4, 3)
        Heavy atom coordinates of the closest 4 heavy atoms in the hydrogen environment.
    distances: np.array, shape(5)
        Distances to the hydrogen in investion. First 4 distances correspond to heavy atom hydrogen distances.
        The last distance is the distance between two hydrogens in case multiple hydrogens are being attached.
    n_hydrogens: int
        Number of hydrogens to attach.

    Returns
    -------
    pos_hydr_1: np.array, shape(3)
        Coordinates of a hydrogen atom.
    pos_hydr_2: np.array, shape(3)
        Coordinates of a hydrogen atom. Is None when non-existent.
    pos_hydr_3: np.array, shape(3)
        Coordinates of a hydrogen atom. Is None when non-existent.
    """
    h_h_distance = distances[-1]
    pos_1 = positions[0]
    dist_1 = distances[0]
    pos_hydr_1 = lebedev_sphere_opt(bond=dist_1, origin=pos_1, points_distances=zip(positions[1:], distances[1:-1]))
    if n_hydrogens > 1:
        p_dist_pairs = list(zip(positions[1:-1], distances[1:-2])) if len(positions) > 2 else list(zip(positions[1], distances[1]))
        p_dist_pairs.append((pos_hydr_1, h_h_distance))
        pos_hydr_2 = lebedev_sphere_opt(bond=dist_1, origin=pos_1, points_distances=p_dist_pairs)
    else:
        pos_hydr_2 = None
    if n_hydrogens == 3:
        p_dist_pairs = [(positions[1], distances[1]), (pos_hydr_1, h_h_distance), (pos_hydr_2, h_h_distance)]
        pos_hydr_3 = lebedev_sphere_opt(bond=dist_1, origin=pos_1, points_distances=p_dist_pairs)
    else:
        pos_hydr_3 = None
    return pos_hydr_1, pos_hydr_2, pos_hydr_3


def hydrogen_lebedev_reconstruction(heavy_atom_coords, predicted_h_distances, heavy_atom_hydrogen_mapping):
    """
    Main function to start hydrogen reconstruction.
    Uses a Lebedev optimization scheme to determine hydrogen coordinates.

    Parameters
    ----------
    heavy_atom_coords: np.array, shape(n_atoms, 3)
        Heavy atom coordinates.
    predicted_h_distances: np.array, shape(n_hyrogens, 5)
        Predicted hydrogen distances of each hydrogen.
    heavy_atom_hydrogen_mapping: tuple
        Contains mapping indices for (n_hydrogens_total, (central_heavy_atom, n_hydrogens_on_atom, neighbor_indices)).

    Returns
    -------
    hydrogen_coords: np.array, shape(n_hydrogens, 3)
        Calculated hydrogen coordinates.

    """
    hydrogen_coords = []
    for i in range(len(heavy_atom_hydrogen_mapping)):
        n_hydrogens = len(heavy_atom_hydrogen_mapping[i][1])
        nbh_idxs = heavy_atom_hydrogen_mapping[i][2]
        center_atom_idx = heavy_atom_hydrogen_mapping[i][0]
        pred_positions = heavy_atom_coords[np.array([center_atom_idx, *nbh_idxs])]
        pred_h_distances = predicted_h_distances[i]

        pos_hydr_1, pos_hydr_2, pos_hydr_3 = get_hydrogen_positions(pred_positions, pred_h_distances,
                                                                    n_hydrogens=n_hydrogens)
        hydrogen_coords.append(pos_hydr_1)
        if pos_hydr_2 is not None:
            hydrogen_coords.append(pos_hydr_2)
        if pos_hydr_3 is not None:
            hydrogen_coords.append(pos_hydr_3)
    return np.array(hydrogen_coords)
