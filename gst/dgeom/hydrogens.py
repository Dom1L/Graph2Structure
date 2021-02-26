import quadpy
import numpy as np


scheme = quadpy.sphere.lebedev_131()


def lebedev_sphere_opt(bond, points_distances, origin):
    """ Assumes that the bonded atom is in the origin, the other two are at pos1, pos2 with proton distances d1 and d2, respectively."""
    candidate_positions = scheme.points * bond
    deltas = np.sum(
        [(np.linalg.norm(candidate_positions - (pos - origin), axis=1) - dist) ** 2 for pos, dist in points_distances],
        axis=0)
    return candidate_positions[np.argmin(deltas)] + origin


def get_hydrogen_positions(positions, distances, n_h):
    h_h_distance = 1.77  # Mean distance between Hydrogens, stdev 0.01, Angstrom
    pos_1 = positions[0]
    dist_1 = distances[0]
    pos_hydr_1 = lebedev_sphere_opt(bond=dist_1, origin=pos_1, points_distances=zip(positions[1:], distances[1:]))
    if n_h > 1:
        p_dist_pairs = list(zip(positions[1:-1], distances[1:-1])) if len(positions) > 2 else list(zip(positions[1], distances[1]))
        p_dist_pairs.append((pos_hydr_1, h_h_distance))
        pos_hydr_2 = lebedev_sphere_opt(bond=dist_1, origin=pos_1, points_distances=p_dist_pairs)
    else:
        pos_hydr_2 = None
    if n_h == 3:
        p_dist_pairs = [(positions[1], distances[1]), (pos_hydr_1, h_h_distance), (pos_hydr_2, h_h_distance)]
        pos_hydr_3 = lebedev_sphere_opt(bond=dist_1, origin=pos_1, points_distances=p_dist_pairs)
    else:
        pos_hydr_3 = None
    return pos_hydr_1, pos_hydr_2, pos_hydr_3


def hydrogen_reconstruction(atom_idxs, heavy_atom_coords, predicted_h_distances):
    hydrogen_coords = []
    for h_id in range(len(atom_idxs)):
        n_heavy = 4 if heavy_atom_coords.shape[0] >= 4 else heavy_atom_coords.shape[0]
        n_h = len(atom_idxs[h_id]) - n_heavy
        neighbor_idx = atom_idxs[h_id][n_h:]
        h1_idx = atom_idxs[h_id][0]
        pred_positions = heavy_atom_coords[neighbor_idx]
        pred_h_distances = predicted_h_distances[h_id]

        pos_hydr_1, pos_hydr_2, pos_hydr_3 = get_hydrogen_positions(pred_positions, pred_h_distances, n_h=n_h)
        hydrogen_coords.append(pos_hydr_1)
        if pos_hydr_2 is not None:
            hydrogen_coords.append(pos_hydr_2)
        if pos_hydr_3 is not None:
            hydrogen_coords.append(pos_hydr_3)
    return hydrogen_coords

