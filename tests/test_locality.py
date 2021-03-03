from glob import glob

import numpy as np

import g2s
from g2s.graph_extractor import xyz2mol_graph, xtb_graph



test_mols = sorted(glob('/home/lem/github/Graph2Structure/tests/test_files/local_test/*.xyz'))

bond_order_matrix, nuclear_charges, coords = xtb_graph(test_mols[1])

distances = g2s.utils.calculate_distances(coords)
gc = g2s.GraphCompound(bond_order_matrix, nuclear_charges, distances)
gc.generate_bond_length(size=len(nuclear_charges), sorting='norm_row')

representation = gc.representation
adjacency = gc.adjacency_matrix
nuclear_charges = gc.nuclear_charges
distances = gc.distances

representation = g2s.utils.vector_to_square([representation])[0]
distances = g2s.utils.vector_to_square([distances])[0]

hydrogen_idxs = np.where(nuclear_charges == 1)[0]
heavy_atom_idxs = np.where(nuclear_charges != 1)[0]

ha_representation = representation[heavy_atom_idxs][:, heavy_atom_idxs]
hy_representation = representation[hydrogen_idxs][:, heavy_atom_idxs]

ha_dm = distances[heavy_atom_idxs][:, heavy_atom_idxs]
ha_nc = nuclear_charges[heavy_atom_idxs]

distance_matrices = []
for n in range(3, 10):
    ha_local_repr = []
    ha_local_nc = []
    ha_local_dist = []
    ha_local_idxs = []

    for i in range(len(heavy_atom_idxs)):
        atomic_representation, local_nuclear_charges, local_idxs, local_distances = get_local_environment(i, ha_representation, ha_nc, ha_dm, n_neighs=n)
        ha_local_repr.append(atomic_representation)
        ha_local_dist.append(local_distances)
        ha_local_nc.append(local_nuclear_charges)
        ha_local_idxs.append(local_idxs)

    sparse_dm = construct_sparse_dm(ha_local_dist, ha_local_idxs, n_atoms=len(heavy_atom_idxs), n_neighs=n+1)
    distance_matrices.append(sparse_dm)

dgsol = DGSOL(np.array(distance_matrices),   np.repeat(np.array([ha_nc]),7, axis=0), vectorized_input=False)
dgsol.solve_distance_geometry('./test_molecules2')

dirs = sorted(glob('/tmp/tmp66p26c7e/test_molecules2/*'))
for i, e in enumerate(range(3, 10)):
    g2s.utils.write_xyz(f'{dirs[i]}/cyclo_{e}.xyz', dgsol.coords[i], dgsol.nuclear_charges[i])