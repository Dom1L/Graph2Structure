from glob import glob

import numpy as np

import g2s
from g2s.graph_extractor import xyz2mol_graph, xtb_graph
from g2s.representations.locality import get_local_environment, construct_sparse_dm



test_mols = sorted(glob('/Users/c0uch1/github/Graph2Structure/tests/test_files/local_test/*.xyz'))

bond_order_matrix, nuclear_charges, coords = xyz2mol_graph(test_mols[1])

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

    hy_local_repr = []
    hy_local_nc = []
    hy_local_dist = []
    hy_local_idxs = []

    for i in range(len(heavy_atom_idxs)):
        atomic_representation, local_nuclear_charges, local_idxs, local_distances = get_local_environment(i, ha_representation, ha_nc, ha_dm, n_neighs=n)
        ha_local_repr.append(atomic_representation)
        ha_local_dist.append(local_distances)
        ha_local_nc.append(local_nuclear_charges)
        ha_local_idxs.append(local_idxs)

    for i in hydrogen_idxs:
        hydro_atomic_representation, hydro_local_nuclear_charges, hydro_local_idxs, hydro_local_distances = get_local_environment(i, representation, nuclear_charges, distances, n_neighs=4)
        hy_local_repr.append(hydro_atomic_representation)
        hy_local_nc.append(hydro_local_nuclear_charges)
        hy_local_dist.append(hydro_local_distances)
        hy_local_idxs.append(hydro_local_idxs)

    sparse_dm = construct_sparse_dm(ha_local_dist, ha_local_idxs, n_atoms=len(heavy_atom_idxs), n_neighs=n+1)
    hy_sparse_dm = construct_sparse_dm(hy_local_dist, hy_local_idxs, n_atoms=len(nuclear_charges), n_neighs=5)
    for j, e in enumerate(heavy_atom_idxs):
        for k, em in enumerate(heavy_atom_idxs):
            hy_sparse_dm[e, em] = sparse_dm[j, k]
            hy_sparse_dm[em, e] = sparse_dm[k, j]

    distance_matrices.append(sparse_dm)

dgsol = DGSOL(np.array([hy_sparse_dm]),   np.repeat(np.array([nuclear_charges]),1, axis=0), vectorized_input=False)
dgsol.solve_distance_geometry('./test_molecule_flat', n_solutions=20)

dirs = sorted(glob('./test_molecules2/*'))
for i, e in enumerate(range(10)):
    g2s.utils.write_xyz(f'test_molecule_flat/lin_{e}hy.xyz', dgsol.coords[0][i], dgsol.nuclear_charges[0])