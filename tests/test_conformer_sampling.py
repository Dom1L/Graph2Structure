from glob import glob
import igraph
import numpy as np
import os
from tqdm import tqdm

import g2s
from g2s.graph_extractor import xyz2mol_graph
from g2s.representations.locality import get_local_environment, get_local_knn_environment, get_unique_environments, get_sparse_bounds_matrix
from g2s.dgeom.lsbuild import lsbuild


def prep_data(filepath, min_neighs):
    bond_order_matrix, nuclear_charges, coords = xyz2mol_graph(filepath)

    distances = g2s.utils.calculate_distances(coords)
    gc = g2s.GraphCompound(bond_order_matrix, nuclear_charges, distances)
    gc.generate_bond_length(size=19, sorting='norm_row')

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

    ha_adj = adjacency[heavy_atom_idxs][:, heavy_atom_idxs]

    ha_local_repr = []
    ha_local_nc = []
    ha_local_dist = []
    ha_local_idxs = []

    hy_local_repr = []
    hy_local_nc = []
    hy_local_dist = []
    hy_local_idxs = []

    for i in range(len(heavy_atom_idxs)):
        atomic_representation, local_nuclear_charges, local_idxs, local_distances = get_local_knn_environment(i, ha_adj,
                                                                                                              ha_representation,
                                                                                                              ha_nc,
                                                                                                              distances=ha_dm,
                                                                                                              depth=2,
                                                                                                              min_neighs=min_neighs)

        ha_local_repr.append(atomic_representation)
        ha_local_dist.append(local_distances)
        ha_local_nc.append(local_nuclear_charges)
        ha_local_idxs.append(local_idxs)

    for i in hydrogen_idxs:
        hydro_atomic_representation, hydro_local_nuclear_charges, hydro_local_idxs, hydro_local_distances = get_local_environment(
            i, representation, nuclear_charges, distances, n_neighs=4)
        hy_local_repr.append(hydro_atomic_representation)
        hy_local_nc.append(hydro_local_nuclear_charges)
        hy_local_dist.append(hydro_local_distances)
        hy_local_idxs.append(hydro_local_idxs)

    mol_data = {'ha_lr': ha_local_repr,
                'ha_lnc': ha_local_nc,
                'ha_ldist': ha_local_dist,
                'ha_lidxs': ha_local_idxs,
                'hy_lr': hy_local_repr,
                'hy_lnc': hy_local_nc,
                'hy_ldist': hy_local_dist,
                'hy_lidxs': hy_local_idxs,
                'nuclear_charges': nuclear_charges,
                'heavy_atom_idxs': heavy_atom_idxs,
                'hydrogen_idxs': hydrogen_idxs,
                'full_dm': distances}

    return mol_data


def get_envs(mol_envs, boundaries=True, tight_envs=True):
    ha_lr = [mol_envs[i]['ha_lr'] for i in range(len(mol_envs))]
    ha_lidxs = [mol_envs[i]['ha_lidxs'] for i in range(len(mol_envs))]
    ha_ldist = [mol_envs[i]['ha_ldist'] for i in range(len(mol_envs))]

    uq_ha_env = get_unique_environments(ha_lr, ha_lidxs, local_distances=ha_ldist)

    sparse_distances = []
    mol_nuclear_charges = []
    for i in tqdm(range(len(mol_envs))):
        sp, mnc = get_sparse_bounds_matrix(uq_ha_env, mol_envs[i], boundaries=boundaries, tight_envs=tight_envs)
        sparse_distances.append(sp)
        mol_nuclear_charges.append(mnc)

    return np.array(sparse_distances), np.array(mol_nuclear_charges)


def sample_bases(outdir, sparse_distances, nuclear_charges, t=0.5, seed=1337):
    potential_bases = []
    for j in range(len(sparse_distances)):
        n_atoms = len(nuclear_charges[j])
        pb = [[i, *np.where(sparse_distances[j][i] != 0.)[0]] for i in range(n_atoms) if np.where(sparse_distances[j][i] != 0.)[0].shape[0] >= 4]
        np.random.seed(seed)
        bases = [sorted([i, *np.random.choice(pb[i][1:], 3, replace=False)]) for i in range(n_atoms)]
        setB = np.tile(np.array([False] * n_atoms), n_atoms).reshape(n_atoms, n_atoms)
        for i in range(n_atoms):
            setB[i][bases[i]] = True
        potential_bases.append((bases, setB))

    for i in tqdm(range(len(sparse_distances))):
        if os.path.isfile(f'{outdir}/{i:04d}/lsbuild_conf_{len(potential_bases[i][1])}.xyz'):
            continue
        coords = [lsbuild(sparse_distances[i], nuclear_charges[i],
                          lstB=potential_bases[i][0][j], setB=potential_bases[i][1][j], t=t)['x'] for
                  j in range(len(potential_bases[i]))]

        os.makedirs(f'{outdir}/{i:04d}', exist_ok=True)
        for j in range(len(coords)):
            g2s.utils.write_xyz(f'{outdir}/{i:04d}/lsbuild_conf_{j}.xyz', coords[j], nuclear_charges[i])


def sample_t(outdir, sparse_distances, nuclear_charges):
    for i in tqdm(range(len(sparse_distances))):
        if os.path.isfile(f'{outdir}/{i:04d}/lsbuild_conf_9.xyz'):
            continue
        coords = [lsbuild(sparse_distances[i], nuclear_charges[i],
                          lstB=None, setB=None, t=t)['x'] for t in np.arange(0.1, 1.0, 0.1)]

        os.makedirs(f'{outdir}/{i:04d}', exist_ok=True)
        for j in range(len(coords)):
            g2s.utils.write_xyz(f'{outdir}/{i:04d}/lsbuild_conf_{j}.xyz', coords[j], nuclear_charges[i])


def sample_random(outdir, sparse_distances, nuclear_charges, n_confs, seed=1337):
    for i in tqdm(range(len(sparse_distances))):
        np.random.seed(seed)
        n_atoms = len(nuclear_charges[i])
        pb = [[k, *np.where(sparse_distances[i][k] != 0.)[0]] for k in range(n_atoms) if np.where(sparse_distances[i][k] != 0.)[0].shape[0] >= 4]
        coords = []
        for j in range(n_confs):
            if os.path.isfile(f'{outdir}/{i:04d}/lsbuild_conf_{j}.xyz'):
                continue
            k = np.random.randint(0, len(pb))
            t = np.random.uniform(0.1, 1.0)
            base = sorted([k, *np.random.choice(pb[k][1:], 3, replace=False)])
            setB = np.array([False] * n_atoms)
            setB[base] = True
            coords.append(lsbuild(sparse_distances[i], nuclear_charges[i], lstB=base, setB=setB, t=t)['x'])

        os.makedirs(f'{outdir}/{i:04d}', exist_ok=True)
        if not coords:
            continue
        for j in range(len(coords)):
            g2s.utils.write_xyz(f'{outdir}/{i:04d}/lsbuild_conf_{j}.xyz', coords[j], nuclear_charges[i])


def run_conf_sampling(filepath, outpath):
    ci_path = sorted(glob(f'{filepath}/*.xyz'))
    mol_envs_4 = [prep_data(ci, 4) for ci in ci_path]
    mol_envs_5 = [prep_data(ci, 5) for ci in ci_path]

    bound_sampling(mol_envs_4, f'{outpath}/4nbh_env')
    bound_sampling(mol_envs_5, f'{outpath}/5nbh_env')


def bound_sampling(env, outpath):
    dg_sampling(*get_envs(env, boundaries=False, tight_envs=True), f'{outpath}/no_bounds')
    dg_sampling(*get_envs(env, boundaries=True, tight_envs=True), f'{outpath}/tight_bounds')
    dg_sampling(*get_envs(env, boundaries=True, tight_envs=False), f'{outpath}/loose_bounds')


def dg_sampling(sparse_distances, nuclear_charges, outpath):
    sample_bases(f'{outpath}/base_sampling', sparse_distances, nuclear_charges, t=0.5, seed=1337)
    sample_t(f'{outpath}/t_sampling', sparse_distances, nuclear_charges)
    sample_random(f'{outpath}/random_sampling', sparse_distances, nuclear_charges, n_confs=10, seed=1337)


if __name__ == '__main__':
    run_conf_sampling('/data/lemm/g2s/const_isomers', '/data/lemm/g2s/conf_sampling')
