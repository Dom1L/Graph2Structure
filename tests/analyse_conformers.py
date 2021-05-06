import os
import pickle
from glob import glob

import numpy as np
os.environ["OMP_NUM_THREADS"] = str(24)
os.environ["MKL_NUM_THREADS"] = str(24)

from tqdm import tqdm
import qml.representations as qml
from qml.kernels import get_global_kernel
from g2s.constants import periodic_table
from g2s.utils import read_xyz
from gtg.reconstruction.rmsd import calc_rmsd


def get_rmsd(filepath):
    # calculate RMSD of everything against everything
    confs = sorted(glob(f'{filepath}/*.xyz'))
    n_confs = len(confs)
    rmsds = np.zeros((n_confs, n_confs))
    n, m = np.triu_indices(n_confs, k=1)
    for i, j in zip(n, m):
        _rmsd = calc_rmsd(confs[i], confs[j])
        rmsds[i, j] = _rmsd
        rmsds[j, i] = _rmsd
    return np.array(rmsds)


def get_fchlsim(filepath):
    confs = sorted(glob(f'{filepath}/*.xyz'))
    coords = []
    for c in confs:
        nuclear_charges, crd = read_xyz(c)
        coords.append(crd)
    coords = np.array(coords)
    sigma = 2
    nuclear_charges = [periodic_table.index(e) for e in nuclear_charges]
    fchl_rep = []
    for conf in coords:
        fchl_rep.append(qml.generate_fchl_acsf(nuclear_charges, conf, pad=len(nuclear_charges)))

    sim_fchl = get_global_kernel(np.array(fchl_rep), np.array(fchl_rep),
                                 np.array([list(nuclear_charges)] * len(fchl_rep)),
                                 np.array([list(nuclear_charges)] * len(fchl_rep)), sigma)

    return sim_fchl


def get_ref_sim(filepaths, reference):
    confs = sorted(glob(f'{filepaths}/*.xyz'))
    n_confs = len(confs)
    rmsds = [calc_rmsd(confs[i], reference) for i in range(n_confs)]

    coords = []
    for c in confs:
        nuclear_charges, crd = read_xyz(c)
        coords.append(crd)
    coords = np.array(coords)
    ref_nuclear_charges, ref_coords = read_xyz(reference)
    ref_coords = np.array(ref_coords)

    sigma = 2
    nuclear_charges = [periodic_table.index(e) for e in nuclear_charges]
    ref_nuclear_charges = np.array([periodic_table.index(e) for e in ref_nuclear_charges])
    noh_idx = np.where(ref_nuclear_charges != 1)[0]
    ref_coords = ref_coords[noh_idx]
    ref_nuclear_charges = ref_nuclear_charges[noh_idx]

    ref_fchl = qml.generate_fchl_acsf(ref_nuclear_charges, ref_coords, pad=len(nuclear_charges))

    fchl_rep = []
    for conf in coords:
        fchl_rep.append(qml.generate_fchl_acsf(nuclear_charges, conf, pad=len(nuclear_charges)))

    sim_fchl = get_global_kernel(np.array(fchl_rep), np.array([ref_fchl]),
                                 np.array([list(nuclear_charges)] * len(fchl_rep)),
                                 np.array([list(ref_nuclear_charges)]), sigma)

    return rmsds, sim_fchl


def count_duplicates(sim_fchl, cutoff=0.98, higher=True):
    n_confs = sim_fchl.shape[0]
    f = sim_fchl[np.triu_indices(n_confs, k=1)]
    i, j = np.triu_indices(n_confs, k=1)

    if higher:
        idx_sim = np.where(f > cutoff)[0]
    else:
        idx_sim = np.where(f < cutoff)[0]

    dupl_dict = {k: [] for k in range(n_confs)}
    for dupl_idx in idx_sim:
        n = False
        for conf in dupl_dict.keys():
            if i[dupl_idx] in dupl_dict[conf]:
                if j[dupl_idx] not in dupl_dict[conf]:
                    dupl_dict[conf].append(j[dupl_idx])
                n = True
        if n:
            dupl_dict.pop(j[dupl_idx], None)
        else:
            dupl_dict[i[dupl_idx]].append(j[dupl_idx])
            dupl_dict.pop(j[dupl_idx], None)

    u_confs = len(dupl_dict.keys())
    uidxs = np.array(list(dupl_dict.keys()))
    u_sim = sim_fchl[uidxs, :][:, uidxs]
    return u_confs, u_sim


def calc_similarities(path, ref_structures):
    envs = ['4nbh_env', '5nbh_env']
    modes = ['loose_bounds', 'no_bounds', 'tight_bounds']
    sampling = ['base_sampling','random_sampling', 't_sampling']

    sim_dict = {'4nbh_env':
                    {
                        'loose_bounds': {'base_sampling':[],'random_sampling':[], 't_sampling':[]},
                        'no_bounds':{'base_sampling':[],'random_sampling':[], 't_sampling':[]},
                        'tight_bounds':{'base_sampling':[],'random_sampling':[], 't_sampling':[]}
                    },
                '5nbh_env':
                    {
                        'loose_bounds': {'base_sampling': [], 'random_sampling': [], 't_sampling': []},
                        'no_bounds': {'base_sampling': [], 'random_sampling': [], 't_sampling': []},
                        'tight_bounds': {'base_sampling': [], 'random_sampling': [], 't_sampling': []}
                    }
    }
    for e in tqdm(envs):
        for m in tqdm(modes, leave=False):
            for s in tqdm(sampling, leave=False):
                conf_paths = sorted(glob(f'{path}/{e}/{m}/{s}/*'))
                for i, c in tqdm(enumerate(conf_paths), total=len(conf_paths), leave=False):
                    _rmsds = get_rmsd(c)
                    fchl_sim = get_fchlsim(c)
                    ref_rmsd, ref_fchl = get_ref_sim(c, ref_structures[i])
                    sim_dict[e][m][s].append([_rmsds, fchl_sim, ref_rmsd, ref_fchl])

    return sim_dict


if __name__ == '__main__':
    bench_results = calc_similarities('/data/lemm/g2s/conf_sampling', sorted(glob('/data/lemm/g2s/const_isomers/*.xyz')))
    pickle.dump(bench_results, open('/data/lemm/g2s/conf_sampling_results.pkl', 'wb'))
