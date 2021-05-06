import os
from glob import glob
import pickle

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
    # u_sim = sim_fchl[uidxs, :][:, uidxs]
    return u_confs, n_confs, uidxs

def calc_similarities(path):
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
    for e in envs:
        for m in modes:
            for s in sampling:
                conf_paths = sorted(glob(f'{path}/{e}/{m}/{s}/*'))
                for c in tqdm(conf_paths):
                    _rmsds = get_rmsd(c)
                    fchl_sim = get_fchlsim(c)
                    sim_dict[e][m][s].append([_rmsds, fchl_sim])


def filter_duplicates(filepath, path):
    filepath = '/data/lemm/g2s/conf_sampling_results.pkl'
    path = '/data/lemm/g2s/conf_sampling'
    sim_dict = pickle.load(open(filepath, 'rb'))

    envs = ['4nbh_env', '5nbh_env']
    modes = ['loose_bounds', 'no_bounds', 'tight_bounds']
    sampling = ['base_sampling','random_sampling', 't_sampling']

    uq_dict = {'4nbh_env':
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
    for e in envs:
        for m in modes:
            for s in sampling:
                for c in tqdm(sim_dict[e][m][s]):
                    rmsds, fchl_sim, ref_rmsd, ref_fchl = c
                    u_confs, n_confs, uidxs = count_duplicates(fchl_sim, cutoff=0.98, higher=True)

                    uq_dict[e][m][s].append([rmsds[uidxs][:, uidxs], fchl_sim[uidxs][:, uidxs],
                                             np.array(ref_rmsd)[uidxs], ref_fchl[0][uidxs], u_confs, n_confs])


    pickle.dump(uq_dict, open('/data/lemm/g2s/conf_sampling_results_unique.pkl', 'wb'))