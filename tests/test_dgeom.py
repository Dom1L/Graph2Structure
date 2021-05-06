import os
import pickle
from glob import glob
from copy import deepcopy

import numpy as np
from rdkit import DistanceGeometry
from tqdm import tqdm

import g2s
from g2s.utils import write_xyz, combine_heavy_hydrogen_coords, vector_to_square, read_xyz, calculate_distances
from g2s.dgeom import hydrogen_lebedev_reconstruction
# from g2s.dgeom.lsbuild import lsbuild
from g2s.dgeom.dgsol import DGSOL
from g2s.dgeom.molassembler import Molassembler
from g2s.graph_extractor.xyz2mol import xyz2mol_graph

from lsbuild import lsbuild

from gtg.reconstruction.rmsd import calc_rmsd


def test_molassembler(bond_order_mat, heavy_dist, nuclear_charges, heavy_hydro_dist, heavy_hydro_map, outpath, stereo):
    molass = Molassembler(bond_order_mat, heavy_dist, nuclear_charges, vectorized_input=False)
    molass.convert_molecules()
    molass.set_stereochem(stereo)
    molass.solve_distance_geometry(n_conformers=1, seed=42, max_attempts=20)
    pred_coords = molass.coords

    for i, e in enumerate(sorted(glob(f'{outpath}/*'))):
        if np.sum(pred_coords[i]) == 0.0:
            continue
        if heavy_hydro_map[i].size > 0.:
            h_coords = hydrogen_lebedev_reconstruction(pred_coords[i][0].astype(float), heavy_hydro_dist[i],
                                                       heavy_hydro_map[i])
            heavy_c, all_nc = combine_heavy_hydrogen_coords(pred_coords[i][0].astype(float), h_coords,
                                                            nuclear_charges[i])
            write_xyz(f'{e}/molass_h.xyz', heavy_c, all_nc)
        else:
            write_xyz(f'{e}/molass_h.xyz', pred_coords[i][0].astype(float), nuclear_charges[i])


def test_dgsol(heavy_dist, nuclear_charges, heavy_hydro_dist, heavy_hydro_map, outpath):
    dgsol = DGSOL(heavy_dist, nuclear_charges, vectorized_input=True)
    dgsol.solve_distance_geometry(f'{outpath}')
    for i, e in enumerate(sorted(glob(f'{outpath}/*'))):
        if heavy_hydro_map[i].size > 0.:
            h_coords = hydrogen_lebedev_reconstruction(dgsol.coords[i][0].astype(float), heavy_hydro_dist[i],
                                                       heavy_hydro_map[i])
            heavy_c, all_nc = combine_heavy_hydrogen_coords(dgsol.coords[i][0].astype(float), h_coords,
                                                            nuclear_charges[i])
            write_xyz(f'{e}/dgsol_h.xyz', heavy_c, all_nc)
        else:
            write_xyz(f'{e}/dgsol_h.xyz', dgsol.coords[i][0].astype(float), dgsol.nuclear_charges[i])


def test_lsbuild(heavy_dist, nuclear_charges, heavy_hydro_dist, heavy_hydro_map, outpath):
    pred_coords = []

    for m in tqdm(sorted(glob(f'{outpath}/*/dgsol.input'))):
        # predc = lsbuild(vector_to_square([dist])[0])
        predc = lsbuild(m)['x']
        pred_coords.append(predc)
    pred_coords = np.array(pred_coords)

    for i, e in enumerate(sorted(glob(f'{outpath}/*'))):
        if heavy_hydro_map[i].size > 0.:
            h_coords = hydrogen_lebedev_reconstruction(pred_coords[i].astype(float), heavy_hydro_dist[i],
                                                       heavy_hydro_map[i])
            heavy_c, all_nc = combine_heavy_hydrogen_coords(pred_coords[i].astype(float), h_coords,
                                                            nuclear_charges[i])
            write_xyz(f'{e}/lsbuild_h.xyz', heavy_c, all_nc)
        else:
            write_xyz(f'{e}/lsbuild_h.xyz', pred_coords[i].astype(float), nuclear_charges[i])


def predict_all(filepath, outpath):
    data = pickle.load(open(filepath, 'rb'))
    test_idx = data['test_idx']
    bo, dists, nc, ss = data['bond_order'][test_idx], data['predictions'], data['nuclear_charges'][test_idx], data['stereocenters'][test_idx]
    hpred, hhm, gt_coords, gt_nc = data['heavy_hydro_dist'][test_idx], data['heavy_hydro_map'][test_idx], data['coords'][test_idx], data['full_nc'][test_idx]

    lrc = data['lrc']

    for i, e in enumerate(lrc):
        os.makedirs(f'{outpath}/{e}', exist_ok=True)
        test_dgsol(dists[i], nc, hpred, hhm, f'{outpath}/{e}')
        # test_lsbuild(dists[i], nc, hpred, hhm, f'{outpath}/{e}')
        # sqr_dists = deepcopy(g2s.utils.vector_to_square(dists[i]))
        # for q in sqr_dists:
        #     DistanceGeometry.DoTriangleSmoothing(q, 0.5)
        # test_molassembler(bo, sqr_dists, nc, hpred, hhm,  f'{outpath}/{e}', ss)
        for j, b in enumerate(sorted(glob(f'{outpath}/{e}/*'))):
            write_xyz(f'{b}/gt.xyz', gt_coords[j], gt_nc[j])


def analyse_results(filepath, outpath):
    data = pickle.load(open(filepath, 'rb'))
    test_idx = data['test_idx']
    dists, nc, ss = data['predictions'], data['nuclear_charges'][test_idx], data['stereocenters'][test_idx]
    gt_dist = data['padded_distances'][test_idx]

    lrc = data['lrc']

    results = {
        'lrc': lrc,
        'pred_dists': dists,
        'gt_dists': gt_dist,
        'gt_ss': ss,
        'dgsol_rc': [],
        'dgsol_mae': [],
        'dgsol_rmsd': [],
        'dgsol_ss': [],
        'lsbuild_rc': [],
        'lsbuild_mae': [],
        'lsbuild_rmsd': [],
        'lsbuild_ss': [],
        'molass_rc': [],
        'molass_mae': [],
        'molass_rmsd': [],
        'molass_ss': []
    }
    for i, e in enumerate(lrc):
        dgsol_rc = []
        dgsol_mae = []
        dgsol_rmsd = []
        dgsol_chiral = []

        lsbuild_recd = []
        lsbuild_mae = []
        lsbuild_rmsd = []
        lsbuild_chiral = []

        molass_recd = []
        molass_mae = []
        molass_rmsd = []
        molass_chiral = []

        for j, b in tqdm(enumerate(sorted(glob(f'{outpath}/{e}/*'))), total=len(gt_dist)):
            nc, dgsol_coords = read_xyz(f'{b}/dgsol_h.xyz')
            _, lsbuild_coords = read_xyz(f'{b}/lsbuild_h.xyz')
            heavy_idxs = np.where(np.array(nc) != 'H')[0]
            triu = np.triu_indices(len(heavy_idxs), k=1)
            gt = vector_to_square([gt_dist[j][np.where(gt_dist[j] != 0.)[0]]])[0][heavy_idxs][:, heavy_idxs][triu]
            pdist = vector_to_square([np.array(dists[i, j])])[0][heavy_idxs][:, heavy_idxs][triu]
            dgsol_dists = calculate_distances(dgsol_coords)[heavy_idxs][:, heavy_idxs][triu]
            lsbuild_dists = calculate_distances(lsbuild_coords)[heavy_idxs][:, heavy_idxs][triu]
            try:
                _, _, _, dgsol_ss = xyz2mol_graph(f'{b}/dgsol_h.xyz', get_chirality=True)
                dgsol_chiral.append(dgsol_ss)
            except:
                dgsol_chiral.append(np.nan)
            try:
                _, _, _, lsbuild_ss = xyz2mol_graph(f'{b}/lsbuild_h.xyz', get_chirality=True)
                lsbuild_chiral.append(lsbuild_ss)
            except:
                lsbuild_chiral.append(np.nan)

            dgsol_rc.append(mae(pdist, dgsol_dists))
            dgsol_mae.append(mae(gt, dgsol_dists))
            dgsol_rmsd.append(calc_rmsd(f'{b}/dgsol_h.xyz', f'{b}/gt.xyz'))

            lsbuild_recd.append(mae(pdist, lsbuild_dists))
            lsbuild_mae.append(mae(gt, lsbuild_dists))
            lsbuild_rmsd.append(calc_rmsd(f'{b}/lsbuild_h.xyz', f'{b}/gt.xyz'))

            if os.path.isfile(f'{b}/molass_h.xyz'):
                _, molass_coords = read_xyz(f'{b}/molass_h.xyz')
                molass_dists = calculate_distances(molass_coords)[heavy_idxs][:, heavy_idxs][triu]
                molass_recd.append(mae(pdist, molass_dists))
                molass_mae.append(mae(gt, molass_dists))
                molass_rmsd.append(calc_rmsd(f'{b}/molass_h.xyz', f'{b}/gt.xyz'))
                try:
                    _, _, _, molass_ss = xyz2mol_graph(f'{b}/molass_h.xyz', get_chirality=True)
                    molass_chiral.append(molass_ss)
                except:
                    molass_chiral.append(np.nan)
            else:
                molass_recd.append(np.nan)
                molass_mae.append(np.nan)
                molass_rmsd.append(np.nan)
                molass_chiral.append(np.nan)

        results['dgsol_rc'].append(dgsol_rc)
        results['dgsol_mae'].append(dgsol_mae)
        results['dgsol_rmsd'].append(dgsol_rmsd)
        results['dgsol_ss'].append(dgsol_chiral)
        results['lsbuild_rc'].append(lsbuild_recd)
        results['lsbuild_mae'].append(lsbuild_mae)
        results['lsbuild_rmsd'].append(lsbuild_rmsd)
        results['lsbuild_ss'].append(lsbuild_chiral)
        results['molass_rc'].append(molass_recd)
        results['molass_mae'].append(molass_mae)
        results['molass_rmsd'].append(molass_rmsd)
        results['molass_ss'].append(molass_chiral)

    return results


def mae(pred, gt):
    return np.mean(np.abs(pred - gt))


def test():
    vec_dist = g2s.utils.vector_to_square(dists[i])

    maes = []
    for i in range(len(sqr_dists)):
        triu = np.triu_indices(9, k =1)
        maes.append(mae(sqr_dists[i][triu], vec_dist[i][triu]))


def test_scaling():
    alkanes = sorted(glob('/home/lem/github/Graph2Structure/tests/test_files/C*0.xyz'))
    import time
    distances = []
    nuclear_charges = []
    sparse = True
    for alk in alkanes:
        elements, xyz = read_xyz(alk)
        dist = calculate_distances(xyz)
        nc = np.array([g2s.constants.periodic_table.index(e) for e in elements])
        heavy_idx = np.where(nc != 1)[0]
        dist = dist[heavy_idx, :][:, heavy_idx]
        if sparse:
            dist[np.where(dist >= 6.0)] = 0.0
        distances.append(dist)
        nuclear_charges.append(nc[heavy_idx])

    outpath = 'alkane_test'
    for i in range(len(distances)):
        dgsol_start = time.time()
        dgsol = DGSOL(distances[i][None,...], nuclear_charges[i][None,...], vectorized_input=False)
        dgsol.solve_distance_geometry(f'{outpath}/{i}')
        dgsol_end = time.time()
        lsbuild_coords = lsbuild(f'{outpath}/{i}/0000/dgsol.input')['x']
        lsbuild_end = time.time()
        write_xyz(f'{outpath}/{i}/0000/lsbuild.xyz', lsbuild_coords, nuclear_charges[i])
        print('DGSOL', dgsol_end-dgsol_start)
        print('lsbuild', lsbuild_end-dgsol_end)

    # natoms [10, 20, 30, 40, 50]

    # Full
    # DGSOL [0.048, 0.18, 0.42, 1.2, 2.1]
    # lsbuild base [0.035, 1.63, 19.42, 113.24, 436.78]
    # lsbuild r-base [0.05, 0.12, 0.58, 1.8, 4.37]

    # Sparse
    # DGSOL [0.043, 0.16, 0.57, 1.2, 2.2]
    # lsbuild r-base [0.01, 0.04, 0.07, 0.12, 0.19]

    # Boundaries
    # DGSOL [0.05, 0.25, 1.6, 8.46, 15.1]
    # lsbuild r-base [0.15, 0.38, 1.89, 6.41, 13.8]

    import matplotlib.pyplot as plt
    from matplotlib import ticker
    natoms = [10, 20, 30, 40, 50]

    fontsize = 15
    fig, axs = plt.subplots(1, 3, figsize=(15, 7.6),sharey='row')

    axs[0].plot(natoms, [0.048, 0.18, 0.42, 1.2, 2.1], label='Full-DGSOL')
    # axs[0].plot(natoms, [0.035, 1.63, 19.42, 113.24, 436.78], label='Full-lsbuild')
    axs[0].plot(natoms, [0.05, 0.12, 0.58, 1.8, 4.37], label='Full-lsbuild')

    axs[1].plot(natoms, [0.043, 0.16, 0.57, 1.2, 2.2], label='Sparse-DGSOL')
    axs[1].plot(natoms,  [0.01, 0.04, 0.07, 0.12, 0.19], label='Sparse-lsbuild')

    axs[2].plot(natoms, [0.05, 0.25, 1.6, 8.46, 15.1], label='Boundary-DGSOL')
    axs[2].plot(natoms, [0.15, 0.38, 1.89, 6.41, 13.8], label='Boundary-lsbuild')

    for a in axs.flatten():
        a.loglog()
        a.grid(True, which="both", color='#d3d3d3')
        # plt.setp(a.get_yticklabels(), visible=True)
        # a.get_yaxis().set_visible(True)
        a.xaxis.set_major_formatter(ticker.ScalarFormatter())
        a.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        a.xaxis.label.set_fontsize(fontsize)
        a.yaxis.label.set_fontsize(fontsize)
        a.yaxis.set_tick_params(labelsize=fontsize, which='both', length=6)
        a.xaxis.set_tick_params(labelsize=fontsize, which='both', length=6)

        a.set(xlabel='Number of atoms', ylabel='Time to Solution [s]')
        a.legend(prop={'size': fontsize})

    axs[0].set_title('Full Distance Matrix', fontsize=fontsize)
    axs[1].set_title('Sparse Distance Matrix', fontsize=fontsize)
    axs[2].set_title('Boundary Distance Matrix', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('dgsol_lsbuild_performance.png', dpi=300)
    plt.show()




if __name__ == '__main__':
    qmrxn = '/data/lemm/g2s/dgbench/DG_bench_qmrxn.pkl'
    ci = '/data/lemm/g2s/dgbench/DG_bench_ci.pkl'

    predict_all(qmrxn, '/data/lemm/g2s/dgbench/qmrxn_dg')
    predict_all(ci, '/data/lemm/g2s/dgbench/ci_dg')

    # rqmrxn = analyse_results(qmrxn, '/data/lemm/g2s/dgbench/qmrxn_dg')
    # rci = analyse_results(ci, '/data/lemm/g2s/dgbench/ci_dg')
    #
    # pickle.dump(rqmrxn, open('/data/lemm/g2s/dgbench/dgbench_qmrxn_results3.pkl', 'wb'))
    # pickle.dump(rci, open('/data/lemm/g2s/dgbench/dgbench_ci_results3.pkl', 'wb'))
