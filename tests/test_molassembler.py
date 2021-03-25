from glob import glob

from tqdm import tqdm
import numpy as np

from g2s.constants import periodic_table
from g2s.graph_extractor import xyz2mol_graph
from g2s.dgeom.molassembler import graph_to_molass, bo_from_molassembler
from g2s.utils import calculate_distances, write_xyz

import scine_molassembler as scm

def test_conversion(filepath):
    adj, nuclear_charges, coords = xyz2mol_graph(filepath)
    elements = [periodic_table[nc] for nc in nuclear_charges]
    graph, reverse_map, bkp_adj = graph_to_molass(adj, elements, debug=True)
    assert bkp_adj.sum() == 0.

    reverse_map = np.array(reverse_map)
    m_bo = bo_from_molassembler(graph)
    assert np.sum(m_bo - adj[reverse_map, :][:, reverse_map]) == 0.


def test_distance_bounds():
    adj, nuclear_charges, coords = xyz2mol_graph(test_mols[0])
    elements = [periodic_table[nc] for nc in nuclear_charges]
    graph, reverse_map, bkp_adj = graph_to_molass(adj, elements, debug=True)
    distances = calculate_distances(coords)

    new_coords = scm.dg.generate_g2s_conformation(graph, distances[reverse_map, :][:, reverse_map], 13)*0.529177



def test_molassembler(filepaths):
    bo_mats = []
    nuclear_charges = []
    ci_coords = []
    ci_dists = []
    for f in tqdm(filepaths):
        b, nc, coords = xyz2mol_graph(f)
        bo_mats.append(b)
        nuclear_charges.append(nc)
        ci_coords.append(coords)
        ci_dists.append(calculate_distances(coords))

    configuration = {'partiality':scm.dg.Partiality.FourAtom, 'refinement_step_limit':10000,
                     'refinement_gradient_target':1e-5, 'spatial_model_loosening':1.0, 'fixed_positions':[]}

    molass = Molassembler(bo_mats, ci_dists, nuclear_charges, vectorized_input=False)
    molass.convert_molecules()
    molass.solve_distance_geometry()



if __name__ == '__main__':
    test_mols = sorted(glob('/data/lemm/g2s/const_isomers/*.xyz'))

    for tm in tqdm(test_mols):
        test_conversion(tm)