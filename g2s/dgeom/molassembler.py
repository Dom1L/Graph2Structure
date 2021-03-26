from copy import deepcopy

import numpy as np
from tqdm import tqdm
import scine_molassembler as scm
from scine_utilities import ElementType

from g2s.utils import vector_to_square

scm.Options.chiral_state_preservation = scm.ChiralStatePreservation.Unique#getattr(scm.ChiralStatePreservation, "None")
 #scm.ChiralStatePreservation.RandomFromMultipleBest

scm_elements = {'H': ElementType.H,
                'C': ElementType.C,
                'O': ElementType.O,
                'N': ElementType.N,
                'S': ElementType.S,
                'P': ElementType.P,
                'F': ElementType.F,
                'Cl': ElementType.Cl,
                'Br': ElementType.Br,
                'I': ElementType.I,
                1: ElementType.H,
                6: ElementType.C,
                8: ElementType.O,
                7: ElementType.N,
                16: ElementType.S,
                15: ElementType.P,
                9: ElementType.F,
                17: ElementType.Cl,
                35: ElementType.Br,
                53: ElementType.I}

scm_bonds = {1: scm.BondType.Single,
             2: scm.BondType.Double,
             3: scm.BondType.Triple,
             scm.BondType.Single: 1,
             scm.BondType.Double: 2,
             scm.BondType.Triple: 3, }


class Molassembler:
    def __init__(self, bo, distances, nuclear_charges, vectorized_input=True):
        self.bohr_to_angstrom = 0.529177
        self.distances = vector_to_square(distances) if vectorized_input else distances
        self.bo_mat = bo
        self.nuclear_charges = nuclear_charges
        self.reverse_map = None
        self.scm_graphs = None
        self.coords = None

    def convert_molecules(self):
        self.reverse_map = []
        self.scm_graphs = []
        for b, nc in tqdm(zip(self.bo_mat, self.nuclear_charges), total=len(self.bo_mat)):
            scm_graph, scm_mapping = self.graph_to_scm(b, nc, debug=False)
            self.reverse_map.append(scm_mapping)
            self.scm_graphs.append(scm_graph)

    def scm_config(self):
        conf = scm.dg.Configuration()
        conf.partiality = scm.dg.Partiality.All
        conf.refinement_step_limit = 10000
        conf.refinement_gradient_target = 1e-05
        conf.spatial_model_loosening = 1.5
        return conf

    def solve_distance_geometry(self, n_conformers=1, seed=42):
        gen_coords = []
        tries = []
        for mol_id in tqdm(range(len(self.scm_graphs))):
            r_map = self.reverse_map[mol_id]
            dist = self.distances[mol_id][r_map, :][:, r_map]
            g = self.scm_graphs[mol_id]
            coords = []
            np.random.seed(seed)
            counter = 0
            while len(coords) < n_conformers:
            # coords = np.array([scm.dg.generate_g2s_conformation(g, dist, seed+i) * self.bohr_to_angstrom for i in range(n_conformers)])
                conf = scm.dg.generate_g2s_conformation(g, dist, np.random.randint(1e6), self.scm_config())
                if not isinstance(conf, scm.dg.Error):
                    coords.append(conf)
                counter += 1
                if counter >= 1000:
                    print('Max tries exceeded')
                    coords.append(np.zeros((len(dist), 3)))
            gen_coords.append(np.array(coords)[:, np.argsort(r_map)])
            tries.append(counter)
        self.coords = gen_coords
        self.tries = tries

    @staticmethod
    def graph_to_scm(bo, elements, debug=False):
        graph = scm.Molecule(scm_elements[elements[0]])
        reverse_map = [0]
        fin_atoms = []
        bkp_adj = deepcopy(bo)
        backlog = [0]
        n = 1
        while len(fin_atoms) != len(elements):
            atom_id = backlog[0]
            for i in range(len(elements)):
                if atom_id == i or bkp_adj[atom_id, i] == 0:
                    continue

                bond_type = scm_bonds[bo[atom_id, i]]
                if i not in reverse_map:
                    graph.add_atom(scm_elements[elements[i]], reverse_map.index(atom_id), bond_type)
                    n += 1
                else:
                    graph.add_bond(reverse_map.index(i), reverse_map.index(atom_id), bond_type)

                bkp_adj[atom_id, i] = 0
                bkp_adj[i, atom_id] = 0
                if i not in fin_atoms and i not in backlog:
                    backlog.append(i)
                if i not in reverse_map:
                    reverse_map.append(i)
            backlog.pop(0)
            fin_atoms.append(atom_id)
        if debug:
            return graph, np.array(reverse_map), bkp_adj
        return graph, np.array(reverse_map)

    @staticmethod
    def bo_from_molassembler(graph):
        n_atoms = graph.graph.N

        bo_mat = np.zeros((n_atoms, n_atoms))

        for i, j in graph.graph.bonds():
            bond_type = scm_bonds[graph.graph.bond_type(scm.BondIndex(i, j))]
            bo_mat[i, j] = bond_type
            bo_mat[j, i] = bond_type
        return bo_mat.astype(int)
