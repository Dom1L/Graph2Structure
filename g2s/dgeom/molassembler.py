from copy import deepcopy

import numpy as np
import scine_molassembler as scim
import scine_utilities as scim_utils


convert_types = {'H': scim_utils.ElementType.H,
                 'C': scim_utils.ElementType.C,
                 'O': scim_utils.ElementType.O,
                 'N': scim_utils.ElementType.N,
                 1: scim.BondType.Single,
                 2: scim.BondType.Double,
                 3: scim.BondType.Triple,
                 scim.BondType.Single: 1,
                 scim.BondType.Double: 2,
                 scim.BondType.Triple: 3,
                 }


def graph_to_molass(bo, elements, debug=False):
    graph = scim.Molecule(convert_types[elements[0]])
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

            bond_type = convert_types[bo[atom_id, i]]
            if i not in reverse_map:
                graph.add_atom(convert_types[elements[i]], reverse_map.index(atom_id), bond_type)
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


def bo_from_molassembler(graph):
    n_atoms = graph.graph.N

    bo_mat = np.zeros((n_atoms, n_atoms))

    for i, j in graph.graph.bonds():
        bond_type = convert_types[graph.graph.bond_type(scim.BondIndex(i, j))]
        bo_mat[i, j] = bond_type
        bo_mat[j, i] = bond_type
    return bo_mat.astype(int)
