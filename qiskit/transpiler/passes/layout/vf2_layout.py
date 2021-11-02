# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""VF2Layout pass to find a layout using subgraph isomorphism"""
import random
from retworkx import PyGraph, PyDiGraph, vf2_mapping
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


class VF2Layout(AnalysisPass):
    """A pass for choosing a Layout of a circuit onto a Coupling graph, as a
    a subgraph isomorphism problem, solved by VF2++.

    If a solution is found that means there is a "perfect layout" and that no
    further swap mapping or routing is needed. If a solution is found the layout
    will be set in the property set as ``property_set['layout']``. However, if no
    solution is found, no ``property_set['layout']`` is set. The stopping reason is
    set in ``property_set['VF2Layout_stop_reason']`` in all the cases and will be
    one of the following values:

        * ``"solution found"``: If a perfect layout was found.
        * ``"nonexistent solution"``: If no perfect layout was found.

    """

    def __init__(self, coupling_map, strict_direction=False, seed=None):
        """Initialize a ``VF2Layout`` pass instance

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            strict_direction (bool): If True, considers the direction of the coupling map.
                                     Default is False.
            seed (int): Sets the seed of the PRNG. -1 Means no node shuffling.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.strict_direction = strict_direction
        self.seed = seed

    def run(self, dag):
        """run the layout method"""
        qubits = dag.qubits
        qubit_indices = {qubit: index for index, qubit in enumerate(qubits)}

        interactions = []
        for node in dag.op_nodes(include_directives=False):
            len_args = len(node.qargs)
            if len_args == 2:
                interactions.append((qubit_indices[node.qargs[0]], qubit_indices[node.qargs[1]]))
            if len_args >= 3:
                raise TranspilerError(
                    "VF2Layout only can handle 2-qubit gates or less. Node "
                    f"{node.name} ({node}) is {len_args}-qubit"
                )

        if self.strict_direction:
            cm_graph = self.coupling_map.graph
            im_graph = PyDiGraph(multigraph=False)
        else:
            cm_graph = self.coupling_map.graph.to_undirected()
            im_graph = PyGraph(multigraph=False)

        cm_nodes = list(cm_graph.node_indexes())
        if self.seed != -1:
            random.Random(self.seed).shuffle(cm_nodes)
            shuffled_cm_graph = type(cm_graph)()
            shuffled_cm_graph.add_nodes_from(cm_nodes)
            new_edges = [(cm_nodes[edge[0]], cm_nodes[edge[1]]) for edge in cm_graph.edge_list()]
            shuffled_cm_graph.add_edges_from_no_data(new_edges)
            cm_nodes = [k for k, v in sorted(enumerate(cm_nodes), key=lambda item: item[1])]
            cm_graph = shuffled_cm_graph
        im_graph.add_nodes_from(range(len(qubits)))
        im_graph.add_edges_from_no_data(interactions)

        mappings = vf2_mapping(cm_graph, im_graph, subgraph=True, id_order=False, induced=False)
        try:
            mapping = next(mappings)
            stop_reason = "solution found"
            layout = Layout({qubits[im_i]: cm_nodes[cm_i] for cm_i, im_i in mapping.items()})
            self.property_set["layout"] = layout
            for reg in dag.qregs.values():
                self.property_set["layout"].add_register(reg)
        except StopIteration:
            stop_reason = "nonexistent solution"

        self.property_set["VF2Layout_stop_reason"] = stop_reason
