# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Evaluate a score for the layout.

Saves in `property_set['layout_score']` a measure for the score, i.e. fidelity of the layout.
The closer the number to one, the better the selection. Therefore, 1 is a perfect error-free layout.
No CX direction is considered.
"""
import numpy as np
from itertools import combinations

from qiskit.circuit import Gate
from qiskit.transpiler.basepasses import AnalysisPass


DEFAULT_CX_ERROR = 5 * 10**(-2)


class LayoutScorer(AnalysisPass):
    """Evaluate a score for a layout.

    Saves in `property_set['layout_score']` (or the property name in property_name) a measure for the score,
    i.e. fidelity of the layout. Alternatively, it returns the score. The closer the number to one, the better
    the selection. Therefore, 1 is a perfect error-free layout. No CX direction is considered.
    """
    def __init__(self, coupling_map, backend_properties=None, property_name='layout_score'):
        """LayoutScorer initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            backend_prop (BackendProperties): Properties of the used backend.
            property_name (str): The property name to save the score. Default: layout_score
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.backend_properties = backend_properties
        self.property_name = property_name

    def run(self, dag):
        """
        Run the LayoutScorer pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to evaluate.
        """
        layout = self.property_set["layout"]
        if layout is None:
            return
        layout_score = self.evaluate(dag, layout)
        self.property_set[self.property_name] = layout_score

    def evaluate(self, dag, layout):
        """ Evaluate the score on a layout and dag. 
        Calculate the score as the product of all two qubit gate fidelities.
        Assign an artificial fidelity to virtual gates that require swap 
        operations in the implementation.

        Args:
            dag (DAGCircuit): DAG to evaluate
            layout (Layout): Layout to evaluate
        Return:
            layout_fidelity (float): The score of the layout
        """
        layout_fidelity = 1.0
        for node in dag.two_qubit_ops():
            physical_qubits = [layout[qubit] for qubit in node.qargs]
            if isinstance(node.op, Gate):
                layout_fidelity *= self._calculate_2q_fidelity(physical_qubits)
        return layout_fidelity

    def _calculate_2q_fidelity(self, qubits):
        """Calculate the 2q fidelity
        Depending on the distance of the qubits, there are different options
        for introducing the additional swaps. Therefore the average over all paths is used.
        See also Arxiv 2103.15695 on page 5 as a reference.

        As an example the fidelity for a cx-gate between qb1 and qb4 in a chain is given as:
        f_14 = 1/3 * f_12 f_23 f_34 (f_12^5 f_23^5 + f_23^5 f_34^5 f_12^5 f_34^5)
        """
        def cx_fid(qubits):
            if self.backend_properties:
                return 1 - self.backend_properties.gate_error("cx", qubits)
            else:
                return 1 - DEFAULT_CX_ERROR

        cplpath = self.coupling_map.shortest_undirected_path(*qubits)
        cplpath_length = len(cplpath) - 1
        cplpath_edges = [[cplpath[k], cplpath[k+1]] for k in range(cplpath_length)]

        path_fid = 1.0
        path_fid *= np.sum([np.prod([cx_fid(qubits)**5 for qubits in qubits_subset])
            for qubits_subset in combinations(cplpath_edges, cplpath_length-1)])
        path_fid *= np.prod([cx_fid(qubits) for qubits in cplpath_edges]) / cplpath_length

        return path_fid
