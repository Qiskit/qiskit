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

"""Evaluate how good the layout selection was.

No CX direction is considered.
Saves in `property_set['layout_score']` the sum of distances for each circuit CX.
The lower the number, the better the selection.
Therefore, 0 is a perfect layout selection.
"""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.converters import circuit_to_dag
from qiskit.circuit.controlflow import IfElseOp, WhileLoopOp, ForLoopOp


class Layout2qDistance(AnalysisPass):
    """Evaluate how good the layout selection was.

    Saves in `property_set['layout_score']` (or the property name in property_name)
    the sum of distances for each circuit CX.
    The lower the number, the better the selection. Therefore, 0 is a perfect layout selection.
    No CX direction is considered.
    """

    def __init__(self, coupling_map, property_name="layout_score", weight_loops=True):
        """Layout2qDistance initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            property_name (str): The property name to save the score. Default: layout_score
            weight_loops (bool): Whether to weight 2q distance of loops by number of loops. If
               while loop exists the returned distance will be NaN.
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.property_name = property_name
        self.weight_loops = weight_loops

    def run(self, dag):
        """
        Run the Layout2qDistance pass on `dag`.
        Args:
            dag (DAGCircuit): DAG to evaluate.
        """
        layout = self.property_set["layout"]

        if layout is None:
            return

        if self.coupling_map is None or len(self.coupling_map.graph) == 0:
            self.property_set[self.property_name] = 0
            return

        self.coupling_map.compute_distance_matrix()

        sum_distance = 0

        virtual_physical_map = layout.get_virtual_bits()
        dist_matrix = self.coupling_map.distance_matrix
        breakpoint()
        for gate in dag.two_qubit_ops():
            physical_q0 = virtual_physical_map[gate.qargs[0]]
            physical_q1 = virtual_physical_map[gate.qargs[1]]

            sum_distance += dist_matrix[physical_q0, physical_q1] - 1
        for node in dag.control_flow_ops():
            sum_distance += self._control_flow_2q_distance(node.op)

        self.property_set[self.property_name] = sum_distance


    def _control_flow_2q_distance(self, cfop):
        distance = 0
        if self.weight_loops:
            if isinstance(cfop, IfElseOp):
                for block in cfop.blocks:
                    _pass = Layout2qDistance(self.coupling_map,
                                             property_name=self.property_name)
                    dag_block = circuit_to_dag(block)
                    _pass.run(dag_block)
                    breakpoint()
                    distance += _pass.property_set[self.property_name]
            elif isinstance(cfop, ForLoopOp):
                index_set, _, block = cfop.params
                _pass = Layout2qDistance(self.coupling_map,
                                         property_name=self.property_name)
                dag_block = circuit_to_dag(block)
                _pass.run(dag_block)
                distance += len(index_set) * _pass.property_set[self.property_name]
            elif isinstance(cfop, WhileLoopOp):
                body = cfop.blocks[0]
                if body.num_nonlocal_gates():
                    # Indeterminate number of loops so indeterminate 2q distance
                    import math
                    distance = math.nan
        else:
            for block in cfop.blocks:
                _pass = Layout2qDistance(self.coupling_map,
                                         property_name=self.property_name)
                dag_block = circuit_to_dag(block)
                _pass.run(dag_block)
        return distance
        
