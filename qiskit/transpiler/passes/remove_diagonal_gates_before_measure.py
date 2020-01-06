# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Remove diagonal gates (including diagonal 2Q gates) before a measurement."""

from qiskit.circuit import Measure
from qiskit.extensions.standard import RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate,\
    CzGate, CrzGate, Cu1Gate, RZZGate
from qiskit.transpiler.basepasses import TransformationPass


class RemoveDiagonalGatesBeforeMeasure(TransformationPass):
    """Remove diagonal gates (including diagonal 2Q gates) before a measurement.

    Transpiler pass to remove diagonal gates (like RZ, T, Z, etc) before
    a measurement. Including diagonal 2Q gates.
    """

    def run(self, dag):
        """Run the RemoveDiagonalGatesBeforeMeasure pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        diagonal_1q_gates = (RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate)
        diagonal_2q_gates = (CzGate, CrzGate, Cu1Gate, RZZGate)

        nodes_to_remove = set()
        for measure in dag.op_nodes(Measure):
            predecessor = next(dag.quantum_predecessors(measure))

            if predecessor.type == 'op' and isinstance(predecessor.op, diagonal_1q_gates):
                nodes_to_remove.add(predecessor)

            if predecessor.type == 'op' and isinstance(predecessor.op, diagonal_2q_gates):
                successors = dag.quantum_successors(predecessor)
                if all([s.type == 'op' and isinstance(s.op, Measure) for s in successors]):
                    nodes_to_remove.add(predecessor)

        for node_to_remove in nodes_to_remove:
            dag.remove_op_node(node_to_remove)

        return dag
