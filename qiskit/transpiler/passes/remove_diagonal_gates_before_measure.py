# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Transpiler pass to remove diagonal gates (like RZ, T, Z, etc) before
a measurement. Including diagonal 2Q gates.
"""

from qiskit.circuit import Measure
from qiskit.extensions.standard import RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate,\
    CzGate, CrzGate, Cu1Gate, RZZGate
from qiskit.transpiler.basepasses import TransformationPass


class RemoveDiagonalGatesBeforeMeasure(TransformationPass):
    """Remove diagonal gates (like RZ, T, Z, etc) before a measurement.
    Including diagonal 2Q gates."""

    def run(self, dag):
        """Return a new circuit that has been optimized."""
        diagonal_1q_gates = (RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate)
        diagonal_2q_gates = (CzGate, CrzGate, Cu1Gate, RZZGate)

        nodes_to_remove = set()
        for measure in dag.op_nodes(Measure):
            predecessor = dag.quantum_predecessors(measure)[0]

            if predecessor.type == 'op' and isinstance(predecessor.op, diagonal_1q_gates):
                nodes_to_remove.add(predecessor)

            if predecessor.type == 'op' and isinstance(predecessor.op, diagonal_2q_gates):
                successors = dag.quantum_successors(predecessor)
                if all([s.type == 'op' and isinstance(s.op, Measure) for s in successors]):
                    nodes_to_remove.add(predecessor)

        for node_to_remove in nodes_to_remove:
            dag.remove_op_node(node_to_remove)

        return dag
