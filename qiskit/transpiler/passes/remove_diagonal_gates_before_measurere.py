# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Transpiler pass to remove diagonal gates (like RZ, T, Z, etc) before
a measurement. Including diagonal control gates.
"""

from qiskit.circuit import Measure
from qiskit.extensions.standard import RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate, CzGate
from qiskit.transpiler.basepasses import TransformationPass


class RemoveDiagonalGatesBeforeMeasure(TransformationPass):
    """Remove diagonal gates (like RZ, T, Z, etc) before a measurement.
    Including diagonal control gates."""

    def run(self, dag):
        """Return a new circuit that has been optimized."""
        diagonal_gates = (RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate)
        diagonal_control_gates = (CzGate)

        nodes_to_remove = set()
        for measure in dag.op_nodes(Measure):
            for predecessor in dag.predecessors(measure):

                if predecessor.type == 'op' and isinstance(predecessor.op, diagonal_gates):
                    nodes_to_remove.add(predecessor)

                if predecessor.type == 'op' and isinstance(predecessor.op, diagonal_control_gates):
                    successors = dag.successors(predecessor)
                    if all([s.type == 'op' and isinstance(s.op, Measure) for s in successors]):
                        nodes_to_remove.add(predecessor)

        for node_to_remove in nodes_to_remove:
            dag.remove_op_node(node_to_remove)

        return dag
