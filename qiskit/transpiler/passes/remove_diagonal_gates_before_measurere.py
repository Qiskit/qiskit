# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Transpiler pass to remove diagonal gates (like RZ, T, Z, etc) before a measurement
"""

from qiskit.circuit import Measure
from qiskit.extensions.standard import RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate
from qiskit.transpiler.basepasses import TransformationPass


class RemoveDiagonalGatesBeforeMeasure(TransformationPass):
    """Remove diagonal gates (like RZ, T, Z, etc) before a measurement"""

    def run(self, dag):
        """Return a new circuit that has been optimized."""
        diagonal_gates = (RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate)

        measures = dag.op_nodes(Measure)
        for measure in measures:
            for predecessor in dag.predecessors(measure):
                if predecessor.type == 'op' and isinstance(predecessor.op, diagonal_gates):
                    dag.remove_op_node(predecessor)
        return dag
