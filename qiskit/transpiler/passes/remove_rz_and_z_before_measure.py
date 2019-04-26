# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Transpiler pass to remove RZ and Z gate before a measurement
"""

from qiskit.circuit import Measure
from qiskit.extensions.standard import RZGate, ZGate
from qiskit.transpiler.basepasses import TransformationPass


class RemoveRZandZbeforeMeasure(TransformationPass):
    """Remove RZ and Z gate before a measurement """

    def run(self, dag):
        """Return a new circuit that has been optimized."""
        measures = dag.op_nodes(Measure)
        for measure in measures:
            predecessor = next(dag.predecessors(measure))
            if predecessor.type == 'op' and isinstance(predecessor.op, (RZGate, ZGate)):
                dag.remove_op_node(predecessor)
        return dag
