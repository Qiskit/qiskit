# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Transpiler pass to remove the swaps in front of measurments by moving the reading qubit
 of the measure intruction.
"""

from qiskit.circuit import Measure
from qiskit.extensions.standard import SwapGate
from qiskit.transpiler._basepasses import TransformationPass

from qiskit.tools.visualization import dag_drawer


class OptimizeSwapBeforeMeasure(TransformationPass):
    """Remove the swaps followed measurement (and adapts the measurement)"""

    def run(self, dag):
        """Return a new circuit that has been optimized."""
        swaps_to_remove = []
        swaps = dag.op_nodes(SwapGate)
        for swap in swaps:
            after_swap = dag.successors(swap)
            dag_drawer(dag)
            successor_one = dag.multi_graph.node[next(after_swap)]
            successor_two = dag.multi_graph.node[next(after_swap)]
            if (successor_one['type'] == 'out' or successor_one['type'] == 'op' and successor_one['op'].name == 'measure') and (successor_two['type'] == 'out' or successor_two['type'] == 'op' and successor_two['op'].name == 'measure'):
                swaps_to_remove.append(swap)
        print(swaps_to_remove)
        return dag
