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
from qiskit.dagcircuit import DAGCircuit


class OptimizeSwapBeforeMeasure(TransformationPass):
    """Remove the swaps followed measurement (and adapts the measurement)"""

    def run(self, dag):
        """Return a new circuit that has been optimized."""
        swaps = dag.op_nodes(SwapGate)
        for swap in swaps:
            final_successor = []
            for successor in dag.successors(swap):
                node = dag.multi_graph.node[successor]
                final_successor.append(
                    node['type'] == 'out' or node['type'] == 'op' and node['op'].name == 'measure')
            if all(final_successor):
                # the node swap needs to be removed and, if a meassure follows, needs to be adapted
                swap_qargs = dag.multi_graph.node[swap]['qargs']
                measure_layer = DAGCircuit()
                for qreg in dag.qregs.values():
                    measure_layer.add_qreg(qreg)
                for creg in dag.cregs.values():
                    measure_layer.add_creg(creg)
                for successor in dag.successors(swap):
                    node = dag.multi_graph.node[successor]
                    if node['type'] == 'op' and node['op'].name == 'measure':
                        # replace measure node with a new one, where qargs is set with the "other"
                        # swap qarg.
                        dag._remove_op_node(successor)
                        old_measure_qarg = node['qargs'][0]
                        new_measure_qarg = swap_qargs[swap_qargs.index(old_measure_qarg) - 1]
                        measure_layer.apply_operation_back(
                            Measure(qubit=new_measure_qarg, bit=node['cargs'][0]))
                dag.extend_back(measure_layer)
                dag._remove_op_node(swap)
        return dag
