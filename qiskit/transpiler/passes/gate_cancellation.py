# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Pass for cancelling self-inverse gates/rotations. The cancellation utilizes the commutation
relations in the circuit. Gates considered include Hadamard, CNOT, X, Y, Z, and Rx/Ry/Rz.
"""

from collections import defaultdict

import networkx as nx
import sympy
from sympy import Number as N

from qiskit.mapper import MapperError
from qiskit.extensions.standard.u1 import U1Gate

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.commutation_analysis import CommutationAnalysis


class GateCancellation(TransformationPass):

    """
    Transformation pass that cancels the redundant
    (self-adjoint) gates through commutation relations
    """

    def __init__(self):
        super().__init__()
        self.requires.append(CommutationAnalysis())

    def run(self, dag):

        """

        Args:
            dag (Dagcircuit): the DAG to be optimized.

        Returns:
            Dagcircuit: the optimized DAG.

        Raises:
            MapperError: when the 1 qubit rotation gates are not found
        """

        # pylint: disable=too-many-locals
        # pylint: disable=too-many-nested-blocks
        # pylint: disable=too-many-branches

        q_gate_list = ['cx', 'cy', 'cz', 'h', 'x', 'y', 'z', 't', 's']

        # Gate sets to be cancelled
        cancellation_sets = defaultdict(lambda: [])

        for wire in dag.wires:

            wire_name = "{0}[{1}]".format(str(wire[0].name), str(wire[1]))
            wire_commutation_set = self.property_set['commutation_set'][wire_name]

            for set_num, com_set in enumerate(wire_commutation_set):

                if dag.multi_graph.node[com_set[0]]['type'] in ['in', 'out']:
                    continue

                for node in com_set:

                    gate_name = dag.multi_graph.node[node]['name']
                    gate_ops = dag.multi_graph.node[node]['qargs']

                    op_num = len(gate_ops)

                    if op_num == 1 and gate_name in q_gate_list:
                        cancellation_sets[(gate_name, wire_name, set_num)].append(node)

                    if op_num == 1 and gate_name in ['u1', 'rz']:

                        cancellation_sets[('z_rotation', wire_name, set_num)].append(node)

                    elif op_num == 2 and gate_ops[0] == wire:

                        second_op_name = "{0}[{1}]".format(str(gate_ops[1][0].name),
                                                           str(gate_ops[1][1]))
                        q2_key = (gate_name, wire_name, second_op_name,
                                  self.property_set['commutation_set'][(node, second_op_name)])

                        cancellation_sets[q2_key].append(node)

        for cancel_set_key in cancellation_sets:

            set_len = len(cancellation_sets[cancel_set_key])

            if ((set_len) > 1
                    and cancel_set_key[0] in q_gate_list):

                gates_to_cancel = cancellation_sets[cancel_set_key]

                for c_node in gates_to_cancel[:(set_len // 2) * 2]:
                    dag._remove_op_node(c_node)

            elif((set_len) > 1
                 and cancel_set_key[0] == 'z_rotation'):

                run = cancellation_sets[cancel_set_key]
                r_qarg = dag.multi_graph.node[run[0]]["qargs"][0]
                total_angle = N(0)  # lambda
                for current_node in run:
                    node = dag.multi_graph.node[current_node]
                    if (node["condition"] is not None
                            or len(node["qargs"]) != 1
                            or node["qargs"][0] != r_qarg):
                        raise MapperError("internal error")
                    current_angle = node["op"].param[0]
                    # Compose gates
                    total_angle = current_angle + total_angle

                    # Simplify the symbolic parameters
                    total_angle = sympy.simplify(total_angle)
                # Replace the data of the first node in the run
                new_op = U1Gate(total_angle, r_qarg)

                nx.set_node_attributes(dag.multi_graph, name='name',
                                       values={run[0]: 'u1'})
                nx.set_node_attributes(dag.multi_graph, name='op',
                                       values={run[0]: new_op})

                # Delete the other nodes in the run
                for current_node in run[1:]:
                    dag._remove_op_node(current_node)

        return dag
