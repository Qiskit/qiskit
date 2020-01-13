# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Analysis pass to find commutation relations between DAG nodes."""

from collections import defaultdict
import numpy as np
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.quantum_info.operators import Operator

_CUTOFF_PRECISION = 1E-10


class CommutationAnalysis(AnalysisPass):
    """Analysis pass to find commutation relations between DAG nodes.

    Property_set['commutation_set'] is a dictionary that describes
    the commutation relations on a given wire, all the gates on a wire
    are grouped into a set of gates that commute.

    TODO: the current pass determines commutativity through matrix multiplication.
    A rule-based analysis would be potentially faster, but more limited.
    """

    def __init__(self):
        super().__init__()
        self.gates_on_wire = {}

    def run(self, dag):
        """Run the CommutationAnalysis pass on `dag`.

        Run the pass on the DAG, and write the discovered commutation relations
        into the property_set.
        """
        # Initiate the commutation set
        self.property_set['commutation_set'] = defaultdict(list)

        # Build a dictionary to keep track of the gates on each qubit
        # The key with format (wire_name) will store the lists of commutation sets
        # The key with format (node, wire_name) will store the index of the commutation set
        # on the wire with wire_name, thus, for example:
        # self.property_set['commutation_set'][wire_name][(node, wire_name)] will give the
        # commutation set that contains node.

        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire.register.name), str(wire.index))
            self.property_set['commutation_set'][wire_name] = []

        # Add edges to the dictionary for each qubit
        for node in dag.topological_op_nodes():
            for (_, _, edge_data) in dag.edges(node):

                edge_name = edge_data['name']
                self.property_set['commutation_set'][(node, edge_name)] = -1

        # Construct the commutation set
        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire.register.name), str(wire.index))

            for current_gate in dag.nodes_on_wire(wire):

                current_comm_set = self.property_set['commutation_set'][wire_name]
                if not current_comm_set:
                    current_comm_set.append([current_gate])

                if current_gate not in current_comm_set[-1]:
                    prev_gate = current_comm_set[-1][-1]
                    does_commute = False
                    try:
                        does_commute = _commute(current_gate, prev_gate)
                    except TranspilerError:
                        pass
                    if does_commute:
                        current_comm_set[-1].append(current_gate)

                    else:
                        current_comm_set.append([current_gate])

                temp_len = len(current_comm_set)
                self.property_set['commutation_set'][(current_gate, wire_name)] = temp_len - 1


def _commute(node1, node2):

    if node1.type != "op" or node2.type != "op":
        return False

    if any([nd.name in {"barrier", "snapshot", "measure", "reset", "copy"}
            for nd in [node1, node2]]):
        return False

    if node1.condition or node2.condition:
        return False

    if node1.op.is_parameterized() or node2.op.is_parameterized():
        return False

    qarg = list(set(node1.qargs + node2.qargs))
    qbit_num = len(qarg)

    qarg1 = [qarg.index(q) for q in node1.qargs]
    qarg2 = [qarg.index(q) for q in node2.qargs]

    id_op = Operator(np.eye(2 ** qbit_num))

    op12 = id_op.compose(node1.op, qargs=qarg1).compose(node2.op, qargs=qarg2)
    op21 = id_op.compose(node2.op, qargs=qarg2).compose(node1.op, qargs=qarg1)

    if_commute = (op12 == op21)

    return if_commute
